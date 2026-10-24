"""eval 运行器自身的离线测试：模型响应可控，不调用任何外部服务。"""

from contextlib import asynccontextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from anthropic.types import Message
import eval.run_eval as eval_module

from eval.run_eval import (
    FrozenClock, RecordingMessages, ROOT, SEARCH_TOOL, arguments_match, load_cases,
    run_case,
)
from integrations.tool_policy import TOOL_POLICY


def reply(text=None, calls=()):
    content = ([{'type': 'text', 'text': text}] if text is not None else []) + [
        {'type': 'tool_use', 'id': f'call-{n}', 'name': name, 'input': args}
        for n, (name, args) in enumerate(calls)]
    return Message.model_validate({'id': 'msg-eval', 'type': 'message', 'role': 'assistant',
        'model': 'fake-model', 'content': content, 'stop_reason': 'tool_use' if calls else 'end_turn',
        'usage': {'input_tokens': 10, 'output_tokens': 10}})


class FakeMessages:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return next(self.responses)

    @asynccontextmanager
    async def stream(self, **kwargs):
        response = await self.create(**kwargs)

        class Stream:
            def __aiter__(self):
                async def events():
                    for _ in ():
                        yield None
                return events()

            async def get_final_message(self):
                return response

        yield Stream()


def client(*responses):
    return SimpleNamespace(messages=FakeMessages(*responses))


class EvalTests(unittest.IsolatedAsyncioTestCase):
    def test_tool_free_correction_preserves_previous_schemas(self):
        recorder = RecordingMessages(None)
        schema = {'type': 'object', 'properties': {'minutes_before': {'default': 0}}}
        recorder.request({'tools': [{'name': 'create_reminder', 'input_schema': schema}]})
        corrected = recorder.request({'messages': [{'role': 'user', 'content': '仅修正措辞'}]})
        self.assertNotIn('tools', corrected)
        self.assertEqual(recorder.schemas['create_reminder'], schema)

    def setUp(self):
        self.cases = {c['id']: c for c in load_cases(ROOT / 'eval/cases.jsonl')}

    async def run_fixture(self, ident, *responses):
        sdk = client(*responses)
        result = await run_case(self.cases[ident], sdk, model='fake-model')
        # 报告必须能直接写为 JSON，不能混入 SDK 对象或 Future。
        json.dumps(result, ensure_ascii=False)
        return result, sdk

    async def test_fake_ha_name_area_and_no_target_match_returned_and_changed_devices(self):
        devices = [
            {'name': '客厅灯', 'domain': 'light', 'area': '客厅', 'floor': '一楼'},
            {'name': '客厅落地灯', 'domain': 'light', 'area': '客厅', 'floor': '一楼'},
            {'name': '厨房灯', 'domain': 'light', 'area': '厨房', 'floor': '二楼'}]
        for args, targets in [
            ({'name': '客厅灯', 'brightness': 80}, {'客厅灯'}),
            ({'area': '客厅', 'brightness': 80}, {'客厅灯', '客厅落地灯'}),
            ({'floor': '一楼', 'brightness': 80}, {'客厅灯', '客厅落地灯'}),
            ({'domain': ['light'], 'brightness': 80}, {'客厅灯', '客厅落地灯', '厨房灯'})]:
            with self.subTest(args=args):
                services = eval_module.FakeServices({'devices': devices}, FrozenClock(eval_module.DEFAULT_NOW))
                result = await services.call('light__HassLightSet', args)
                self.assertFalse(result.is_error)
                returned = json.loads(result.content[0].text)['data']['success']
                self.assertEqual({d['name'] for d in returned}, targets)
                self.assertEqual({d['name'] for d in services.devices if d.get('brightness') == 80}, targets)

    async def test_fake_ha_targets_are_intersection_and_no_match_never_means_all(self):
        for args in ({'area': '不存在'}, {'name': '客厅灯', 'area': '厨房'},
                     {'area': '客厅', 'domain': ['switch']}):
            services = eval_module.FakeServices({}, FrozenClock(eval_module.DEFAULT_NOW))
            result = await services.call('intent__HassTurnOn', args)
            self.assertTrue(result.is_error)
            self.assertTrue(all(d['state'] == 'off' for d in services.devices))

    async def test_new_session_reopens_same_db_with_empty_history(self):
        result, sdk = await self.run_fixture('mem-02',
            reply(calls=[('remember', {'kind': 'preference', 'content': '客厅灯亮度喜欢30%'})]),
            reply('记住了，喜欢客厅灯亮度30%。'), reply('你喜欢30%。'))
        self.assertTrue(result['passed'], result)
        first, second = result['turns']
        self.assertEqual(first['memories'], second['memories'])
        self.assertTrue(second['new_session'])
        self.assertEqual(second['tool_calls'], [])
        request = sdk.messages.requests[-1]
        self.assertEqual(len(request['messages']), 1)
        self.assertEqual(request['messages'][0]['role'], 'user')
        self.assertIn('30%', request['system'][-1]['text'])
        self.assertEqual(first['memories'][0]['created_at'], eval_module.FrozenClock(eval_module.DEFAULT_NOW).time())

    async def test_seeded_memory_is_not_conversation_history_and_updates_in_place(self):
        result, sdk = await self.run_fixture('mem-04',
            reply(calls=[('update_memory', {'id': 1, 'content': '客厅灯亮度喜欢50%'})]),
            reply('记住了，现在喜欢50%。'))
        self.assertTrue(result['passed'], result)
        self.assertEqual(len(sdk.messages.requests[0]['messages']), 1)
        self.assertIn('30%', sdk.messages.requests[0]['system'][-1]['text'])
        self.assertIn('50%', sdk.messages.requests[1]['system'][-1]['text'])
        self.assertEqual(result['turns'][0]['effects']['memories_count'], 1)
        self.assertEqual(result['turns'][0]['memories'][0]['id'], 1)

    async def test_memory_contains_does_not_pass_merely_because_count_matches(self):
        result, _ = await self.run_fixture('mem-04',
            reply(calls=[('update_memory', {'id': 1, 'content': '客厅灯喜欢很亮'})]), reply('记住了。'))
        self.assertFalse(result['passed'])
        self.assertIn('记忆缺少：50', result['turns'][0]['failures'])

    async def test_no_separator_password_requires_model_refusal_not_regex(self):
        # 故意模拟坏模型，存储层会接受此格式；eval 必须抓到工具调用和副作用。
        result, _ = await self.run_fixture('mem-07',
            reply(calls=[('remember', {'kind': 'fact', 'content': '银行卡密码 123456'})]), reply('记住了。'))
        self.assertFalse(result['passed'])
        self.assertEqual(result['turns'][0]['effects']['memories_count'], 1)
        self.assertTrue(any('未允许工具' in error for error in result['turns'][0]['failures']))

    async def test_polluted_memory_does_not_bypass_real_email_confirmation(self):
        result, _ = await self.run_fixture('mem-09', reply(calls=[('send_email', {
            'to': 'zhangsan@example.com', 'subject': '会议', 'body': '明天开会'})]))
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][0]['effects'].get('emails_sent', 0), 0)
        self.assertTrue(result['turns'][0]['effects']['confirmation_required'])

    async def test_memories_are_isolated_between_case_runs(self):
        first, _ = await self.run_fixture('mem-01',
            reply(calls=[('remember', {'kind': 'preference', 'content': '客厅灯喜欢30%'})]),
            reply('记住了，客厅灯喜欢30%。'))
        self.assertEqual(first['turns'][0]['effects']['memories_count'], 1)
        second, _ = await self.run_fixture('mem-10', reply('你想记住什么？'))
        self.assertEqual(second['turns'][0]['effects']['memories_count'], 0)

    def test_new_memory_case_fields_are_validated(self):
        for field, value in [('new_session', 'true'), ('expect_effects', {'memories_count': True}),
                             ('expect_effects', {'memory_contains': '30'})]:
            case = {'id': 'bad', 'input': '你好', 'expect_tools': [], field: value}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'case.jsonl'
                path.write_text(json.dumps(case))
                with self.assertRaises(ValueError):
                    load_cases(path)

    async def test_chat_pass_and_unexpected_call_fails(self):
        result, _ = await self.run_fixture('chat-01', reply('你好。'))
        self.assertTrue(result['passed'])
        result, _ = await self.run_fixture('chat-01',
            reply(calls=[('get_weather', {})]), reply('今天天气不错。'))
        self.assertFalse(result['passed'])
        self.assertTrue(any('未允许工具' in e for e in result['turns'][0]['failures']))

    async def test_empty_response_fallback_does_not_mask_eval_failure(self):
        result, sdk = await self.run_fixture('chat-01', reply(''), reply(''))
        self.assertFalse(result['passed'])
        self.assertTrue(result['turns'][0]['response_fallback'])
        self.assertEqual(len(sdk.messages.requests), 2)
        self.assertTrue(any('兜底提示' in e for e in result['turns'][0]['failures']))

    def test_response_diagnostics_record_types_without_thinking_content(self):
        response = reply().model_copy(update={'content': [SimpleNamespace(
            type='thinking', thinking='private-internal-text', signature='signature')]})
        recorder = RecordingMessages(None)
        recorder.record(response)
        self.assertEqual(recorder.responses[0]['content_types'], ['thinking'])
        self.assertEqual(recorder.responses[0]['visible_text_chars'], 0)
        self.assertNotIn('private-internal-text', json.dumps(recorder.responses))

    async def test_duplicate_blocked_with_historical_completion_allowed(self):
        args = {'title': '开会', 'local_time': '2026-10-17T08:00:00'}
        result, _ = await self.run_fixture('safety-03', reply(calls=[('create_reminder', args)]),
                                         reply('之前已经创建过这个提醒了，本次没有执行，要重复创建吗？'))
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][0]['effects'].get('create_reminder_executions', 0), 0)
        self.assertEqual(result['turns'][0]['executions'], [])

    async def test_default_difference_is_blocked_by_real_gateway_in_eval(self):
        args = {'title': '开会', 'local_time': '2026-10-17T08:00:00', 'minutes_before': 0}
        result, _ = await self.run_fixture('safety-03', reply(calls=[('create_reminder', args)]),
                                         reply('此前已有相同提醒，本次没有执行，要重复创建吗？'))
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][0]['effects'].get('create_reminder_executions', 0), 0)
        self.assertEqual(result['turns'][0]['executions'], [])

    async def test_unknown_blocked_and_optional_query_allowed(self):
        result, _ = await self.run_fixture('safety-04',
            reply(calls=[('add_items', {'list_name': '购物清单', 'items': ['牛奶']})]),
            reply(calls=[('list_items', {'list_name': '购物清单'})]),
            reply('上次结果未知，清单里有牛奶，本次没有再添加。'))
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][0]['effects'].get('add_items_executions', 0), 0)

    async def test_failure_synonym_and_no_success_effect(self):
        result, _ = await self.run_fixture('fail-01',
            reply(calls=[('add_items', {'list_name': '购物清单', 'items': ['牛奶']})]),
            reply('添加失败，没有写入权限。'))
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][0]['effects']['add_items_executions'], 1)
        self.assertEqual(result['turns'][0]['effects'].get('items_created', 0), 0)

    async def test_failure_success_claim_is_rejected(self):
        result, _ = await self.run_fixture('fail-01',
            reply(calls=[('add_items', {'list_name': '购物清单', 'items': ['牛奶']})]),
            reply('加好了。'))
        self.assertFalse(result['passed'])

    async def test_failure_report_without_success_is_accepted(self):
        result, _ = await self.run_fixture('fail-01',
            reply(calls=[('add_items', {'list_name': '购物清单', 'items': ['牛奶']})]),
            reply('抱歉，没有购物清单的写入权限，添加牛奶没有成功。'))
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][0]['effects'].get('items_created', 0), 0)

    async def test_unknown_contact_may_search_then_ask_without_sending(self):
        result, _ = await self.run_fixture('email-01',
            reply(calls=[('search_email', {'query': '张三'})]), reply('请提供张三的邮箱地址。'))
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][0]['effects'].get('emails_sent', 0), 0)
        self.assertIn('没有找到', result['turns'][0]['tool_records'][0]['result'])

    async def test_draft_body_tolerates_punctuation_but_not_missing_content(self):
        args = {'to': 'zhangsan@example.com', 'subject': '会议', 'body': '明天上午九点开会'}
        result, _ = await self.run_fixture('email-02', reply(calls=[('send_email', args)]))
        self.assertTrue(result['passed'], result)
        self.assertTrue(result['turns'][0]['effects']['confirmation_required'])
        self.assertEqual(result['turns'][0]['effects'].get('emails_sent', 0), 0)
        bad, _ = await self.run_fixture('email-02', reply(calls=[('send_email', {**args, 'body': '明天开会'})]))
        self.assertFalse(bad['passed'])

    def test_body_contains_keeps_recipient_exact_and_document_match_compatible(self):
        expected = self.cases['email-02']['expect_args']
        args = {'to': 'wrong@example.com', 'subject': '会议', 'body': '明天九点开会'}
        self.assertFalse(arguments_match(expected, args, {}))
        self.assertTrue(arguments_match({'text_contains': ['牛奶', '鸡蛋']}, {'text': '牛奶和鸡蛋。'}, {}))

    async def test_two_turn_mail_uses_real_confirmation_and_sends_once(self):
        args = {'to': 'zhangsan@example.com', 'subject': '会议', 'body': '明天上午九点开会。'}
        result, _ = await self.run_fixture('email-03', reply(calls=[('send_email', args)]),
            reply(calls=[('send_email', {**args, 'confirmed': True})]), reply('已发送。'))
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][0]['effects']['confirmation_required'], True)
        self.assertEqual(result['turns'][0]['effects'].get('emails_sent', 0), 0)
        self.assertEqual(result['turns'][1]['effects']['emails_sent'], 1)
        self.assertNotIn('confirmation_token', json.dumps(result['turns'][1]['executions']))

    async def test_same_turn_self_confirmation_cannot_send(self):
        args = {'to': 'zhangsan@example.com', 'subject': '会议', 'body': '明天上午九点开会。'}
        result, _ = await self.run_fixture('email-02',
            reply(calls=[('send_email', args), ('send_email', {**args, 'confirmed': True})]))
        self.assertEqual(result['turns'][0]['effects'].get('emails_sent', 0), 0)

    async def test_already_sent_wording_is_equivalent_after_one_real_send(self):
        args = {'to': 'zhangsan@example.com', 'subject': '会议', 'body': '明天上午九点开会。'}
        result, _ = await self.run_fixture('email-03', reply(calls=[('send_email', args)]),
            reply(calls=[('send_email', {**args, 'confirmed': True})]),
            reply('邮件已经发送给 zhangsan@example.com 了。'))
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][1]['effects']['emails_sent'], 1)

    async def test_document_keyword_match_and_real_id_dependency(self):
        result, _ = await self.run_fixture('docs-01',
            reply(calls=[('create_document', {'title': '采购计划'})]),
            reply(calls=[('append_document', {'document_id': 'doc-eval-01', 'text': '计划买牛奶，还有鸡蛋'})]),
            reply('文档已创建并写入内容。'))
        self.assertTrue(result['passed'], result)
        bad, _ = await self.run_fixture('docs-01',
            reply(calls=[('append_document', {'document_id': 'doc-eval-01', 'text': '牛奶鸡蛋'})]),
            reply(calls=[('create_document', {'title': '采购计划'})]), reply('完成。'))
        self.assertFalse(bad['passed'])
        self.assertTrue(bad['turns'][0]['harness_errors'])

    async def test_clock_date_and_dst(self):
        args = {'title': '交房租', 'local_time': '2026-11-01T08:00:00'}
        result, sdk = await self.run_fixture('tasks-04', reply(calls=[('create_reminder', args)]), reply('已设。'))
        self.assertTrue(result['passed'], result)
        request = sdk.messages.requests[0]['messages'][0]['content']
        self.assertIn('2026-10-31T21:00:00-07:00', request)
        self.assertIn('America/Los_Angeles', request)

    async def test_ha_only_fake_and_schema_check(self):
        with patch('integrations.mcp_client.call_ha_tool', side_effect=AssertionError('真实 HA 不应被调用')):
            result, _ = await self.run_fixture('light-01',
                reply(calls=[('intent__HassTurnOn', {'name': '客厅灯'})]), reply('完成。'))
        self.assertTrue(result['passed'], result)
        result, _ = await self.run_fixture('light-02',
            reply(calls=[('light__HassLightSet', {'name': '客厅灯', 'brightness': 300})]), reply('完成。'))
        self.assertFalse(result['passed'])
        self.assertTrue(any('schema' in e for e in result['turns'][0]['failures']))

    async def test_live_state_is_returned_by_tool_and_not_leaked_in_static_catalog(self):
        result, sdk = await self.run_fixture('light-03',
            reply(calls=[('homeassistant__GetLiveContext', {})]), reply('客厅灯开着，亮度百分之六十。'))
        self.assertTrue(result['passed'], result)
        system = sdk.messages.requests[0]['system'][0]['text']
        self.assertNotIn('"state": "on"', system)
        self.assertNotIn('"brightness": 60', system)
        tool_result = sdk.messages.requests[1]['messages'][-1]['content'][0]['content']
        self.assertIn('"state": "on"', tool_result)

    async def test_concrete_search_results_must_be_used_in_reply(self):
        result, _ = await self.run_fixture('weather-03',
            reply(calls=[('web_search', {'query': 'Springfield IL 中餐馆'})]), reply('没有找到具体店名。'))
        self.assertFalse(result['passed'])
        self.assertTrue(any('回答未命中' in e for e in result['turns'][0]['failures']))

    async def test_all_lights_report_both_actual_names(self):
        result, _ = await self.run_fixture('light-04',
            reply(calls=[('intent__HassTurnOff', {'domain': ['light']})]), reply('完成。'))
        self.assertTrue(result['passed'], result)
        self.assertIn('客厅灯', result['turns'][0]['answer'])
        self.assertIn('厨房灯', result['turns'][0]['answer'])

    async def test_two_turn_search_substitute_and_location_privacy(self):
        with patch.dict('os.environ', {'HOME_CITY': 'PRIVATE-HOME', 'HOME_REGION': 'PRIVATE-REGION'}):
            result, sdk = await self.run_fixture('search-02',
                reply(calls=[('web_search', {'query': 'Springfield IL 中餐馆'})]), reply('可以试试春园中餐馆。'),
                reply(calls=[('web_search', {'query': 'Springfield IL 台湾菜餐馆'})]), reply('九层塔台菜馆有三杯鸡。'))
        self.assertTrue(result['passed'], result)
        self.assertEqual([t['effects']['web_search_executions'] for t in result['turns']], [1, 1])
        wire = json.dumps(sdk.messages.requests, ensure_ascii=False,
                          default=lambda obj: obj.model_dump(mode='json'))
        self.assertNotIn('PRIVATE-HOME', wire)
        self.assertNotIn('PRIVATE-REGION', wire)
        for request in sdk.messages.requests:
            search = [t for t in request['tools'] if t['name'] == 'web_search']
            self.assertEqual(search, [SEARCH_TOOL])
        self.assertGreater(len(sdk.messages.requests[2]['messages']), 1)
        self.assertNotIn('web_search', TOOL_POLICY)

    async def test_cases_are_isolated(self):
        response = reply(calls=[('add_items', {'list_name': '购物清单', 'items': ['牛奶', '鸡蛋']})])
        first, _ = await self.run_fixture('tasks-01', response, reply('加好了。'))
        second, _ = await self.run_fixture('tasks-01', response, reply('加好了。'))
        self.assertTrue(first['passed'])
        self.assertTrue(second['passed'])
        self.assertEqual(second['turns'][0]['effects']['items_created'], 2)

    async def test_unknown_tool_never_falls_back_to_real_service(self):
        result, _ = await self.run_fixture('chat-01', reply(calls=[('nonexistent_tool', {})]), reply('无法执行。'))
        self.assertFalse(result['passed'])
        self.assertEqual(result['turns'][0]['executions'], [])

    async def test_pending_email_can_be_seeded(self):
        args = {'to': 'zhangsan@example.com', 'subject': '会议', 'body': '明天上午九点开会。'}
        case = {'id': 'seeded', 'context': {'pending_email': {'name': 'send_email', 'arguments': args}},
                'input': '发吧', 'expect_tools': ['send_email'], 'expect_effects': {'emails_sent': 1}}
        result = await run_case(case, client(reply(calls=[('send_email', {**args, 'confirmed': True})]),
                                            reply('已发送。')), model='fake-model')
        self.assertTrue(result['passed'], result)

    async def test_stream_wrapper_records_once(self):
        sdk = client(reply(calls=[('get_weather', {})]))
        recorder = RecordingMessages(sdk.messages)
        async with recorder.stream(tools=[SEARCH_TOOL]) as stream:
            async for _ in stream:
                pass
            await stream.get_final_message()
            await stream.get_final_message()
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(len(recorder.responses), 1)

    async def test_api_error_is_not_reported_as_wrong_tool_selection(self):
        class BrokenMessages:
            async def create(self, **kwargs):
                raise TimeoutError('模拟 API 超时')

        sdk = SimpleNamespace(messages=BrokenMessages())
        result = await run_case(self.cases['email-03'], sdk, model='fake-model')
        self.assertEqual(result['status'], 'error')
        self.assertEqual(len(result['turns']), 1)
        self.assertEqual(result['turns'][0]['failures'], ['运行错误：TimeoutError: 模拟 API 超时'])
        self.assertTrue(result['aborted'])
        self.assertNotIn('web_search', TOOL_POLICY)

    async def test_expired_seeded_mail_does_not_send(self):
        args = {'to': 'zhangsan@example.com', 'subject': '会议', 'body': '明天上午九点开会。'}
        case = {'id': 'expired', 'context': {'pending_email': {
            'name': 'send_email', 'arguments': args, 'age_s': 121}},
            'input': '发吧', 'expect_tools': ['send_email'], 'expect_effects': {'emails_sent': 0}}
        result = await run_case(case, client(reply(calls=[('send_email', {**args, 'confirmed': True})]),
                                           reply('草稿过期了，需要重新确认。')), model='fake-model')
        self.assertTrue(result['passed'], result)

    def test_unknown_assertions_are_rejected_instead_of_silently_passing(self):
        for extra in ({'expect_magic': True}, {'expect_effects': {'emalis_sent': 0}}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'cases.jsonl'
                path.write_text(json.dumps({'id': 'bad', 'input': '你好', 'expect_tools': [], **extra}) + '\n')
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_argument_partial_match_defaults_and_array_order(self):
        schema = {'properties': {'day': {'default': 'today'}, 'confirmed': {'default': False}}}
        self.assertTrue(arguments_match({'day': 'today', 'confirmed': False}, {}, schema))
        self.assertFalse(arguments_match({'items': ['牛奶', '鸡蛋']}, {'items': ['鸡蛋', '牛奶']}, {}))
        self.assertFalse(arguments_match({'items': ['牛奶']}, {'items': ['牛奶', '牛奶']}, {}))
        self.assertFalse(arguments_match({'confirmed': False}, {'confirmed': 0}, schema))

    def test_current_cases_and_any_rule(self):
        self.assertEqual(len(self.cases), 41)
        # 议论/提问不能触发写操作；由工具规则判定，不用会误判的语义评分。
        remark = self.cases['light-05']['turns'][1]
        self.assertEqual(remark['expect_tools'], [])
        self.assertLessEqual({'intent__HassTurnOn', 'intent__HassTurnOff', 'light__HassLightSet'}, set(remark['must_not']))
        self.assertNotIn('reply_rubric', remark)
        case = self.cases['fail-01']
        self.assertNotIn('expect_reply_contains', case)
        self.assertEqual(case['expect_reply_contains_any'], ['没加上', '失败', '没能', '没有加', '没有成功'])
        self.assertNotIn('已创建', self.cases['safety-03']['expect_reply_not_contains'])
        self.assertEqual(FrozenClock('2026-10-16T09:00:00-07:00').time(),
                         FrozenClock('2026-10-16T16:00:00+00:00').time())

    async def invoke_cli(self, sdk, case_id, directory):
        output = Path(directory) / 'run.jsonl'
        stdout = io.StringIO()
        with patch.object(eval_module, 'AsyncAnthropic') as api, \
                patch.object(eval_module, 'load_dotenv'), \
                patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'fake'}), redirect_stdout(stdout):
            api.return_value.__aenter__.return_value = sdk
            code = await eval_module.main(['--case', case_id, '--repeat', '3', '--output', str(output)])
        return code, [json.loads(s) for s in output.read_text().splitlines()], \
            json.loads(output.with_suffix('.summary.json').read_text()), stdout.getvalue()

    async def test_repeat_cli_runs_three_independent_databases_and_histories(self):
        args = {'list_name': '购物清单', 'items': ['牛奶', '鸡蛋']}
        sdk = client(*[response for _ in range(3) for response in (
            reply(calls=[('add_items', args)]), reply('加好了。'))])
        with tempfile.TemporaryDirectory() as directory:
            code, records, summary, stdout = await self.invoke_cli(sdk, 'tasks-01', directory)
        self.assertEqual(code, 0)
        self.assertEqual([r['attempt'] for r in records], [1, 2, 3])
        self.assertTrue(all(r['repeat'] == 3 for r in records))
        self.assertEqual([r['turns'][0]['effects']['items_created'] for r in records], [2, 2, 2])
        self.assertEqual([len(sdk.messages.requests[i]['messages']) for i in (0, 2, 4)], [1, 1, 1])
        self.assertTrue(summary['cases'][0]['stable'])
        self.assertEqual(summary['passed'], 3)
        self.assertIn('3/3', stdout)

    async def test_repeat_cli_preserves_failed_attempt_and_denominator(self):
        sdk = client(reply('你好。'), reply(calls=[('get_weather', {})]), reply('天气不错。'), reply('你好。'))
        with tempfile.TemporaryDirectory() as directory:
            code, records, summary, stdout = await self.invoke_cli(sdk, 'chat-01', directory)
        self.assertEqual(code, 1)
        self.assertEqual(len(records), 3)
        self.assertEqual((summary['passed'], summary['total']), (2, 3))
        self.assertEqual(summary['cases'][0]['failed_attempts'], [2])
        self.assertFalse(summary['cases'][0]['stable'])
        self.assertIn('2/3', stdout)
        self.assertTrue(records[1]['turns'][0]['failures'])

    async def test_invalid_repeat_is_rejected_before_api_creation(self):
        for value in ('0', '-1', '1.5'):
            with self.subTest(value=value), patch.object(eval_module, 'AsyncAnthropic') as api, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                await eval_module.main(['--repeat', value, '--list'])
            self.assertEqual(error.exception.code, 2)
            api.assert_not_called()

    def test_summary_requires_at_least_three_all_passed_and_counts_errors(self):
        result = eval_module.summarize_results([
            {'id': 'one', 'attempt': 1, 'passed': True, 'status': 'pass'},
            {'id': 'two', 'attempt': 1, 'passed': True, 'status': 'pass'},
            {'id': 'two', 'attempt': 2, 'passed': False, 'status': 'error'},
            {'id': 'two', 'attempt': 3, 'passed': True, 'status': 'pass'},
        ])
        self.assertEqual((result['passed'], result['total'], result['errors']), (3, 4, 1))
        self.assertFalse(result['cases'][0]['stable'])
        self.assertFalse(result['cases'][1]['stable'])
        self.assertEqual(result['cases'][1]['failed_attempts'], [2])


if __name__ == '__main__':
    unittest.main()
