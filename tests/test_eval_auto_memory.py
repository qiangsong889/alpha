"""端到端 eval 使用真实自动记忆生命周期，模型替身只负责返回受控响应。"""
import asyncio
import io
import json
import tempfile
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import eval.run_eval as eval_module
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from eval.run_eval import ROOT, load_cases, run_case
from tests.test_eval import FakeMessages, reply


def extraction(*candidates):
    return reply(calls=[('submit_candidates', {'candidates': list(candidates)})])

SAVE = {'decision': 'save', 'operation': 'add', 'kind': 'preference',
        'content': '不吃香菜', 'evidence': '我不吃香菜'}
ASK = {'decision': 'ask', 'operation': 'add', 'kind': 'fact',
       'content': '对花生过敏', 'evidence': '我对花生过敏'}


class Dispatch:
    def __init__(self, main, background):
        self.main = FakeMessages(*main)
        self.background = FakeMessages(*background)

    async def create(self, **kwargs):
        target = self.main if kwargs['model'] == 'fake-model' else self.background
        return await target.create(**kwargs)


class AutoEvalTests(IsolatedAsyncioTestCase):
    async def test_judge_receives_actual_before_after_database_snapshots(self):
        self.cases = {row['id']: row for row in load_cases(ROOT / 'eval/cases.jsonl')}
        dispatch = Dispatch([reply('好的。'), reply('你好。'), reply('不错。')],
                            [extraction(SAVE), extraction(), extraction()])
        judge = NS(model='fake-judge', last_usage=None,
                   evaluate=AsyncMock(return_value={'pass': True, 'reason': '数据库已保存'}))
        result = await run_case(self.cases['auto-02'], NS(messages=dispatch), model='fake-model',
                                auto_memory=True, judge=judge)
        self.assertTrue(result['passed'], result)
        evidence = judge.evaluate.call_args.kwargs['memory_records']
        self.assertEqual(evidence['before'], [])
        self.assertEqual(evidence['after'][0]['content'], '不吃香菜')
        self.assertEqual(evidence['after'][0]['source'], 'auto')
        self.assertEqual(judge.evaluate.call_args.kwargs['tool_records'], [])

    async def test_tool_free_memory_wording_correction_is_supported(self):
        result, sdk = await self.run_fixture('chat-01',
            [reply('记住了。'), reply('你好。')], [extraction()])
        self.assertTrue(result['passed'], result)
        self.assertEqual(len(sdk.main.requests), 2)
        self.assertNotIn('tools', sdk.main.requests[1])
        self.assertEqual(result['turns'][0]['answer'], '你好。')
        self.assertEqual(result['turns'][0]['memories'], [])

    async def test_saved_paraphrase_does_not_add_a_correction_request(self):
        result, sdk = await self.run_fixture('auto-02',
            [reply('好的，记住了，以后避开香菜。'), reply('你好。'), reply('不错。')],
            [extraction(SAVE), extraction(), extraction()])
        self.assertTrue(result['passed'], result)
        self.assertEqual(len(sdk.main.requests), 3)  # 三轮正常回答，没有第四次修正请求。
        self.assertTrue(all('tools' in request for request in sdk.main.requests))
        self.assertEqual(result['turns'][0]['answer'], '好的，记住了，以后避开香菜。记下了：不吃香菜。')

    def setUp(self):
        self.cases = {row['id']: row for row in load_cases(ROOT / 'eval/cases.jsonl')}

    async def run_fixture(self, ident, main, background):
        dispatch = Dispatch(main, background)
        result = await run_case(self.cases[ident], NS(messages=dispatch), model='fake-model', auto_memory=True)
        return result, dispatch

    async def test_notice_is_checked_and_emitted_once_in_same_answer(self):
        result, sdk = await self.run_fixture('auto-02',
            [reply('好的。'), reply('你好。'), reply('不错。')],
            [extraction(SAVE), extraction(), extraction()])
        self.assertTrue(result['passed'], result)
        first, second, third = result['turns']
        self.assertEqual(first['memories'][0]['source'], 'auto')
        self.assertTrue(first['answer'].endswith('记下了：不吃香菜。'))
        self.assertNotIn('记下了', second['answer'])
        self.assertFalse(third['auto_notice_in_answer'])
        self.assertNotIn('记下了', third['answer'])
        self.assertEqual(first['tool_calls'], [])
        self.assertFalse({'remember', 'update_memory', 'forget'} & {t['name'] for t in sdk.main.requests[0]['tools']})
        self.assertEqual(len(sdk.background.requests), 3)

    async def test_restart_uses_database_without_previous_history(self):
        result, sdk = await self.run_fixture('auto-01', [
            reply(calls=[('web_search', {'query': 'Springfield 川菜馆'})]),
            reply('春园川菜馆。'), reply('麻婆豆腐，不放香菜。')],
            [extraction(SAVE), extraction()])
        self.assertTrue(result['passed'], result)
        second_request = sdk.main.requests[-1]
        self.assertEqual(len(second_request['messages']), 1)
        self.assertIn('不吃香菜', str(second_request['system']))
        self.assertEqual(result['turns'][1]['memories'][0]['source'], 'auto')

    async def test_sensitive_candidate_is_asked_next_turn_then_saved_explicitly(self):
        result, sdk = await self.run_fixture('auto-04', [
            reply(calls=[('web_search', {'query': 'Springfield 泰国餐厅'})]), reply('先联系餐馆核实过敏安全。'),
            reply('你好。'), reply(calls=[('remember', {'kind': 'fact', 'content': '对花生过敏'})]), reply('记住了你对花生过敏。')],
            [extraction(ASK), extraction()])
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][0]['memories'], [])
        self.assertEqual(result['turns'][1]['memories'], [])
        self.assertTrue(result['turns'][0]['auto_question_at_end'])
        self.assertFalse(result['turns'][1]['auto_question_at_end'])
        last = result['turns'][2]
        self.assertEqual(last['memories'][0]['source'], 'explicit')
        self.assertEqual(last['auto_memory']['responses'], [])
        self.assertEqual(last['auto_memory']['extractions'], [])
        self.assertEqual(len(sdk.background.requests), 2)

    async def test_explicit_memory_tool_skips_background_entirely(self):
        result, sdk = await self.run_fixture('mem-01', [
            reply(calls=[('remember', {'content': '客厅灯亮度30%', 'kind': 'preference'})]),
            reply('记住了，客厅灯亮度30%。')], [])
        self.assertTrue(result['passed'], result)
        self.assertEqual(sdk.background.requests, [])

    async def test_regression_guard_catches_unexpected_auto_save_without_case_count_assertion(self):
        bad = {**SAVE, 'content': '用户说你好', 'evidence': '你好', 'kind': 'fact'}
        result, _ = await self.run_fixture('chat-01', [reply('你好。')], [extraction(bad)])
        self.assertFalse(result['passed'])
        self.assertIn('原有业务用例产生了非预期的自动记忆变更', result['turns'][0]['harness_errors'])

    async def test_unknown_background_failure_cannot_pass_as_no_memory(self):
        result, _ = await self.run_fixture('auto-05', [reply('这是邮件内容。')], [])
        self.assertEqual(result['status'], 'error')
        self.assertIn('后台记忆提取失败', result['turns'][0]['error'])

    async def test_parallel_extraction_finishes_before_answer_and_next_turn(self):
        entered, release = asyncio.Event(), asyncio.Event()
        dispatch = Dispatch([reply('好的。'), reply('你好。'), reply('不错。')],
                            [extraction(SAVE), extraction(), extraction()])
        original = dispatch.background.create
        async def blocked(**kwargs):
            entered.set()
            await release.wait()
            return await original(**kwargs)
        dispatch.background.create = blocked
        task = asyncio.create_task(run_case(self.cases['auto-02'], NS(messages=dispatch),
                                           model='fake-model', auto_memory=True))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            self.assertEqual(len(dispatch.main.requests), 1)
            self.assertFalse(task.done())
            release.set()
            result = await asyncio.wait_for(task, 3)
            self.assertTrue(result['passed'], result)
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_auto_flag_does_not_inherit_local_env_when_disabled(self):
        dispatch = Dispatch([reply('你好。')], [])
        with patch.dict('os.environ', {'ALPHA_AUTO_MEMORY': '1'}):
            result = await run_case(self.cases['chat-01'], NS(messages=dispatch), model='fake-model')
        self.assertTrue(result['passed'], result)
        self.assertFalse(result['auto_memory'])
        self.assertEqual(dispatch.background.requests, [])

    async def test_cli_enables_real_lifecycle_and_records_flag(self):
        dispatch = Dispatch([reply('好的。'), reply('你好。'), reply('不错。')],
                            [extraction(SAVE), extraction(), extraction()])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run.jsonl'
            with patch.object(eval_module, 'AsyncAnthropic') as api, \
                    patch.object(eval_module, 'load_dotenv'), \
                    patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'fake'}), redirect_stdout(io.StringIO()):
                api.return_value.__aenter__.return_value = NS(messages=dispatch)
                code = await eval_module.main(['--case', 'auto-02', '--model', 'fake-model', '--auto-memory', '--output', str(output)])
            self.assertEqual(code, 0)
            summary = json.loads(output.with_suffix('.summary.json').read_text())
            self.assertTrue(summary['auto_memory'])
            record = json.loads(output.read_text())
            self.assertTrue(record['passed'])
            self.assertEqual(record['turns'][0]['effects']['auto_extraction_calls'], 1)

    async def test_auto_case_without_flag_is_rejected_before_api_call(self):
        with patch.object(eval_module, 'AsyncAnthropic') as api, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                await eval_module.main(['--case', 'auto-02'])
            api.assert_not_called()

    async def test_unrequested_main_memory_call_cannot_write_even_if_model_forges_hidden_tool(self):
        dispatch = Dispatch([
            reply(calls=[('remember', {'kind': 'preference', 'content': '不吃香菜'})]),
            reply('好的。')], [extraction()])
        case = {'id': 'implicit-forged', 'input': '我不吃香菜', 'expect_tools': [], 'requires_auto_memory': True}
        result = await run_case(case, NS(messages=dispatch), model='fake-model', auto_memory=True)
        self.assertFalse(result['passed'])
        self.assertEqual(result['turns'][0]['memories'], [])
        self.assertTrue(result['turns'][0]['tool_records'][0]['is_error'])
