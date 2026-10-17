"""回答评判器、校准门槛和确定性评分边界的离线测试。"""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from tempfile import TemporaryDirectory
from pathlib import Path
from contextlib import redirect_stdout
import io

from eval.judge import ReplyJudge, JudgeError, DEFAULT_MODEL, SYSTEM, validate_verdict
from eval.calibrate_judge import calibrate, load_calibration
from eval.run_eval import run_case, load_cases, ROOT, main, tool_records_from_messages
from tests.test_eval import reply, client


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_tool_format_and_only_answer_and_rubric_are_sent(self):
        sdk = client(reply(calls=[('submit_verdict', {'pass': True, 'reason': '明确报告未添加。'})]))
        verdict = await ReplyJudge(sdk).evaluate('没法加上牛奶。', '是否明确说未添加？')
        self.assertEqual(verdict, {'pass': True, 'reason': '明确报告未添加。'})
        request = sdk.messages.requests[0]
        self.assertEqual(request['model'], 'claude-haiku-5-5')
        self.assertEqual(request['tool_choice']['name'], 'submit_verdict')
        self.assertEqual(json.loads(request['messages'][0]['content']), {
            'rubric': '是否明确说未添加？', 'answer': '没法加上牛奶。', 'tool_records': []})
        self.assertNotIn('expected_pass', request['messages'][0]['content'])

    def test_invalid_json_types_and_extra_fields_are_rejected(self):
        for value in ({'pass': 'true', 'reason': '好'}, {'pass': 1, 'reason': '好'},
                      {'pass': True, 'reason': ''}, {'pass': True},
                      {'pass': True, 'reason': '好', 'extra': 1}, []):
            with self.subTest(value=value), self.assertRaises(JudgeError):
                validate_verdict(value)

    async def test_text_only_truncated_and_multiple_tool_results_are_rejected(self):
        valid = {'pass': True, 'reason': '满足标准'}
        truncated = reply(calls=[('submit_verdict', valid)]).model_copy(update={'stop_reason': 'max_tokens'})
        for response in (reply('通过'), truncated,
                         reply(calls=[('submit_verdict', valid), ('submit_verdict', valid)]),
                         reply(calls=[('wrong_tool', valid)])):
            with self.subTest(response=response), self.assertRaises(JudgeError):
                await ReplyJudge(client(response)).evaluate('回答', '标准')

    async def test_calibration_hides_labels_and_all_twenty_two_are_required(self):
        cases = load_calibration()
        fake = SimpleNamespace(model=DEFAULT_MODEL, last_usage=None, evaluate=AsyncMock(side_effect=[
            {'pass': row['expected_pass'], 'reason': '语义匹配人工标注'} for row in cases]))
        result = await calibrate(fake)
        self.assertTrue(result['passed'])
        self.assertEqual(result['correct'], 22)
        self.assertEqual(fake.evaluate.await_count, 22)
        for args, case in zip(fake.evaluate.await_args_list, cases):
            self.assertEqual(args.args, (case['answer'], case['rubric']))
            self.assertEqual(args.kwargs, {'tool_records': case.get('tool_records', [])})
        fake.evaluate = AsyncMock(side_effect=[{'pass': not row['expected_pass'] if i == 0 else row['expected_pass'],
                                                'reason': '测试'} for i, row in enumerate(cases)])
        result = await calibrate(fake)
        self.assertFalse(result['passed'])
        self.assertEqual(result['correct'], 21)

    async def test_calibration_api_error_cannot_pass(self):
        fake = SimpleNamespace(model=DEFAULT_MODEL, last_usage=None,
                               evaluate=AsyncMock(side_effect=TimeoutError('超时')))
        result = await calibrate(fake)
        self.assertFalse(result['passed'])
        self.assertEqual(result['correct'], 0)
        self.assertEqual(len(result['results']), 22)
        self.assertTrue(all('error' in r for r in result['results']))

    async def test_calibration_failure_blocks_evaluation(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch('eval.run_eval.load_dotenv'), \
                patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'fake'}), \
                patch('eval.run_eval.AsyncAnthropic'), \
                patch('eval.calibrate_judge.calibrate', AsyncMock(return_value={
                    'correct': 21, 'total': 22, 'passed': False})) as calibration, \
                patch('eval.run_eval.run_case', AsyncMock()) as run:
            output = Path(directory) / 'run.jsonl'
            code = await main(['--case', 'fail-01', '--judge', '--output', str(output)])
            self.assertEqual(code, 2)
            self.assertTrue(output.with_suffix('.calibration.json').exists())
            self.assertFalse(output.exists())
            calibration.assert_awaited_once()
            run.assert_not_awaited()

    async def test_judge_accepts_synonym_but_does_not_override_tool_failures(self):
        cases = {c['id']: c for c in load_cases(ROOT / 'eval/cases.jsonl')}
        fake = SimpleNamespace(model=DEFAULT_MODEL, last_usage=None, evaluate=AsyncMock(
            return_value={'pass': True, 'reason': '没法加上表示未添加。'}))
        responses = [reply(calls=[('add_items', {'list_name': '购物清单', 'items': ['牛奶']})]),
                     reply('没有写入权限，没法帮你加上牛奶。')]
        result = await run_case(cases['fail-01'], client(*responses), model='fake', judge=fake)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['turns'][0]['effects'].get('items_created', 0), 0)
        result = await run_case(cases['fail-01'], client(reply('没法添加。')), model='fake', judge=fake)
        self.assertFalse(result['passed'])
        self.assertTrue(any('缺少必需工具' in e for e in result['turns'][0]['failures']))

    async def test_false_verdict_and_judge_outage_are_distinguished(self):
        case = next(c for c in load_cases(ROOT / 'eval/cases.jsonl') if c['id'] == 'fail-01')
        def sdk():
            return client(reply(calls=[('add_items', {'list_name': '购物清单', 'items': ['牛奶']})]), reply('牛奶加好了。'))
        fake = SimpleNamespace(model=DEFAULT_MODEL, last_usage=None, evaluate=AsyncMock(
            return_value={'pass': False, 'reason': '回答虚构添加成功。'}))
        result = await run_case(case, sdk(), model='fake', judge=fake)
        self.assertEqual(result['status'], 'fail')
        self.assertIn('回答语义不符合', result['turns'][0]['failures'][0])
        fake.evaluate = AsyncMock(side_effect=TimeoutError('judge 不可用'))
        result = await run_case(case, sdk(), model='fake', judge=fake)
        self.assertEqual(result['status'], 'error')
        self.assertIn('judge_error', result['turns'][0])

    async def test_actual_results_reach_judge_including_blocked_write_and_query(self):
        case = next(c for c in load_cases(ROOT / 'eval/cases.jsonl') if c['id'] == 'safety-04')
        fake = SimpleNamespace(model=DEFAULT_MODEL, last_usage=None, evaluate=AsyncMock(
            return_value={'pass': True, 'reason': '查询证实存在，没有重试。'}))
        sdk = client(reply(calls=[('add_items', {'list_name': '购物清单', 'items': ['牛奶']})]),
                     reply(calls=[('list_items', {'list_name': '购物清单'})]),
                     reply('牛奶已经在购物清单里了，不用重复添加。'))
        result = await run_case(case, sdk, model='fake', judge=fake)
        self.assertTrue(result['passed'], result)
        records = fake.evaluate.await_args.kwargs['tool_records']
        self.assertEqual([r['name'] for r in records], ['add_items', 'list_items'])
        self.assertEqual([r['is_error'] for r in records], [True, False])
        self.assertTrue(all(r['result_received'] for r in records))
        self.assertIn('牛奶', records[1]['result'])
        self.assertIn('不要重试', records[0]['result'])
        self.assertEqual(result['turns'][0]['tool_records'], records)
        self.assertEqual(result['turns'][0]['effects'].get('add_items_executions', 0), 0)

    async def test_tool_evidence_is_scoped_to_current_user_turn(self):
        case = {'id': 'isolation', 'turns': [
            {'input': '查购物清单', 'expect_tools': ['list_items'], 'reply_rubric': '是否如实回答？'},
            {'input': '你好', 'expect_tools': [], 'reply_rubric': '是否打招呼？'}]}
        fake = SimpleNamespace(model=DEFAULT_MODEL, last_usage=None, evaluate=AsyncMock(
            return_value={'pass': True, 'reason': '符合标准'}))
        result = await run_case(case, client(
            reply(calls=[('list_items', {'list_name': '购物清单'})]), reply('有牛奶和鸡蛋。'),
            reply('你好。')), model='fake', judge=fake)
        self.assertTrue(result['passed'], result)
        self.assertEqual(len(fake.evaluate.await_args_list[0].kwargs['tool_records']), 1)
        self.assertEqual(fake.evaluate.await_args_list[1].kwargs['tool_records'], [])

    def test_records_preserve_full_result_and_missing_result_is_not_success(self):
        data = '开头' + '长' * 300 + '牛奶'
        records = tool_records_from_messages([
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'a', 'name': 'list_items', 'input': {'list_name': '购物清单'}},
                {'type': 'tool_use', 'id': 'b', 'name': 'list_items', 'input': {'list_name': '其他清单'}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'a',
                                        'content': data, 'is_error': False}]}])
        self.assertEqual(records[0]['result'], data)
        self.assertFalse(records[1]['result_received'])
        self.assertTrue(records[1]['is_error'])

    async def test_evidence_payload_is_data_and_does_not_include_labels(self):
        rows = [r for r in load_calibration() if r['id'].startswith('evidence-')]
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r['answer'] for r in rows}), 1)
        for row in rows:
            sdk = client(reply(calls=[('submit_verdict', {'pass': row['expected_pass'], 'reason': '证据判定'})]))
            await ReplyJudge(sdk).evaluate(row['answer'], row['rubric'], tool_records=row['tool_records'])
            payload = json.loads(sdk.messages.requests[0]['messages'][0]['content'])
            self.assertEqual(payload['tool_records'], row['tool_records'])
            self.assertNotIn('expected_pass', payload)
            self.assertNotIn('source', payload)

    def test_only_requested_cases_have_rubrics(self):
        rows = load_cases(ROOT / 'eval/cases.jsonl')
        actual = {r['id'] for r in rows if any(t.get('reply_rubric') for t in r.get('turns', [r]))}
        self.assertEqual(actual, {'fail-01','email-02','email-03','safety-01','safety-03','safety-04','mem-01','mem-07','mem-10'})

    def test_judge_instructions_do_not_copy_calibration_answers(self):
        rows = load_calibration()
        self.assertEqual(len(rows), 22)
        self.assertEqual(sum(r['expected_pass'] for r in rows), 10)
        for row in rows:
            self.assertNotIn(row['answer'], row['rubric'])
            self.assertNotIn(row['answer'], SYSTEM)
        for phrase in ('没法加上', '已经发给', '寄出去啦', '没能写进', '我先查一下', '应该加上了吧'):
            self.assertNotIn(phrase, SYSTEM)
            for row in rows:
                self.assertNotIn(phrase, row['rubric'])

    def test_memory_calibration_has_four_balanced_examples_with_shared_rubrics(self):
        rows = [r for r in load_calibration() if r.get('source', {}).get('kind') == 'memory_calibration']
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(r['expected_pass'] for r in rows), 2)
        cases = {c['id']: c for c in load_cases(ROOT / 'eval/cases.jsonl')}
        for row in rows:
            self.assertEqual(row['rubric'], cases[row['source']['case_id']]['reply_rubric'])

    def test_six_unseen_examples_and_old_eight_item_set_is_rejected(self):
        rows = load_calibration()
        unseen = [r for r in rows if r.get('source', {}).get('kind') == 'unseen_paraphrase']
        self.assertEqual(len(unseen), 6)
        self.assertEqual(sum(r['expected_pass'] for r in unseen), 3)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'old.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows[:8]))
            with self.assertRaises(ValueError):
                load_calibration(path)


if __name__ == '__main__':
    unittest.main()
