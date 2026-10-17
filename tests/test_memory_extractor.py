"""用假的 Haiku 验证提取边界和评分规则，不请求网络或创建记忆数据库。"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from brain.memory_extractor import Candidate, ExtractionError, MemoryExtractor
from core.memory import Memory
from eval.run_extraction import load_cases, run_attempt, score_case, summarize


def item(content='不吃香菜', evidence='我不吃香菜', **changes):
    return {'decision': 'save', 'operation': 'add', 'kind': 'preference',
            'content': content, 'evidence': evidence, **changes}


def response(candidates):
    return NS(stop_reason='tool_use', usage=None, content=[
        NS(type='tool_use', name='submit_candidates', input={'candidates': candidates})])


def extractor(candidates):
    client = NS(messages=NS(create=AsyncMock(return_value=response(candidates))))
    return MemoryExtractor(client)


class ExtractorTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_evidence_is_dropped_without_logging_input(self):
        model = extractor([item(evidence='编造的证据')])
        with self.assertLogs('alpha.memory_extractor', level='INFO') as logs:
            actual = await model.extract('我不吃香菜', [], [])
        self.assertEqual(actual, [])
        self.assertEqual(model.dropped, {'invalid_evidence': 1})
        self.assertNotIn('香菜', ''.join(logs.output))
        self.assertNotIn('编造的证据', ''.join(logs.output))

    async def test_update_id_must_exist_and_kind_must_match(self):
        model = extractor([item(operation='update', memory_id=99),
                           item(operation='update', memory_id=1, kind='fact')])
        actual = await model.extract('我不吃香菜', [Memory(1, 'preference', '吃香菜', 0, 0, 'seed')], [])
        self.assertEqual(actual, [])
        self.assertEqual(model.dropped, {'unknown_memory_id': 1, 'memory_kind_mismatch': 1})

    async def test_all_memory_tool_calls_skip_model_before_request(self):
        model = extractor([item()])
        for name in ('remember', 'update_memory', 'forget'):
            self.assertEqual(await model.extract('记住我不吃香菜', [], [name]), [])
            self.assertEqual(model.model_calls, 0)
        model.client.messages.create.assert_not_awaited()

    async def test_secrets_and_exact_existing_memory_are_dropped(self):
        model = extractor([item(), item(content='密码是 abc123', evidence='密码是 abc123', kind='fact')])
        actual = await model.extract('我不吃香菜，密码是 abc123',
                                     [Memory(1, 'preference', '不吃香菜', 0, 0, 'seed')], [])
        self.assertEqual(actual, [])
        self.assertEqual(model.dropped, {'exact_duplicate': 1, 'invalid_or_secret_content': 1})

    async def test_valid_candidates_use_only_user_and_public_memories_and_no_database(self):
        model = extractor([item(' 不吃香菜 '), item('不喝牛奶', '不喝牛奶')])
        memories = [Memory(1, 'fact', '普通事实', 10, 20, 'private-turn')]
        with patch('core.memory.sqlite3.connect', side_effect=AssertionError('不应连接数据库')):
            actual = await model.extract('我不吃香菜，也不喝牛奶', memories, [])
        self.assertEqual([c.content for c in actual], ['不吃香菜', '不喝牛奶'])
        request = model.client.messages.create.await_args.kwargs
        payload = json.loads(request['messages'][0]['content'])
        self.assertEqual(set(payload), {'user_text', 'memories'})
        self.assertEqual(payload['memories'], [{'id': 1, 'kind': 'fact', 'content': '普通事实'}])
        self.assertNotIn('private-turn', request['messages'][0]['content'])
        self.assertEqual(request['tool_choice']['name'], 'submit_candidates')
        self.assertNotIn('memory_id', actual[0].to_dict())

    async def test_invalid_structure_and_truncation_are_errors_not_empty_decisions(self):
        model = extractor([item(memory_id=1)])  # add 不允许带更新 ID。
        with self.assertRaises(ExtractionError):
            await model.extract('我不吃香菜', [], [])
        model.client.messages.create.return_value = NS(stop_reason='max_tokens', content=[], usage=None)
        with self.assertRaises(ExtractionError):
            await model.extract('我不吃香菜', [], [])
        model.client.messages.create.return_value = NS(stop_reason='end_turn', content=[], usage=None)
        with self.assertRaises(ExtractionError):
            await model.extract('我不吃香菜', [], [])

    async def test_diagnostics_reset_when_next_call_skips(self):
        model = extractor([item(evidence='错误')])
        await model.extract('我不吃香菜', [], [])
        self.assertEqual(model.model_calls, 1)
        await model.extract('任意输入', [], ['forget'])
        self.assertEqual(model.model_calls, 0)
        self.assertEqual(model.dropped, {})
        self.assertEqual(model.raw_candidate_count, 0)

    async def test_timeout_is_reported_and_excluded_from_confusion(self):
        model = extractor([])
        model.client.messages.create.side_effect = TimeoutError('原文不应出现在报告')
        row = await run_attempt({'id': 'error', 'user': '我不吃香菜', 'expect_candidates': []}, model)
        self.assertFalse(row['passed'])
        self.assertIsNone(row['score'])
        self.assertEqual(row['error'], {'type': 'TimeoutError'})
        report = summarize([row])
        self.assertEqual(report['counts']['runtime_errors'], 1)
        self.assertEqual(report['confusion']['none']['none'], 0)
        self.assertIsNone(report['content_pass_rate'])


class ExtractionScoreTests(unittest.TestCase):
    def setUp(self):
        self.cases = {c['id']: c for c in load_cases()}

    def test_reversed_candidates_produce_identical_scores(self):
        candidates = [Candidate(**item()), Candidate(**item('不喝牛奶', '不喝牛奶'))]
        first = score_case(self.cases['ext-32'], candidates, model_calls=1)
        second = score_case(self.cases['ext-32'], list(reversed(candidates)), model_calls=1)
        self.assertEqual(first, second)
        self.assertTrue(first['passed'])
        self.assertEqual(first['content_pass'], 2)

    def test_ask_misclassified_as_save_remains_paired_and_counts_as_unsafe(self):
        actual = [Candidate(**item('用户对花生过敏', '我对花生过敏', kind='fact'))]
        score = score_case(self.cases['ext-11'], actual, model_calls=1)
        self.assertEqual(score['confusion']['ask']['save'], 1)
        self.assertEqual(score['unsafe_saves'], 1)
        self.assertEqual(score['extras'], [])
        self.assertFalse(score['passed'])

    def test_extras_and_duplicate_outputs_cannot_hide_in_a_save_case(self):
        actual = [Candidate(**item()), Candidate(**item('不喝牛奶', '我不吃香菜'))]
        score = score_case(self.cases['ext-01'], actual, model_calls=1)
        self.assertEqual(score['unsafe_saves'], 1)
        self.assertEqual(len(score['extras']), 1)
        # 两次同一候选不能靠整句 evidence 假装覆盖第二个信息。
        duplicate = [Candidate(**item(evidence='我不吃香菜，也不喝牛奶'))] * 2
        score = score_case(self.cases['ext-32'], duplicate, model_calls=1)
        self.assertEqual(len(score['missing']), 1)
        self.assertEqual(len(score['extras']), 1)

    def test_invented_generalization_fails_content_despite_real_evidence(self):
        score = score_case(self.cases['ext-11'], [Candidate(**item(
            '用户对坚果过敏', '我对花生过敏', decision='ask', kind='fact'))], model_calls=1)
        self.assertEqual(score['content_pass'], 0)
        self.assertEqual(score['confusion']['ask']['ask'], 1)
        self.assertFalse(score['passed'])

    def test_wrong_update_target_is_not_a_safe_save(self):
        case = self.cases['ext-09']
        actual = [Candidate(**item('空调习惯二十四度', '空调二十四度', operation='update', memory_id=2))]
        score = score_case(case, actual, model_calls=1)
        self.assertEqual(score['field_errors']['memory_id'], 1)
        self.assertEqual(score['unsafe_saves'], 1)

    def test_skip_case_fails_if_model_called_even_with_empty_output(self):
        for ident in ('ext-31', 'ext-37'):
            self.assertTrue(score_case(self.cases[ident], [], model_calls=0)['passed'])
            score = score_case(self.cases[ident], [], model_calls=1)
            self.assertFalse(score['passed'])
            self.assertFalse(score['skip_ok'])

    def test_misses_extras_and_empty_case_have_explicit_matrix_cells(self):
        missing = score_case(self.cases['ext-01'], [], model_calls=1)
        self.assertEqual(missing['confusion']['save']['none'], 1)
        self.assertEqual(missing['content_total'], 1)
        extra = score_case(self.cases['ext-17'], [Candidate(**item())], model_calls=1)
        self.assertEqual(extra['confusion']['none']['save'], 1)
        self.assertEqual(extra['unsafe_saves'], 1)
        empty = score_case(self.cases['ext-17'], [], model_calls=1)
        self.assertEqual(empty['confusion']['none']['none'], 1)

    def test_all_37_annotations_are_valid(self):
        self.assertEqual(len(self.cases), 37)
