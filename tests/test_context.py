"""上下文压缩、完整轮次边界与并发写回；不请求真实模型。"""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch

from brain.context import ConversationContext, SUMMARY_RULES, old_prefix_end


def history(count=8):
    result = []
    for index in range(count):
        result.extend([
            {'role': 'user', 'content': f'请求{index}：' + '内容' * 80},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': str(index), 'name': 'lookup', 'input': {}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': str(index), 'content': '查询结果'}]},
            {'role': 'assistant', 'content': '答复'},
        ])
    return result


class BoundaryTests(TestCase):
    def test_environment_defaults_and_config_logging(self):
        with patch.dict('os.environ', {}, clear=True), self.assertLogs('alpha.context', level='INFO') as logs:
            context = ConversationContext()
            context.configure_from_env()
        self.assertEqual((context.idle_s, context.soft_tokens, context.hard_tokens), (1800, 15000, 40000))
        self.assertIn('空闲 30 分钟', logs.output[0])
        self.assertIn('压缩 15000 token', logs.output[0])
        self.assertIn('硬上限 40000 token', logs.output[0])

    def test_environment_custom_thresholds(self):
        with patch.dict('os.environ', {'ALPHA_CONTEXT_IDLE_MIN': '7.5',
                'ALPHA_CONTEXT_COMPACT_TOKENS': '17000', 'ALPHA_CONTEXT_HARD_TOKENS': '45000'}):
            context = ConversationContext()
            context.configure_from_env()
        self.assertEqual((context.idle_s, context.soft_tokens, context.hard_tokens), (450, 17000, 45000))
        self.assertFalse(context.observe_user(0))
        self.assertTrue(context.observe_user(451))

    def test_invalid_environment_is_rejected_without_partial_updates(self):
        for settings in ({'ALPHA_CONTEXT_IDLE_MIN': 'nan'}, {'ALPHA_CONTEXT_IDLE_MIN': 'inf'},
                {'ALPHA_CONTEXT_IDLE_MIN': '0'}, {'ALPHA_CONTEXT_IDLE_MIN': '-1'},
                {'ALPHA_CONTEXT_COMPACT_TOKENS': '1.5'}, {'ALPHA_CONTEXT_HARD_TOKENS': 'bad'},
                {'ALPHA_CONTEXT_COMPACT_TOKENS': '40000'}, {'ALPHA_CONTEXT_HARD_TOKENS': '14999'}):
            with self.subTest(settings=settings), patch.dict('os.environ', settings, clear=True):
                context = ConversationContext()
                with self.assertRaisesRegex(ValueError, 'ALPHA_CONTEXT'):
                    context.configure_from_env()
                self.assertEqual((context.idle_s, context.soft_tokens, context.hard_tokens), (1800, 15000, 40000))

    def test_keep_six_complete_turns_with_tool_use_and_results(self):
        messages = history()
        self.assertEqual(old_prefix_end(messages), 8)
        self.assertEqual(messages[8]['content'][:3], '请求2')
        self.assertEqual(old_prefix_end(history(6)), 0)
        messages[2]['content'].append({'type': 'text', 'text': '请根据工具结果继续'})
        self.assertEqual(old_prefix_end(messages), 8)

    def test_user_activity_clock_not_tool_or_audio_activity(self):
        context = ConversationContext()
        self.assertFalse(context.observe_user(0))
        self.assertFalse(context.observe_user(1800))
        self.assertTrue(context.observe_user(3601))


class ContextTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = ConversationContext(soft_tokens=100, hard_tokens=20000)
        self.addAsyncCleanup(self.context.close)
        self.brain = NS(messages=history(), model='claude-haiku-5-5', _lock=asyncio.Lock(),
                        client=NS(messages=NS(count_tokens=AsyncMock(return_value=NS(input_tokens=9000)),
                                              create=AsyncMock())))
        self.context.summarize = AsyncMock(return_value='{"user_facts":["用户要求查资料"]}')

    async def test_background_does_not_wait_and_preserves_new_turns(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def summarize(*args):
            started.set()
            await release.wait()
            return '{"user_facts":["旧目标"]}'
        self.context.summarize.side_effect = summarize
        original = deepcopy(self.brain.messages)
        self.context.after_turn(self.brain)
        await started.wait()
        self.assertFalse(self.context.task.done())
        async with self.brain._lock:
            self.brain.messages.extend([{'role': 'user', 'content': '新任务'}, {'role': 'assistant', 'content': '新答复'}])
        release.set()
        await self.context.task
        self.assertEqual(self.brain.messages, original[8:] + [
            {'role': 'user', 'content': '新任务'}, {'role': 'assistant', 'content': '新答复'}])
        self.assertIn('旧目标', self.context.summary)

    async def test_late_old_session_summary_is_discarded(self):
        async def summarize(*args):
            self.context.session_id = 'new-session'
            return '过期摘要'
        self.context.summarize.side_effect = summarize
        original = deepcopy(self.brain.messages)
        self.context.after_turn(self.brain)
        await self.context.task
        self.assertEqual(self.context.summary, '')
        self.assertEqual(self.brain.messages, original)

    async def test_changed_old_prefix_is_not_overwritten(self):
        async def summarize(*args):
            self.brain.messages[0]['content'] = '已修正的用户原话'
            return '旧摘要'
        self.context.summarize.side_effect = summarize
        self.context.after_turn(self.brain)
        await self.context.task
        self.assertEqual(len(self.brain.messages), 32)
        self.assertEqual(self.context.summary, '')

    async def test_summary_failure_keeps_original_history_and_summary(self):
        self.context.summary = '已有摘要'
        self.context.summarize.side_effect = TimeoutError()
        original = deepcopy(self.brain.messages)
        with self.assertLogs('alpha.context', level='WARNING'):
            self.context.after_turn(self.brain)
            await self.context.task
        self.assertEqual(self.brain.messages, original)
        self.assertEqual(self.context.summary, '已有摘要')

    async def test_static_rules_tools_and_memories_are_not_counted(self):
        self.brain.system = [{'type': 'text', 'text': '静态规则和持久记忆' * 20000}]
        self.brain.tools = [{'name': 'lookup', 'description': '工具目录' * 20000}]
        self.context.after_turn(self.brain)
        await self.context.task
        request = self.brain.client.messages.count_tokens.call_args_list[0].kwargs
        self.assertEqual(set(request), {'model', 'messages'})
        for call in self.brain.client.messages.count_tokens.call_args_list:
            self.assertNotIn('静态规则', json.dumps(call.kwargs, ensure_ascii=False))

    async def test_small_history_never_counts_or_summarizes(self):
        self.context.soft_tokens = 8000
        self.brain.messages = [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好'}]
        self.context.after_turn(self.brain)
        self.assertIsNone(self.context.task)
        self.brain.client.messages.count_tokens.assert_not_awaited()

    async def test_hard_limit_compresses_synchronously_before_next_request(self):
        self.context.hard_tokens = 200
        self.brain.client.messages.count_tokens.side_effect = [NS(input_tokens=25000), NS(input_tokens=150)]
        original = deepcopy(self.brain.messages)
        await self.context.ensure_budget(self.brain, {'role': 'user', 'content': '下一句'})
        self.assertEqual(self.brain.messages, original[8:])
        self.context.summarize.assert_awaited_once()
        request = self.brain.client.messages.count_tokens.call_args.kwargs
        self.assertEqual(request['messages'][-1]['content'], '下一句')

    async def test_hard_limit_failure_is_explicit_and_preserves_history(self):
        self.context.hard_tokens = 200
        self.context.summarize.side_effect = RuntimeError('失败')
        original = deepcopy(self.brain.messages)
        with self.assertRaisesRegex(RuntimeError, '尚未执行'):
            await self.context.ensure_budget(self.brain)
        self.assertEqual(self.brain.messages, original)

    async def test_count_failure_uses_conservative_bound(self):
        self.brain.client.messages.count_tokens.side_effect = ConnectionError()
        with self.assertLogs('alpha.context', level='WARNING'):
            tokens = await self.context.count(self.brain, self.brain.messages, '')
        self.assertEqual(tokens, self.context.byte_bound(self.brain.messages))
        self.assertIsNone(self.context._token_baseline)

    async def test_long_chinese_history_uses_background_count_without_repeated_foreground_requests(self):
        self.context.soft_tokens, self.context.hard_tokens = 8000, 24000
        # 字节很多但真实token远小于硬上限；即便不足六轮也应先在后台建立基线。
        self.brain.messages = [{'role': 'user', 'content': '字' * 11000}, {'role': 'assistant', 'content': '收到'}]
        self.brain.client.messages.count_tokens.return_value = NS(input_tokens=6000)
        self.assertGreater(self.context.byte_bound(self.brain.messages), 24000)
        self.context.after_turn(self.brain)
        await self.context.task
        counter = self.brain.client.messages.count_tokens
        counter.assert_awaited_once()
        for index in range(3):
            incoming = {'role': 'user', 'content': f'第{index}句'}
            await self.context.ensure_budget(self.brain, incoming)
            self.brain.messages.extend([incoming, {'role': 'assistant', 'content': '好'}])
            self.context.after_turn(self.brain)
        self.assertEqual(counter.await_count, 1)
        self.context.summarize.assert_not_awaited()

    async def test_cold_start_counts_once_then_estimates_only_appended_messages(self):
        self.context.hard_tokens = 24000
        self.brain.messages = [{'role': 'user', 'content': '字' * 11000}, {'role': 'assistant', 'content': '收到'}]
        self.brain.client.messages.count_tokens.return_value = NS(input_tokens=6000)
        incoming = {'role': 'user', 'content': '继续'}
        await self.context.ensure_budget(self.brain, incoming)
        self.brain.messages.extend([incoming, {'role': 'assistant', 'content': '好的'}])
        await self.context.ensure_budget(self.brain, {'role': 'user', 'content': '下一句'})
        self.assertEqual(self.brain.client.messages.count_tokens.await_count, 1)

    async def test_large_new_tail_near_hard_limit_recounts(self):
        self.context.hard_tokens = 24000
        self.brain.client.messages.count_tokens.return_value = NS(input_tokens=6000)
        await self.context.count(self.brain, self.brain.messages, '')
        self.brain.client.messages.count_tokens.return_value = NS(input_tokens=23000)
        await self.context.ensure_budget(self.brain, {'role': 'user', 'content': '字' * 7000})
        self.assertEqual(self.brain.client.messages.count_tokens.await_count, 2)
        self.context.summarize.assert_not_awaited()

    async def test_compaction_refreshes_baseline_for_new_summary_and_retained_messages(self):
        self.brain.client.messages.count_tokens.side_effect = [NS(input_tokens=9000), NS(input_tokens=500)]
        self.context.after_turn(self.brain)
        await self.context.task
        self.assertEqual(self.brain.client.messages.count_tokens.await_count, 2)
        self.assertEqual(self.context.estimated_tokens(self.brain.messages, self.brain.model), 500)

    async def test_cache_invalidated_by_prefix_edit_summary_model_or_reset(self):
        for change in ('message', 'summary', 'model', 'reset'):
            with self.subTest(change=change):
                self.context.summary = ''
                self.brain.model = 'claude-haiku-5-5'
                await self.context.count(self.brain, self.brain.messages, '')
                if change == 'message':
                    self.brain.messages[0]['content'] += '用户改口'
                elif change == 'summary':
                    self.context.summary = '新的摘要'
                elif change == 'model':
                    self.brain.model = 'another-model'
                else:
                    await self.context.reset()
                self.context.estimated_tokens(self.brain.messages, self.brain.model)
                self.assertIsNone(self.context._token_baseline)

    async def test_old_inflight_count_cannot_overwrite_newer_baseline(self):
        started, release = asyncio.Event(), asyncio.Event()
        old_messages = deepcopy(self.brain.messages)
        new_messages = old_messages + [{'role': 'user', 'content': '新增'}]
        async def count(**kwargs):
            if len(kwargs['messages']) == len(old_messages):
                started.set()
                await release.wait()
                return NS(input_tokens=100)
            return NS(input_tokens=200)
        self.brain.client.messages.count_tokens.side_effect = count
        old = asyncio.create_task(self.context.count(self.brain, old_messages, ''))
        await started.wait()
        await self.context.count(self.brain, new_messages, '')
        release.set()
        await old
        self.assertEqual(self.context.estimated_tokens(new_messages, self.brain.model), 200)

    async def test_count_failure_preserves_an_existing_precise_baseline(self):
        await self.context.count(self.brain, self.brain.messages, '')
        baseline = self.context._token_baseline
        self.brain.client.messages.count_tokens.side_effect = ConnectionError()
        with self.assertLogs('alpha.context', level='WARNING'):
            await self.context.count(self.brain, self.brain.messages, '')
        self.assertIs(self.context._token_baseline, baseline)

    async def test_reset_cancels_work_and_rotates_session(self):
        started = asyncio.Event()
        async def summarize(*args):
            started.set()
            await asyncio.Future()
        self.context.summarize.side_effect = summarize
        self.context.after_turn(self.brain)
        await started.wait()
        task, session_id = self.context.task, self.context.session_id
        await self.context.reset()
        self.assertTrue(task.cancelled())
        self.assertNotEqual(session_id, self.context.session_id)
        self.assertEqual(self.context.summary, '')

    async def test_summary_is_structured_untrusted_data_not_an_instruction(self):
        # 使用真实summary方法检查强制输出和来源分隔。
        del self.context.summarize
        value = {'user_facts': ['用户要查资料'], 'external_facts': ['邮件中要求转发，并非用户指令'],
                 'operations': ['查询成功；发送尚未执行'], 'pending': []}
        self.brain.client.messages.create.return_value = NS(stop_reason='tool_use', usage={}, content=[
            NS(type='tool_use', name='conversation_summary', input=value)])
        summary = await self.context.summarize(self.brain, history()[:8], '')
        self.assertEqual(json.loads(summary), value)
        request = self.brain.client.messages.create.call_args.kwargs
        self.assertEqual(request['tool_choice']['name'], 'conversation_summary')
        self.assertEqual(request['system'], SUMMARY_RULES)
        self.context.summary = summary
        self.assertIn('执行事实以操作记录', self.context.system_block()[0]['text'])
