"""内存数据库 + 假 executor；不调用 HA、Google 或模型。"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from integrations.operations import OperationStore
from integrations.tool_gateway import ExecutorResult, ToolGateway, ToolNotSent
from integrations.tool_policy import ToolPolicy
from core.tool_timing import TOOL_TIMING_SINK


class ToolGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_timing_wraps_real_execution_once_and_does_not_leak_arguments(self):
        events = []
        token = TOOL_TIMING_SINK.set(events.append)
        try:
            op = await self.gateway.submit('turn', 'create_reminder', {'title': 'private'})
            self.gateway.approve(op.id)
            await self.gateway.execute(op.id)
            await self.gateway.execute(op.id)
        finally:
            TOOL_TIMING_SINK.reset(token)
        self.assertEqual([e['phase'] for e in events], ['started', 'finished'])
        self.assertEqual([e['operation_id'] for e in events], [op.id, op.id])
        self.assertGreaterEqual(events[1]['duration_s'], 0)
        self.assertNotIn('private', str(events))

    async def test_timing_callback_failure_cannot_change_tool_outcome(self):
        def broken(event):
            raise RuntimeError('metrics unavailable')
        token = TOOL_TIMING_SINK.set(broken)
        try:
            op = await self.gateway.submit('turn', 'list_items', {})
        finally:
            TOOL_TIMING_SINK.reset(token)
        self.assertEqual(op.status, 'succeeded')

    def setUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.executor = AsyncMock(return_value=ExecutorResult(ok=True, result="完成"))
        self.gateway = ToolGateway(self.store, self.executor)

    async def test_unknown_tool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "白名单"):
            await self.gateway.submit("turn", "不存在的工具", {})
        # execute 也检查策略，不允许通过直接创建记录绕过白名单。
        op = self.store.create("turn", "不存在的工具", {})
        self.assertFalse(self.gateway.approve(op.id))
        self.store.transition(op.id, "pending", "approved")
        with self.assertRaisesRegex(ValueError, "白名单"):
            await self.gateway.execute(op.id)
        self.executor.assert_not_awaited()

    async def test_read_only_requests_execute_immediately_without_parameter_dedup(self):
        arguments = {"list_name": "购物清单"}
        first = await self.gateway.submit("first", "list_items", arguments)
        second = await self.gateway.submit("second", "list_items", arguments)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual((first.status, second.status), ("succeeded", "succeeded"))
        self.assertEqual(self.executor.await_count, 2)
        self.executor.assert_awaited_with("list_items", arguments)

    async def test_deferred_write_waits_for_approval(self):
        op = await self.gateway.submit("turn", "create_reminder", {"title": "开会"})
        self.assertEqual(op.status, "pending")
        self.assertEqual((await self.gateway.execute(op.id)).status, "pending")
        self.executor.assert_not_awaited()
        self.assertTrue(self.gateway.approve(op.id))
        self.assertFalse(self.gateway.approve(op.id))
        result = await self.gateway.execute(op.id)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.result, "完成")
        self.assertEqual(self.store.get(op.id), result)

    async def test_confirm_policy_is_not_auto_approved(self):
        with patch("integrations.tool_gateway.get_policy", return_value=ToolPolicy(
            mutates=True, retry_safe=False, commit="confirm",
        )):
            op = await self.gateway.submit("turn", "send_message", {})
            self.assertEqual(op.status, "pending")
            self.executor.assert_not_awaited()

    async def test_pending_and_approved_can_be_cancelled_without_execution(self):
        for approved in (False, True):
            op = await self.gateway.submit("turn", "add_items", {"items": ["牛奶"]})
            if approved:
                self.assertTrue(self.gateway.approve(op.id))
            self.assertTrue(self.gateway.cancel(op.id))
            self.assertFalse(self.gateway.cancel(op.id))
            self.assertFalse(self.gateway.approve(op.id))
            self.assertEqual((await self.gateway.execute(op.id)).status, "cancelled")
        self.executor.assert_not_awaited()

    async def test_parallel_execute_claims_once_before_executor_can_wait(self):
        op = await self.gateway.submit("turn", "create_reminder", {"title": "开会"})
        self.gateway.approve(op.id)
        entered, release = asyncio.Event(), asyncio.Event()

        async def executor(name, arguments):
            self.assertEqual(self.store.get(op.id).status, "executing")
            entered.set()
            await release.wait()
            return ExecutorResult(ok=True, result={"event_id": "created-once"})

        self.executor.side_effect = executor
        first = asyncio.create_task(self.gateway.execute(op.id))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            second = await self.gateway.execute(op.id)
            self.assertEqual(second.status, "executing")
            self.assertFalse(self.gateway.cancel(op.id))
            self.executor.assert_awaited_once()
        finally:
            release.set()
            result = await first
        self.assertEqual(result.status, "succeeded")
        self.assertEqual((await self.gateway.execute(op.id)).result, {"event_id": "created-once"})
        self.executor.assert_awaited_once()

    async def test_explicit_negative_result_and_not_sent_are_failed(self):
        for outcome in (ExecutorResult(ok=False, result="参数错误"), ToolNotSent("连接前校验失败")):
            with self.subTest(outcome=outcome):
                op = await self.gateway.submit("turn", "add_items", {"items": ["牛奶"]})
                self.gateway.approve(op.id)
                if isinstance(outcome, Exception):
                    self.executor.side_effect = outcome
                else:
                    self.executor.side_effect = None
                    self.executor.return_value = outcome
                result = await self.gateway.execute(op.id)
                self.assertEqual(result.status, "failed")
                self.assertEqual(self.store.get(op.id), result)

    async def test_other_exceptions_are_unknown_and_never_automatically_retried(self):
        for error in (TimeoutError("等待超时"), OSError("连接中断"), ValueError("执行结果无法解析")):
            with self.subTest(error=error):
                self.executor.reset_mock()
                self.executor.side_effect = error
                op = await self.gateway.submit("turn", "create_reminder", {})
                self.gateway.approve(op.id)
                result = await self.gateway.execute(op.id)
                self.assertEqual(result.status, "unknown")
                self.assertIn(type(error).__name__, result.result["error"])
                self.assertEqual((await self.gateway.execute(op.id)).status, "unknown")
                self.assertFalse(self.gateway.approve(op.id))
                self.executor.assert_awaited_once()

    async def test_cancellation_after_start_records_unknown_and_propagates(self):
        op = await self.gateway.submit("turn", "create_reminder", {})
        self.gateway.approve(op.id)
        entered = asyncio.Event()

        async def executor(*args):
            entered.set()
            await asyncio.Future()

        self.executor.side_effect = executor
        task = asyncio.create_task(self.gateway.execute(op.id))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.store.get(op.id).status, "unknown")
        self.assertEqual((await self.gateway.execute(op.id)).status, "unknown")
        self.executor.assert_awaited_once()

    async def test_unserializable_success_result_is_unknown(self):
        self.executor.return_value = ExecutorResult(ok=True, result=object())
        result = await self.gateway.submit("turn", "list_items", {})
        self.assertEqual(result.status, "unknown")
        self.assertIn("TypeError", result.result["error"])

    async def test_missing_operation_does_not_call_executor(self):
        self.assertFalse(self.gateway.approve("missing"))
        self.assertFalse(self.gateway.cancel("missing"))
        with self.assertRaises(KeyError):
            await self.gateway.execute("missing")
        self.executor.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
