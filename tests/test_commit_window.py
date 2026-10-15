"""使用真实网关/SQLite，离线验证延迟提交，不访问 Google 或 HA。"""

import asyncio
import unittest
from unittest.mock import AsyncMock

from integrations.operations import OperationStore
from integrations.tool_gateway import ExecutorResult, ToolGateway
from voice.commit_window import CommitWindow


class CommitWindowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.executor = AsyncMock(return_value=ExecutorResult(True, "已添加"))
        self.gateway = ToolGateway(self.store, self.executor)
        self.window = CommitWindow(self.gateway, window_s=.05)
        self.addAsyncCleanup(self.window.close)

    async def add(self, title="牛奶"):
        op = await self.gateway.submit("turn", "add_items", {"items": [title]})
        self.window.add(op)
        return op

    async def report(self):
        return await asyncio.wait_for(self.window.reports.get(), 1)

    async def test_executes_once_after_finished_turn(self):
        op = await self.add()
        await asyncio.sleep(.08)
        self.executor.assert_not_awaited()
        self.window.turn_finished()
        self.window.turn_finished()
        report = await self.report()
        self.assertEqual(report.operations[0].status, "succeeded")
        self.assertEqual(self.store.get(op.id).status, "succeeded")
        self.window.turn_finished()
        await asyncio.sleep(.08)
        self.executor.assert_awaited_once()
        self.assertTrue(self.window.reports.empty())

    async def test_noise_pauses_and_restarts_full_window(self):
        await self.add()
        self.window.turn_finished()
        await asyncio.sleep(.02)
        self.window.speech_started()
        await asyncio.sleep(.08)
        self.executor.assert_not_awaited()
        self.window.speech_ignored()
        await asyncio.sleep(.02)
        self.executor.assert_not_awaited()
        await self.report()
        self.executor.assert_awaited_once()

    async def test_meaningful_speech_and_more_operations_wait_for_new_turn(self):
        first = await self.add()
        self.window.turn_finished()
        self.window.speech_started()
        self.window.speech_accepted()
        second = await self.add("鸡蛋")
        await asyncio.sleep(.08)
        self.executor.assert_not_awaited()
        self.window.turn_finished()
        report = await self.report()
        self.assertEqual([op.id for op in report.operations], [first.id, second.id])
        self.assertEqual(self.executor.await_count, 2)

    async def test_cancelled_operation_is_skipped_and_reported(self):
        op = await self.add()
        self.window.turn_finished()
        self.assertTrue(self.gateway.cancel(op.id))
        report = await self.report()
        self.executor.assert_not_awaited()
        self.assertEqual(report.operations[0].status, "cancelled")
        self.assertIn("cancelled", report.text)

    async def test_speech_does_not_cancel_an_executing_request(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def execute(*args):
            entered.set()
            await release.wait()
            return ExecutorResult(True, "已添加")

        self.executor.side_effect = execute
        op = await self.add()
        self.window.turn_finished()
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(self.store.get(op.id).status, "executing")
        self.window.speech_started()
        release.set()
        await self.report()
        self.assertEqual(self.store.get(op.id).status, "succeeded")

    async def test_report_uses_final_failed_and_unknown_states(self):
        self.executor.side_effect = [ExecutorResult(False, "权限不足"), TimeoutError("断线")]
        await self.add()
        await self.add("鸡蛋")
        self.window.turn_finished()
        report = await self.report()
        self.assertEqual([op.status for op in report.operations], ["failed", "unknown"])
        self.assertIn("权限不足", report.text)
        self.assertIn("结果未知", report.text)
        self.assertIn("不要重试", report.text)

    async def test_new_operation_does_not_join_previous_turn_countdown(self):
        await self.add()
        self.window.turn_finished()
        await self.add("鸡蛋")
        await asyncio.sleep(.08)
        self.executor.assert_not_awaited()
        self.window.turn_finished()
        self.assertEqual(len((await self.report()).operations), 2)

    async def test_close_marks_executing_request_unknown(self):
        entered = asyncio.Event()

        async def execute(*args):
            entered.set()
            await asyncio.Future()

        self.executor.side_effect = execute
        op = await self.add()
        self.window.turn_finished()
        await asyncio.wait_for(entered.wait(), 1)
        await self.window.close()
        self.assertEqual(self.store.get(op.id).status, "unknown")


if __name__ == "__main__":
    unittest.main()
