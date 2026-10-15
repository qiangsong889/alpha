"""真实 gateway 与 SQLite，假的远端 executor；不访问设备或网络。"""

import asyncio
from dataclasses import replace
import unittest
from unittest.mock import AsyncMock, patch

from core.task_batch import TaskChain
from integrations.batch_executor import BatchExecutor
from integrations.operations import OperationStore
from integrations.tool_gateway import ExecutorResult, ToolGateway


class BatchExecutorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.remote = AsyncMock(return_value=ExecutorResult(True, "成功"))
        self.gateway = ToolGateway(self.store, self.remote)
        self.chain = TaskChain()
        self.batch = self.chain.new_batch()
        self.executor = BatchExecutor(self.chain, self.gateway)
        self.addAsyncCleanup(self.executor.close)

    def add(self, title="牛奶", batch=None):
        op = self.store.create("turn", "add_items", {"list_name": "购物清单", "items": [title]})
        return self.chain.add((batch or self.batch).id, op)

    def enqueue(self, batch=None):
        batch = batch or self.batch
        self.assertTrue(self.chain.queue(batch.id, version=self.chain.batch(batch.id).version,
                                         confirmed=True))
        return self.executor.enqueue(batch.id)

    def start(self):
        return asyncio.create_task(self.executor.run())

    async def report(self):
        return await asyncio.wait_for(self.executor.reports.get(), 1)

    async def test_fifo_batches_and_duplicate_enqueue_execute_once(self):
        self.add("牛奶")
        self.assertTrue(self.enqueue())
        self.assertFalse(self.executor.enqueue(self.batch.id))
        second = self.chain.new_batch()
        self.add("鸡蛋", second)
        self.enqueue(second)
        self.start()
        first_report, second_report = await self.report(), await self.report()
        self.assertEqual([first_report.batch_id, second_report.batch_id], [self.batch.id, second.id])
        self.assertEqual([call.args[1]["items"] for call in self.remote.await_args_list],
                         [["牛奶"], ["鸡蛋"]])
        self.assertTrue(all(item.task.status == "succeeded" for item in first_report.tasks))

    async def test_collect_and_queue_next_batch_while_previous_request_waits(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def remote(name, args):
            if args["items"] == ["牛奶"]:
                entered.set()
                await release.wait()
            return ExecutorResult(True, "成功")

        self.remote.side_effect = remote
        first = self.add()
        self.enqueue()
        self.start()
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(self.chain.task(first.task_id).status, "executing")
        second = self.chain.new_batch()
        new = self.add("鸡蛋", second)
        self.enqueue(second)
        self.remote.assert_awaited_once()
        self.assertEqual(self.chain.task(new.task_id).status, "queued")
        release.set()
        await self.report()
        await self.report()
        self.assertEqual(self.remote.await_count, 2)

    async def test_next_batch_is_not_executed_until_explicitly_submitted(self):
        self.add()
        self.enqueue()
        second = self.chain.new_batch()
        new = self.add("鸡蛋", second)
        self.start()
        await self.report()
        self.assertEqual(self.chain.task(new.task_id).status, "pending")
        self.remote.assert_awaited_once()

    async def test_cancel_queued_task_syncs_store_and_never_sends(self):
        task = self.add()
        self.enqueue()
        self.assertTrue(self.executor.cancel(task.task_id))
        self.assertFalse(self.executor.cancel(task.task_id))
        self.start()
        report = await self.report()
        self.assertEqual(report.tasks[0].task.status, "cancelled")
        self.assertEqual(self.store.get(task.operation_id).status, "cancelled")
        self.remote.assert_not_awaited()

    async def test_cancel_remaining_task_while_first_is_executing(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def remote(*args):
            entered.set()
            await release.wait()
            return ExecutorResult(True, "完成")

        self.remote.side_effect = remote
        first, second = self.add(), self.add("鸡蛋")
        self.enqueue()
        self.start()
        await asyncio.wait_for(entered.wait(), 1)
        self.assertFalse(self.executor.cancel(first.task_id))
        self.assertTrue(self.executor.cancel(second.task_id))
        release.set()
        report = await self.report()
        self.assertEqual([item.task.status for item in report.tasks], ["succeeded", "cancelled"])
        self.remote.assert_awaited_once()

    async def test_cancel_after_success_does_not_claim_undo(self):
        task = self.add()
        self.enqueue()
        self.start()
        await self.report()
        self.assertFalse(self.executor.cancel(task.task_id))
        self.assertEqual(self.chain.cancellation_effect(task.task_id), "compensate")
        self.assertEqual(self.store.get(task.operation_id).status, "succeeded")

    async def test_cancel_remaining_includes_waiting_batches_but_not_current_request(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def remote(*args):
            entered.set()
            await release.wait()
            return ExecutorResult(True, "成功")

        self.remote.side_effect = remote
        first, second = self.add(), self.add("鸡蛋")
        self.enqueue()
        later = self.chain.new_batch()
        third = self.add("酱油", later)
        self.enqueue(later)
        self.start()
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(self.executor.cancel_remaining(), (second.task_id, third.task_id))
        self.assertEqual(self.chain.task(first.task_id).status, "executing")
        release.set()
        await self.report()
        await self.report()
        self.remote.assert_awaited_once()

    async def test_replacement_history_is_cancelled_before_execution(self):
        milk = self.add()
        replacement = self.store.create("turn", "add_items", {"list_name": "购物清单", "items": ["酱油"]})
        soy = self.chain.replace(milk.task_id, replacement)
        self.enqueue()
        self.start()
        report = await self.report()
        self.assertEqual([item.task.status for item in report.tasks], ["cancelled", "succeeded"])
        self.assertEqual(self.store.get(milk.operation_id).status, "cancelled")
        self.remote.assert_awaited_once_with("add_items", soy.arguments)

    async def test_late_dependency_runs_curtain_before_light(self):
        light, curtain = self.add("灯"), self.add("窗帘")
        self.chain.set_dependencies(light.task_id, (curtain.task_id,))
        self.enqueue()
        self.start()
        await self.report()
        self.assertEqual([call.args[1]["items"] for call in self.remote.await_args_list],
                         [["窗帘"], ["灯"]])

    async def test_failed_or_unknown_dependency_skips_only_dependents(self):
        for outcome, status in ((ExecutorResult(False, "失败"), "failed"),
                                (TimeoutError("断线"), "unknown")):
            with self.subTest(status=status):
                chain = TaskChain()
                batch = chain.new_batch()
                remote = AsyncMock(side_effect=[outcome, ExecutorResult(True, "酱油已加")])
                gateway = ToolGateway(self.store, remote)
                executor = BatchExecutor(chain, gateway)
                self.addAsyncCleanup(executor.close)
                tasks = []
                for title in ("窗帘", "灯", "酱油"):
                    op = self.store.create("turn", "add_items", {"list_name": "购物清单", "items": [title]})
                    tasks.append(chain.add(batch.id, op))
                chain.set_dependencies(tasks[1].task_id, (tasks[0].task_id,))
                chain.queue(batch.id, version=4, confirmed=True)
                executor.enqueue(batch.id)
                asyncio.create_task(executor.run())
                report = await asyncio.wait_for(executor.reports.get(), 1)
                self.assertEqual([item.task.status for item in report.tasks],
                                 [status, "skipped", "succeeded"])
                self.assertIn("没有执行", report.tasks[1].text)
                self.assertEqual(self.store.get(tasks[1].operation_id).status, "cancelled")
                self.assertEqual(remote.await_count, 2)
                if status == "unknown":
                    self.assertIn("不要重试", report.text)
                await executor.close()

    async def test_dependency_across_queued_batches(self):
        previous = self.add()
        self.enqueue()
        second = self.chain.new_batch()
        op = self.store.create("turn", "add_items", {"list_name": "购物清单", "items": ["鸡蛋"]})
        self.chain.add(second.id, op, depends_on=(previous.task_id,))
        self.enqueue(second)
        self.start()
        await self.report()
        await self.report()
        self.assertEqual(self.remote.await_count, 2)

    async def test_missing_predecessor_rejected_before_enqueue(self):
        previous = self.add()
        self.chain.queue(self.batch.id, version=1)
        second = self.chain.new_batch()
        op = self.store.create("turn", "add_items", {"list_name": "购物清单", "items": ["鸡蛋"]})
        self.chain.add(second.id, op, depends_on=(previous.task_id,))
        self.chain.queue(second.id, version=1)
        with self.assertRaises(ValueError):
            self.executor.enqueue(second.id)
        self.remote.assert_not_awaited()

    async def test_gateway_cancel_outside_queue_is_observed(self):
        task = self.add()
        self.enqueue()
        self.gateway.cancel(task.operation_id)
        self.start()
        report = await self.report()
        self.assertEqual(report.tasks[0].task.status, "cancelled")
        self.remote.assert_not_awaited()

    async def test_shutdown_unknown_current_and_cancels_all_unsent_batches(self):
        entered = asyncio.Event()

        async def remote(*args):
            entered.set()
            await asyncio.Future()

        self.remote.side_effect = remote
        first, second = self.add(), self.add("鸡蛋")
        self.enqueue()
        next_batch = self.chain.new_batch()
        third = self.add("酱油", next_batch)
        self.enqueue(next_batch)
        runner = self.start()
        await asyncio.wait_for(entered.wait(), 1)
        await self.executor.close()
        self.assertTrue(runner.done())
        self.assertEqual([self.store.get(task.operation_id).status for task in (first, second, third)],
                         ["unknown", "cancelled", "cancelled"])
        self.assertEqual([self.chain.task(task.task_id).status for task in (first, second, third)],
                         ["unknown", "cancelled", "cancelled"])
        reports = [await self.report(), await self.report()]
        self.assertIn("结果未知", reports[0].text)
        self.assertEqual(reports[1].tasks[0].task.status, "cancelled")
        self.remote.assert_awaited_once()
        await self.executor.close()
        self.assertTrue(self.executor.reports.empty())

    async def test_close_before_run_cancels_queued_records(self):
        task = self.add()
        self.enqueue()
        await self.executor.close()
        self.assertEqual(self.store.get(task.operation_id).status, "cancelled")
        self.remote.assert_not_awaited()

    async def test_collecting_batch_and_mismatched_arguments_rejected(self):
        task = self.add()
        with self.assertRaises(ValueError):
            self.executor.enqueue(self.batch.id)
        self.chain.queue(self.batch.id, version=1)
        with patch.object(self.store, "get", return_value=replace(
            self.store.get(task.operation_id), arguments={"items": ["错误内容"]},
        )):
            with self.assertRaises(ValueError):
                self.executor.enqueue(self.batch.id)
        self.remote.assert_not_awaited()

    async def test_only_one_runner_and_closed_queue_rejects_enqueue(self):
        runner = self.start()
        await asyncio.sleep(0)
        with self.assertRaises(RuntimeError):
            await self.executor.run()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        self.add()
        self.chain.queue(self.batch.id, version=1)
        with self.assertRaises(RuntimeError):
            self.executor.enqueue(self.batch.id)


if __name__ == "__main__":
    unittest.main()
