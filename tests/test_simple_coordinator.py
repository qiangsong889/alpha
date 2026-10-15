"""简化主流程：真实网关和内存 SQLite，不访问外部服务。"""

import asyncio
import unittest
from unittest.mock import AsyncMock

from core.simple_coordinator import SimpleCoordinator
from integrations.operations import OperationStore
from integrations.tool_gateway import ToolGateway, ExecutorResult


class SimpleCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.remote = AsyncMock(return_value=ExecutorResult(True, "成功"))
        self.gateway = ToolGateway(self.store, self.remote)
        self.flow = SimpleCoordinator(self.gateway, window_s=.02, incomplete_s=.06)
        self.addAsyncCleanup(self.flow.close)
        self.user()

    def user(self, key="u1", text="加牛奶"):
        self.flow.speech_started(key)
        self.flow.speech_stopped(key)
        self.flow.record_text(key, text)
        self.flow.user_turn(key)

    async def add(self, title="牛奶"):
        return await self.flow.invoke(self.flow.current_turn, "add_items", {
            "list_name": "购物清单", "items": [title],
        })

    def receipt(self, title="牛奶", task_id="new", list_id="list"):
        return ExecutorResult(True, {"text": "已添加", "structured_content": {
            "version": 1, "status": "succeeded", "list_name": "购物清单", "list_id": list_id,
            "created": [{"task_id": task_id, "title": title}],
        }})

    def inspected(self, title="牛奶", present=True):
        return ExecutorResult(True, {"text": "查询", "structured_content": {"present": present, "title": title, "resource": {"title": title, "notes": "原备注", "status": "needsAction"}}})

    async def report(self, ack=True):
        report = await asyncio.wait_for(self.flow.reports.get(), 1)
        if ack:
            self.flow.report_finished()
        return report

    async def created(self, title="牛奶", task_id="new"):
        self.remote.return_value = self.receipt(title, task_id)
        result = await self.add(title)
        self.flow.turn_finished(self.flow.current_turn)
        await self.report()
        return result

    async def test_device_executes_without_turn_end_or_window(self):
        self.flow.window_s = 100
        result = await self.flow.invoke("u1", "intent__HassTurnOff", {"name": "客厅灯"})
        self.remote.assert_awaited_once()
        self.assertEqual(result["status"], "succeeded")
        self.assertFalse(self.flow.has_pending)

    async def test_write_waits_for_tool_response_end_not_finish_collection(self):
        op = await self.add()
        await asyncio.sleep(.04)
        self.remote.assert_not_awaited()
        self.flow.turn_finished("u1")
        self.assertEqual((await self.report()).operations[0].status, "succeeded")
        self.assertEqual(self.store.get(op["operation_id"]).status, "succeeded")

    async def test_three_writes_do_not_require_confirmation(self):
        for title in ("牛奶", "鸡蛋", "酱油"):
            await self.add(title)
        self.flow.turn_finished("u1")
        report = await self.report()
        self.assertEqual(len(report.operations), 3)
        self.assertEqual(self.remote.await_count, 3)

    async def test_recent_creation_shortcut_requires_exactly_one_known_resource(self):
        self.assertFalse(self.flow.has_unambiguous_recent_creation())
        self.remote.return_value = ExecutorResult(True, {"structured_content": {
            "version": 1, "status": "succeeded", "list_id": "list", "list_name": "购物清单",
            "created": [{"task_id": "milk", "title": "牛奶"}, {"task_id": "bread", "title": "面包"}]}})
        await self.flow.invoke("u1", "add_items", {"list_name": "购物清单", "items": ["牛奶", "面包"]})
        self.flow.turn_finished("u1")
        await self.report()
        self.assertFalse(self.flow.has_unambiguous_recent_creation())

    async def test_model_snapshot_is_compact_and_keeps_creation_ids(self):
        await self.created("面包", "new-bread")
        item = self.flow.snapshot()["recent_created"][0]
        self.assertEqual(item["receipt"]["list_id"], "list")
        self.assertEqual(item["receipt"]["created"][0]["task_id"], "new-bread")
        self.assertNotIn("created_at", item)
        self.assertNotIn("updated_at", item)
        self.assertNotIn("result", item)

    async def test_later_device_action_disables_recent_creation_shortcut(self):
        await self.created("面包", "new-bread")
        self.assertTrue(self.flow.has_unambiguous_recent_creation())
        await self.flow.invoke("u1", "intent__HassTurnOff", {"name": "客厅灯"})
        self.assertFalse(self.flow.has_unambiguous_recent_creation())

    async def test_cancel_pending_cancels_all_before_execution(self):
        first, second = await self.add(), await self.add("面包")
        self.flow.turn_finished("u1")
        self.user("u2", "算了，不加了")
        output = self.flow.cancel_pending()
        self.assertIn("2", output["text"])
        self.flow.turn_finished("u2")
        await asyncio.sleep(.07)
        self.remote.assert_not_awaited()
        for item in (first, second):
            self.assertEqual(self.store.get(item["operation_id"]).status, "cancelled")

    async def test_cancel_does_not_claim_inflight_request_was_cancelled(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def remote(*args):
            entered.set()
            await release.wait()
            return ExecutorResult(True, "已添加")
        self.remote.side_effect = remote
        first, second = await self.add(), await self.add("面包")
        self.flow.turn_finished("u1")
        await asyncio.wait_for(entered.wait(), 1)
        output = self.flow.cancel_pending()
        self.assertIn("已开始执行", output["text"])
        release.set()
        await self.report()
        self.assertEqual(self.store.get(first["operation_id"]).status, "succeeded")
        self.assertEqual(self.store.get(second["operation_id"]).status, "cancelled")
        self.remote.assert_awaited_once()

    async def test_incomplete_utterance_waits_then_asks_without_submitting(self):
        self.user("u2", "还有……")
        await self.add()
        self.flow.turn_finished("u2")
        await asyncio.sleep(.03)
        self.assertTrue(self.flow.notices.empty())
        notice = await asyncio.wait_for(self.flow.notices.get(), 1)
        self.assertTrue(self.flow.is_notice_current(notice))
        self.remote.assert_not_awaited()

    async def test_noise_resumes_timer_without_requiring_another_model_control(self):
        await self.add()
        self.flow.turn_finished("u1")
        self.flow.speech_started("noise")
        await asyncio.sleep(.04)
        self.remote.assert_not_awaited()
        self.flow.speech_ignored()
        await self.report()
        self.remote.assert_awaited_once()

    def delete_executor(self):
        async def remote(name, args):
            if name.startswith("inspect_"):
                return self.inspected("牛奶")
            return ExecutorResult(True, "已删除牛奶")
        self.remote.side_effect = remote

    async def test_original_task_is_deleted_without_confirmation_after_window(self):
        self.delete_executor()
        result = await self.flow.delete_item("task", "old", "list")
        self.assertEqual(result["status"], "pending")
        self.assertNotIn("needs_confirmation", result)
        self.remote.assert_not_awaited()
        self.flow.turn_finished("u1")
        report = await self.report()
        self.assertEqual(report.operations[0].status, "succeeded")
        self.remote.assert_awaited_with("delete_task", {"list_id": "list", "task_id": "old"})
        backup = self.store.deletion_backup(result["operation_id"])
        self.assertEqual(backup["resource"]["notes"], "原备注")

    async def test_cancel_before_deletion_sends_no_delete(self):
        result = await self.flow.delete_item("task", "old", "list")
        self.flow.cancel_pending()
        self.flow.turn_finished("u1")
        await asyncio.sleep(.04)
        self.remote.assert_not_awaited()
        self.assertEqual(self.store.get(result["operation_id"]).status, "cancelled")

    async def test_backup_failure_prevents_deletion(self):
        self.remote.return_value = ExecutorResult(False, "无法读取原数据")
        result = await self.flow.delete_item("task", "old", "list")
        self.flow.turn_finished("u1")
        report = await self.report()
        self.assertEqual(report.operations[0].status, "failed")
        self.assertIn("未执行删除", report.text)
        self.assertEqual(self.remote.await_args.args[0], "inspect_task")
        self.assertIsNone(self.store.deletion_backup(result["operation_id"]))

    async def test_cancel_during_backup_prevents_delete(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def remote(name, args):
            entered.set()
            await release.wait()
            return self.inspected()
        self.remote.side_effect = remote
        result = await self.flow.delete_item("task", "old", "list")
        self.flow.turn_finished("u1")
        await asyncio.wait_for(entered.wait(), 1)
        self.flow.cancel_pending()
        release.set()
        await asyncio.sleep(.04)
        self.remote.assert_awaited_once()
        self.assertEqual(self.store.get(result["operation_id"]).status, "cancelled")

    async def test_own_creation_undo_is_queued_and_exact(self):
        created = await self.created("面包", "new-bread")
        self.user("u2", "刚才的面包不要了")
        self.delete_executor()
        result = await self.flow.undo_recent("面包")
        self.assertEqual(result["status"], "pending")
        self.flow.turn_finished("u2")
        await self.report()
        self.remote.assert_awaited_with("delete_task", {"list_id": "list", "task_id": "new-bread"})
        self.assertEqual(self.store.get(created["operation_id"]).status, "succeeded")

    async def test_repeated_delete_returns_same_operation_without_sending_twice(self):
        self.delete_executor()
        first = await self.flow.delete_item("task", "old", "list")
        second = await self.flow.delete_item("task", "old", "list")
        self.assertEqual(first["operation_id"], second["operation_id"])
        self.flow.turn_finished("u1")
        await self.report()
        third = await self.flow.delete_item("task", "old", "list")
        self.assertEqual(third["status"], "succeeded")
        self.assertEqual([c.args[0] for c in self.remote.await_args_list].count("delete_task"), 1)

    async def test_reminder_also_deletes_without_confirmation_and_saves_details(self):
        event = {"summary": "开会", "start": {"dateTime": "2026-10-16T08:00:00-07:00"},
                 "end": {"dateTime": "2026-10-16T08:15:00-07:00"}, "reminders": {"useDefault": True}}
        async def remote(name, args):
            if name == "inspect_reminder":
                return ExecutorResult(True, {"structured_content": {"present": True, "title": "开会", "resource": event}})
            return ExecutorResult(True, "已删除开会提醒")
        self.remote.side_effect = remote
        result = await self.flow.delete_item("reminder", "old-event")
        self.flow.turn_finished("u1")
        await self.report()
        self.remote.assert_awaited_with("delete_reminder", {"event_id": "old-event"})
        self.assertEqual(self.store.deletion_backup(result["operation_id"])["resource"], event)

    async def test_restoration_uses_stored_snapshot_and_is_not_repeated(self):
        self.delete_executor()
        result = await self.flow.delete_item("task", "old", "list")
        self.flow.turn_finished("u1")
        await self.report()
        self.user("u2", "牛奶删错了，恢复")
        restored = await self.flow.restore_recent("牛奶")
        self.assertEqual(restored["status"], "pending")
        self.assertEqual((await self.flow.restore_recent("牛奶"))["operation_id"], restored["operation_id"])
        self.flow.turn_finished("u2")
        await self.report()
        self.remote.assert_awaited_with("restore_task", {
            "list_id": "list", "resource": self.store.deletion_backup(result["operation_id"])["resource"]})
        self.assertEqual(self.store.get(result["operation_id"]).status, "succeeded")

    async def test_unknown_delete_does_not_restore_blindly(self):
        async def remote(name, args):
            if name == "inspect_task":
                return self.inspected()
            raise TimeoutError("超时")
        self.remote.side_effect = remote
        await self.flow.delete_item("task", "old", "list")
        self.flow.turn_finished("u1")
        self.assertEqual((await self.report()).operations[0].status, "unknown")
        self.user("u2", "恢复")
        self.assertTrue((await self.flow.restore_recent())["is_error"])
        self.assertNotIn("restore_task", [c.args[0] for c in self.remote.await_args_list])

    async def test_batches_execute_serially_and_wait_for_previous_report(self):
        entered, release = asyncio.Event(), asyncio.Event()
        order = []
        async def remote(name, args):
            order.append(args["items"][0])
            if len(order) == 1:
                entered.set()
                await release.wait()
            return ExecutorResult(True, "已添加" + args["items"][0])
        self.remote.side_effect = remote
        await self.add("面包")
        self.flow.turn_finished("u1")
        await asyncio.wait_for(entered.wait(), 1)
        self.user("u2", "加牛奶和鸡蛋")
        await self.add("牛奶")
        await self.add("鸡蛋")
        self.flow.turn_finished("u2")
        await asyncio.sleep(.04)
        release.set()
        first = await self.report(ack=False)
        self.assertEqual(len(first.operations), 1)
        await asyncio.sleep(.04)
        self.assertEqual(order, ["面包"])
        self.flow.report_finished()
        second = await self.report()
        self.assertEqual(len(second.operations), 2)
        self.assertEqual(order, ["面包", "牛奶", "鸡蛋"])

    async def test_cancel_also_cancels_next_queued_batch(self):
        await self.add()
        self.flow.turn_finished("u1")
        await self.report(ack=False)
        self.user("u2", "加面包")
        op = await self.add("面包")
        self.flow.turn_finished("u2")
        await asyncio.sleep(.04)
        self.flow.cancel_pending()
        self.flow.report_finished()
        await asyncio.sleep(.04)
        self.assertEqual(self.store.get(op["operation_id"]).status, "cancelled")
        self.remote.assert_awaited_once()

    async def test_unknown_creation_without_proof_is_not_undone(self):
        self.remote.side_effect = TimeoutError("结果丢失")
        await self.add()
        self.flow.turn_finished("u1")
        await self.report()
        self.user("u2", "撤销")
        output = await self.flow.undo_recent()
        self.assertTrue(output["is_error"])
        self.assertNotIn("delete_task", [call.args[0] for call in self.remote.await_args_list])

    async def test_shutdown_cancels_pending_and_marks_inflight_unknown(self):
        entered = asyncio.Event()
        async def remote(*args):
            entered.set()
            await asyncio.Future()
        self.remote.side_effect = remote
        first, second = await self.add(), await self.add("面包")
        self.flow.turn_finished("u1")
        await asyncio.wait_for(entered.wait(), 1)
        await self.flow.close()
        self.assertEqual(self.store.get(first["operation_id"]).status, "unknown")
        self.assertEqual(self.store.get(second["operation_id"]).status, "cancelled")
