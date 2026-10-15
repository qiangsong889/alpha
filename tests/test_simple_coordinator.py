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
        return ExecutorResult(True, {"text": "查询", "structured_content": {"present": present, "title": title}})

    async def report(self):
        return await asyncio.wait_for(self.flow.reports.get(), 1)

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

    async def test_own_creation_undo_is_direct_and_exact(self):
        created = await self.created("面包", "new-bread")
        self.user("u2", "刚才的面包不要了")
        self.remote.return_value = ExecutorResult(True, "删除成功")
        result = await self.flow.undo_recent("面包")
        self.assertFalse(result["is_error"])
        self.assertIn("已经删掉了面包", result["text"])
        self.remote.assert_awaited_with("delete_task", {"list_id": "list", "task_id": "new-bread"})
        self.assertEqual(self.store.get(created["operation_id"]).status, "succeeded")

    async def test_direct_delete_of_own_created_id_also_skips_confirmation(self):
        await self.created()
        self.user("u2", "删掉它")
        self.remote.return_value = ExecutorResult(True, "成功")
        result = await self.flow.delete_item("task", "new", "list")
        self.assertEqual(result["status"], "succeeded")
        self.assertNotIn("needs_confirmation", result)

    async def test_repeated_undo_does_not_send_duplicate_delete(self):
        await self.created()
        self.user("u2", "撤销")
        self.remote.return_value = ExecutorResult(True, "删除成功")
        await self.flow.undo_recent()
        await self.flow.undo_recent()
        self.assertEqual([call.args[0] for call in self.remote.await_args_list].count("delete_task"), 1)

    async def test_old_process_creation_does_not_become_owned(self):
        await self.created()
        other = SimpleCoordinator(self.gateway)
        self.addAsyncCleanup(other.close)
        other.user_turn("other-u1")
        self.remote.return_value = self.inspected()
        result = await other.delete_item("task", "new", "list")
        self.assertTrue(result["needs_confirmation"])
        self.assertNotIn("delete_task", [call.args[0] for call in self.remote.await_args_list])

    async def test_reading_existing_items_does_not_grant_ownership(self):
        self.remote.return_value = ExecutorResult(True, {"structured_content": {
            "list_id": "list", "items": [{"task_id": "old", "title": "牛奶"}],
        }})
        await self.flow.invoke("u1", "list_items", {"list_name": "购物清单"})
        self.remote.return_value = self.inspected()
        self.assertTrue((await self.flow.delete_item("task", "old", "list"))["needs_confirmation"])

    async def test_first_confirmed_true_still_requests_confirmation(self):
        self.remote.return_value = self.inspected()
        result = await self.flow.delete_item("task", "old", "list", confirmed=True)
        self.assertTrue(result["needs_confirmation"])
        self.assertEqual([call.args[0] for call in self.remote.await_args_list], ["inspect_task"])

    async def test_same_user_turn_cannot_confirm_even_with_true(self):
        self.remote.return_value = self.inspected()
        await self.flow.delete_item("task", "old", "list")
        result = await self.flow.delete_item("task", "old", "list", confirmed=True)
        self.assertTrue(result["needs_confirmation"])
        self.remote.assert_awaited_once()

    async def test_new_turn_without_confirmed_true_does_not_delete(self):
        self.remote.return_value = self.inspected()
        await self.flow.delete_item("task", "old", "list")
        self.user("u2", "为什么要删？")
        self.assertTrue((await self.flow.delete_item("task", "old", "list"))["needs_confirmation"])
        self.remote.assert_awaited_once()

    async def test_fresh_turn_and_true_delete_same_target(self):
        self.remote.return_value = self.inspected()
        await self.flow.delete_item("task", "old", "list")
        self.user("u2", "对，删掉")
        self.remote.return_value = ExecutorResult(True, "已删除")
        output = await self.flow.delete_item("task", "old", "list", confirmed=True)
        self.assertEqual(output["status"], "succeeded")
        self.remote.assert_awaited_with("delete_task", {"list_id": "list", "task_id": "old"})

    async def test_confirmation_cannot_be_moved_to_another_target_or_list(self):
        self.remote.return_value = self.inspected()
        await self.flow.delete_item("task", "old", "list")
        self.user("u2", "对")
        for task, tasklist in (("different", "list"), ("old", "other-list")):
            output = await self.flow.delete_item("task", task, tasklist, confirmed=True)
            self.assertTrue(output["needs_confirmation"])
        self.assertNotIn("delete_task", [call.args[0] for call in self.remote.await_args_list])

    async def test_noise_does_not_count_as_new_confirmation_input(self):
        self.remote.return_value = self.inspected()
        await self.flow.delete_item("task", "old", "list")
        self.flow.speech_started("noise")
        self.flow.speech_ignored()
        output = await self.flow.delete_item("task", "old", "list", confirmed=True)
        self.assertTrue(output["needs_confirmation"])

    async def test_speech_started_before_challenge_does_not_count_as_after_it(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def inspect(*args):
            entered.set()
            await release.wait()
            return self.inspected()
        self.remote.side_effect = inspect
        first = asyncio.create_task(self.flow.delete_item("task", "old", "list"))
        await asyncio.wait_for(entered.wait(), 1)
        self.flow.speech_started("u2")
        release.set()
        await first
        self.flow.record_text("u2", "对")
        self.flow.user_turn("u2")
        result = await self.flow.delete_item("task", "old", "list", confirmed=True)
        self.assertTrue(result["needs_confirmation"])
        self.remote.assert_awaited_once()

    async def test_reconnect_clears_delete_confirmation_but_keeps_own_scope(self):
        await self.created()
        self.user("u2", "删除旧条目")
        self.remote.return_value = self.inspected()
        await self.flow.delete_item("task", "old", "list")
        self.flow.on_reconnect()
        self.user("new-session-u1", "对")
        self.assertTrue((await self.flow.delete_item("task", "old", "list", confirmed=True))["needs_confirmation"])
        self.assertIn(("task", "list", "new"), self.flow._owned())

    async def test_own_reminder_undo_uses_receipt_id_without_confirming(self):
        self.remote.return_value = ExecutorResult(True, {"structured_content": {
            "version": 1, "status": "succeeded", "event_id": "my-event",
        }})
        await self.flow.invoke("u1", "create_reminder", {"title": "开会", "local_time": "2026-11-03T08:00:00"})
        self.flow.turn_finished("u1")
        await self.report()
        self.user("u2", "提醒不要了")
        self.remote.return_value = ExecutorResult(True, "已删除")
        output = await self.flow.undo_recent("开会")
        self.assertFalse(output["is_error"])
        self.remote.assert_awaited_with("delete_reminder", {"event_id": "my-event"})

    async def test_original_reminder_requires_new_input_and_true(self):
        self.remote.return_value = self.inspected("原有会议")
        self.assertTrue((await self.flow.delete_item("reminder", "old-event"))["needs_confirmation"])
        self.user("u2", "对，删掉")
        self.remote.return_value = ExecutorResult(True, "删除成功")
        output = await self.flow.delete_item("reminder", "old-event", confirmed=True)
        self.assertEqual(output["status"], "succeeded")

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
