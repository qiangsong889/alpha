"""批次流程离线验收：真实存储/网关/队列，假的远端。"""

import asyncio
import unittest
from unittest.mock import AsyncMock

from core.task_coordinator import TaskCoordinator
from integrations.operations import OperationStore
from integrations.tool_gateway import ToolGateway, ExecutorResult


class TaskCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.remote = AsyncMock(return_value=ExecutorResult(True, "成功"))
        self.gateway = ToolGateway(self.store, self.remote)
        self.flow = TaskCoordinator(self.gateway, window_s=.01, incomplete_s=.04)
        self.addAsyncCleanup(self.flow.close)
        asyncio.create_task(self.flow.run())

    def user(self, name="u1", text="添加牛奶"):
        self.flow.speech_started(name)
        self.flow.speech_stopped(name)
        self.flow.record_text(name, text)
        self.flow.user_turn(name)

    def add(self, items=None):
        return self.flow.add(self.flow.current_turn, "add_items", {
            "list_name": "购物清单", "items": items or ["牛奶"],
        })

    def finish(self, complete=True, **kwargs):
        turn = self.flow.current_turn
        self.flow.finish_collection(complete, **kwargs)
        self.flow.finish_turn(turn)
        self.flow.turn_answered(turn)

    async def notice(self):
        return await asyncio.wait_for(self.flow.notices.get(), 1)

    async def report(self):
        return await asyncio.wait_for(self.flow.reports.get(), 1)

    async def prepare_plan(self):
        self.user()
        self.add(["牛奶", "鸡蛋", "酱油"])
        self.finish()
        notice = await self.notice()
        self.assertEqual(notice.kind, "recap")
        self.remote.assert_not_awaited()
        return notice

    async def test_one_or_two_actions_execute_after_window(self):
        self.user()
        self.add(["牛奶", "鸡蛋"])
        self.remote.assert_not_awaited()
        self.finish()
        report = await self.report()
        self.assertEqual(report.operations[0].status, "succeeded")
        self.remote.assert_awaited_once()

    async def test_device_tool_does_not_bypass_collection(self):
        self.user(text="关灯")
        self.flow.add("u1", "intent__HassTurnOff", {"name": "客厅灯"})
        self.remote.assert_not_awaited()
        await asyncio.sleep(.03)
        self.remote.assert_not_awaited()
        self.finish()
        await self.report()
        self.remote.assert_awaited_once()

    async def test_three_actions_need_playback_and_new_confirmation_then_final_interpretation(self):
        notice = await self.prepare_plan()
        plan_id, version = notice.batch_id, notice.version
        self.assertEqual((await self.flow.confirm(plan_id, version)).reason, "not_played")
        self.flow.recap_played(plan_id, version, False)
        self.assertEqual((await self.flow.confirm(plan_id, version)).reason, "no_user_turn")
        self.user("u2", "对，执行吧")
        self.assertTrue((await self.flow.confirm(plan_id, version)).ok)
        self.remote.assert_not_awaited()
        self.finish()
        await self.report()
        self.remote.assert_awaited_once()

    async def test_confirm_then_wait_in_same_sentence_does_not_execute(self):
        notice = await self.prepare_plan()
        self.flow.recap_played(notice.batch_id, notice.version, False)
        self.user("u2", "对，等等")
        await self.flow.confirm(notice.batch_id, notice.version)
        self.finish(complete=False, question="还要改哪一项？")
        self.assertEqual((await self.notice()).kind, "clarify")
        self.remote.assert_not_awaited()

    async def test_interrupted_recap_rejects_confirmation(self):
        notice = await self.prepare_plan()
        self.flow.recap_played(notice.batch_id, notice.version, True)
        self.user("u2", "对")
        self.assertEqual((await self.flow.confirm(notice.batch_id, notice.version)).reason, "not_played")
        self.remote.assert_not_awaited()

    async def test_incomplete_input_asks_and_missing_interpretation_never_auto_executes(self):
        self.user(text="还有，再把")
        self.add()
        self.flow.finish_turn("u1")
        notice = await self.notice()
        self.assertEqual(notice.kind, "clarify")
        self.assertEqual(self.flow.plan.status, "waiting_for_clarification")
        self.remote.assert_not_awaited()

    async def test_replace_milk_with_soy_before_submission(self):
        self.user()
        result = self.add()
        old_id = self.flow.plan.items[0].task_id
        old_version = self.flow.plan.version
        changed = await self.flow.update(self.flow.batch_id, old_id, {
            "list_name": "购物清单", "items": ["酱油"],
        })
        self.assertTrue(changed.ok)
        self.assertEqual(self.store.get(result.operations[0].id).status, "cancelled")
        self.assertGreater(self.flow.plan.version, old_version)
        self.finish()
        await self.report()
        self.remote.assert_awaited_once_with("add_items", {"list_name": "购物清单", "items": ["酱油"]})

    async def test_late_dependency_and_independent_failure_report(self):
        self.user()
        for name in ("灯", "窗帘", "酱油"):
            self.add([name])
        self.flow.depend(1, [2])
        self.finish()
        notice = await self.notice()
        self.flow.recap_played(notice.batch_id, notice.version, False)
        self.user("u2", "对")
        await self.flow.confirm(notice.batch_id, notice.version)
        self.remote.side_effect = [ExecutorResult(False, "窗帘失败"), ExecutorResult(True, "酱油成功")]
        self.finish()
        report = await self.report()
        self.assertEqual([op.status for op in report.operations], ["cancelled", "failed", "succeeded"])
        self.assertIn("依赖未成功", report.text)

    async def test_new_batch_collected_while_old_request_in_flight(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def remote(name, args):
            if args["items"] == ["牛奶"]:
                entered.set()
                await release.wait()
            return ExecutorResult(True, "成功")
        self.remote.side_effect = remote
        self.user()
        self.add()
        self.finish()
        await asyncio.wait_for(entered.wait(), 1)
        self.user("u2", "加鸡蛋")
        self.add(["鸡蛋"])
        self.finish()
        await asyncio.sleep(.03)
        self.remote.assert_awaited_once()
        release.set()
        await self.report()
        await self.report()
        self.assertEqual(self.remote.await_count, 2)

    async def test_reconnect_preserves_tasks_but_requires_new_playback(self):
        notice = await self.prepare_plan()
        self.flow.recap_played(notice.batch_id, notice.version, False)
        self.flow.on_reconnect()
        self.assertEqual(self.flow.plan.version, notice.version)
        self.user("u2", "对")
        self.assertEqual((await self.flow.confirm(notice.batch_id, notice.version)).reason, "not_played")
        self.assertEqual((await self.notice()).kind, "recap")

    async def test_successful_addition_undo_requires_confirmation_and_preserves_source(self):
        self.remote.return_value = ExecutorResult(True, {"text": "已添加", "structured_content": {
            "version": 1, "status": "succeeded", "list_name": "购物清单", "list_id": "list",
            "created": [{"task_id": "new-milk", "title": "牛奶"}],
        }})
        self.user()
        added = self.add().operations[0]
        self.finish()
        await self.report()
        self.user("u2", "牛奶不要了")
        result = await self.flow.undo(1)
        self.assertTrue(result.ok)
        self.assertEqual(result.operations[0].name, "delete_task")
        self.assertEqual(result.operations[0].arguments, {"list_id": "list", "task_id": "new-milk"})
        self.assertEqual(self.store.get(added.id).status, "succeeded")
        self.finish()
        self.assertEqual((await self.notice()).kind, "recap")
        self.remote.assert_awaited_once()

    async def test_unknown_addition_verified_by_marker_then_can_undo(self):
        self.remote.side_effect = TimeoutError("结果丢失")
        self.user()
        op = self.add().operations[0]
        self.finish()
        await self.report()
        self.assertEqual(self.store.get(op.id).status, "unknown")
        self.remote.side_effect = None
        self.remote.return_value = ExecutorResult(True, {"text": "已核实", "structured_content": {
            "version": 1, "status": "succeeded", "list_name": "购物清单", "list_id": "list",
            "created": [{"task_id": "real-milk", "title": "牛奶"}],
        }})
        self.user("u2", "核实一下，然后取消")
        result = await self.flow.undo(1)
        self.assertTrue(result.ok)
        self.assertEqual(self.store.get(op.id).status, "succeeded")
        self.assertEqual(result.operations[0].arguments["task_id"], "real-milk")

    async def test_not_found_does_not_prove_failed_and_does_not_retry(self):
        self.flow.auto_verify = False
        self.remote.side_effect = TimeoutError()
        self.user()
        op = self.add().operations[0]
        self.finish()
        await self.report()
        self.remote.side_effect = None
        self.remote.return_value = ExecutorResult(True, {"structured_content": {"status": "unknown", "created": []}})
        result = await self.flow.verify_operation(op.id)
        self.assertFalse(result.ok)
        self.assertEqual(self.store.get(op.id).status, "unknown")
        self.assertEqual(self.remote.await_count, 2)

    async def test_unknown_write_is_automatically_verified_without_repeating_write(self):
        async def remote(name, arguments):
            if name == "add_items":
                raise TimeoutError("创建响应丢失")
            self.assertEqual(name, "find_created_items")
            return ExecutorResult(True, {"structured_content": {
                "version": 1, "status": "succeeded", "list_name": "购物清单", "list_id": "list",
                "created": [{"task_id": "created-before-disconnect", "title": "牛奶"}],
            }})
        self.remote.side_effect = remote
        self.user()
        op = self.add().operations[0]
        self.finish()
        report = await self.report()
        self.assertEqual(report.operations[0].status, "succeeded")
        self.assertEqual(self.store.get(op.id).status, "succeeded")
        self.assertEqual(self.remote.await_count, 2)

    async def test_close_cancels_unsubmitted_tasks(self):
        self.user()
        op = self.add().operations[0]
        await self.flow.close()
        self.assertEqual(self.store.get(op.id).status, "cancelled")

    async def test_reminder_can_be_undone_by_receipt(self):
        self.remote.return_value = ExecutorResult(True, {"text": "已创建", "structured_content": {
            "version": 1, "status": "succeeded", "event_id": "exact-event",
        }})
        self.user()
        self.flow.add("u1", "create_reminder", {"title": "开会", "local_time": "2026-11-03T08:00:00"})
        self.finish()
        await self.report()
        self.user("u2", "提醒不要了")
        result = await self.flow.undo(1)
        self.assertEqual(result.operations[0].name, "delete_reminder")
        self.assertEqual(result.operations[0].arguments, {"event_id": "exact-event"})

    async def test_verification_updates_memory_and_delete_absence_proves_undo(self):
        self.user()
        op = self.gateway.register("u1", "delete_reminder", {"event_id": "event"})
        task = self.flow.chain.add(self.flow.batch_id, op)
        self.gateway.store.transition(op.id, "pending", "approved")
        self.gateway.store.transition(op.id, "approved", "executing")
        self.gateway.store.transition(op.id, "executing", "unknown")
        self.flow.chain.observe_operation(task.task_id, self.store.get(op.id))
        self.remote.return_value = ExecutorResult(True, {"structured_content": {"present": False}})
        result = await self.flow.verify_operation(op.id)
        self.assertTrue(result.ok)
        self.assertEqual(self.flow.chain.task(task.task_id).status, "succeeded")

    async def test_chat_without_tasks_does_not_generate_clarification(self):
        self.user(text="你好啊")
        self.flow.finish_turn("u1")
        await asyncio.sleep(.07)
        self.assertTrue(self.flow.notices.empty())

    async def test_past_success_duplicate_requires_new_confirmation(self):
        self.user()
        self.add()
        self.finish()
        await self.report()
        self.user("u2", "再加一次牛奶")
        result = self.add()
        self.assertEqual(result.reason, "duplicate_requires_confirmation")
        self.finish()
        self.assertEqual((await self.notice()).kind, "recap")
        self.remote.assert_awaited_once()

    async def test_update_queued_task_moves_its_dependents_to_next_batch(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def remote(name, args):
            if args["items"] == ["正在执行"]:
                entered.set()
                await release.wait()
            return ExecutorResult(True, "成功")
        self.remote.side_effect = remote
        self.user()
        self.add(["正在执行"])
        self.add(["牛奶"])
        self.add(["之后执行"])
        self.flow.depend(3, [2])
        self.flow._submit(confirmed=True)
        await asyncio.wait_for(entered.wait(), 1)
        self.user("u2", "牛奶改成酱油")
        result = await self.flow.update(self.flow.batch_id, 2, {"list_name": "购物清单", "items": ["酱油"]})
        self.assertEqual(result.reason, "replanned")
        tasks = self.flow.plan.items
        self.assertEqual([task.replaces_task_id for task in tasks], [2, 3])
        self.assertEqual(tasks[1].depends_on, (tasks[0].task_id,))
        self.finish()
        await asyncio.sleep(.03)
        release.set()
        await self.report()
        await self.report()
        self.assertEqual([call.args[1]["items"] for call in self.remote.await_args_list],
                         [["正在执行"], ["酱油"], ["之后执行"]])

    async def test_expired_confirmation_can_be_reopened_by_fresh_playback(self):
        notice = await self.prepare_plan()
        self.flow.recap_played(notice.batch_id, notice.version, False)
        now = self.flow.clock()
        self.flow.clock = lambda: now + 121
        self.assertTrue(self.flow.expire())
        self.user("u2", "对")
        self.assertEqual((await self.flow.confirm(notice.batch_id, notice.version)).reason, "expired")
        self.flow.recap_played(notice.batch_id, notice.version, False)
        self.user("u3", "对")
        self.assertTrue((await self.flow.confirm(notice.batch_id, notice.version)).ok)

    async def test_cancel_inflight_stages_undo_only_after_success(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def remote(*args):
            entered.set()
            await release.wait()
            return ExecutorResult(True, {"text": "添加成功", "structured_content": {
                "version": 1, "status": "succeeded", "list_name": "购物清单", "list_id": "list",
                "created": [{"task_id": "new", "title": "牛奶"}],
            }})
        self.remote.side_effect = remote
        self.user()
        self.add()
        self.finish()
        await asyncio.wait_for(entered.wait(), 1)
        self.user("u2", "牛奶不要了")
        self.assertEqual((await self.flow.undo(1)).reason, "waiting_for_outcome")
        release.set()
        await self.report()
        notice = await self.notice()
        self.assertEqual(notice.kind, "recap")
        self.assertEqual(self.flow.plan.items[0].name, "delete_task")
        self.remote.assert_awaited_once()
