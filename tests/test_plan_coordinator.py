"""确认规则的离线测试：真实网关/内存数据库，假的时间和 executor。"""

import asyncio
from dataclasses import replace
from datetime import datetime
import unittest
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from core.plan_coordinator import PlanCoordinator
from integrations.operations import OperationStore
from integrations.tool_gateway import ExecutorResult, ToolGateway


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class PlanCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.executor = AsyncMock(return_value=ExecutorResult(True, "已添加"))
        self.gateway = ToolGateway(self.store, self.executor)
        self.clock = FakeClock()
        self.coordinator = PlanCoordinator(self.gateway, self.clock)

    async def collect(self, title="牛奶"):
        op = await self.gateway.submit("request", "add_items", {
            "list_name": "购物清单", "items": [title],
        })
        self.assertTrue(self.coordinator.collect(op).ok)
        return op

    def played(self, interrupted=False):
        plan = self.coordinator.plan
        return self.coordinator.recap_played(plan.id, plan.version, interrupted)

    async def confirm(self, duplicates_intended=False):
        plan = self.coordinator.plan
        return await self.coordinator.confirm(plan.id, plan.version, duplicates_intended)

    async def collect_reminder(self):
        op = await self.gateway.submit("request", "create_reminder", {
            "title": "开会", "local_time": "2026-10-16T08:00:00",
        })
        self.assertTrue(self.coordinator.collect(op).ok)
        return op

    async def test_parallel_update_and_remove_use_stable_ids_despite_position_change(self):
        milk, reminder = await self.collect(), await self.collect_reminder()
        self.played()
        self.coordinator.user_turn("change-nine-and-remove-milk")
        plan = self.coordinator.plan
        submitted, removed = asyncio.Event(), asyncio.Event()
        submit = self.gateway.submit

        async def delayed_submit(*args):
            submitted.set()
            await removed.wait()
            return await submit(*args)

        async def remove_milk():
            await submitted.wait()
            result = self.coordinator.remove(plan.id, 1)
            removed.set()
            return result

        with patch.object(self.gateway, "submit", side_effect=delayed_submit):
            updated, deleted = await asyncio.wait_for(asyncio.gather(
                self.coordinator.update(plan.id, 2, {
                    "title": "开会", "local_time": "2026-10-16T09:00:00",
                }), remove_milk(),
            ), 1)
        self.assertTrue(updated.ok)
        self.assertTrue(deleted.ok)
        self.assertEqual(plan.version, 4)
        self.assertEqual([item.item_id for item in plan.items], [2])
        self.assertEqual(self.store.get(milk.id).status, "cancelled")
        self.assertEqual(self.store.get(reminder.id).status, "cancelled")
        now = datetime(2026, 10, 15, 9, tzinfo=ZoneInfo("America/Los_Angeles"))
        self.assertEqual(plan.recap(now), "我确认一下：② 明天早上 9:00 提醒你开会。对吗？")
        self.executor.assert_not_awaited()

    async def test_sequential_edits_in_one_user_turn_do_not_consume_confirmation(self):
        await self.collect()
        await self.collect_reminder()
        self.played()
        self.coordinator.user_turn("change-nine-and-remove-milk")
        plan = self.coordinator.plan
        updated = await self.coordinator.update(plan.id, 2, {
            "title": "开会", "local_time": "2026-10-16T09:00:00",
        })
        deleted = self.coordinator.remove(plan.id, 1)
        self.assertTrue(updated.ok)
        self.assertTrue(deleted.ok)
        self.assertEqual((await self.confirm()).reason, "not_played")
        # 播完新复述后，原来那句话仍然早于新复述；必须再收到新的用户输入。
        self.played()
        self.assertEqual((await self.confirm()).reason, "no_user_turn")
        self.coordinator.user_turn("yes-after-new-recap")
        self.assertTrue((await self.confirm()).ok)
        self.executor.assert_awaited_once()

    async def test_explicit_duplicate_intent_executes_both_operations(self):
        first, second = await self.collect(), await self.collect()
        self.played()
        self.coordinator.user_turn("both-bottles")
        self.assertEqual((await self.confirm(False)).reason, "quantity_ambiguous")
        self.executor.assert_not_awaited()
        result = await self.confirm(True)
        self.assertTrue(result.ok)
        self.assertEqual([op.id for op in result.operations], [first.id, second.id])
        self.assertEqual([op.status for op in result.operations], ["succeeded", "succeeded"])
        self.assertEqual(self.executor.await_count, 2)

    async def test_item_numbers_survive_removal_replacement_and_new_collection(self):
        await self.collect()
        await self.collect_reminder()
        plan = self.coordinator.plan
        self.coordinator.user_turn("edits")
        self.assertTrue(self.coordinator.remove(plan.id, 1).ok)
        self.assertEqual(plan.items[0].item_id, 2)
        self.assertTrue((await self.coordinator.update(plan.id, 2, {
            "title": "开会", "local_time": "2026-10-16T09:00:00",
        })).ok)
        self.assertEqual(plan.items[0].item_id, 2)
        await self.collect("鸡蛋")
        self.assertEqual([item.item_id for item in plan.items], [2, 3])
        self.assertEqual(self.coordinator.remove(plan.id, 1).reason, "operation_unavailable")

    async def test_update_to_already_removed_item_discards_replacement_record(self):
        old = await self.collect()
        self.coordinator.user_turn("edit")
        plan = self.coordinator.plan
        submit = self.gateway.submit

        async def remove_before_returning(*args):
            result = await submit(*args)
            self.assertTrue(self.coordinator.remove(plan.id, 1).ok)
            return result

        with patch.object(self.gateway, "submit", side_effect=remove_before_returning):
            result = await self.coordinator.update(plan.id, 1, {
                "list_name": "购物清单", "items": ["鸡蛋"],
            })
        self.assertEqual(result.reason, "operation_unavailable")
        self.assertEqual(plan.items, ())
        rows = self.store._db.execute("SELECT status FROM operations ORDER BY rowid").fetchall()
        self.assertEqual([row["status"] for row in rows], ["cancelled", "cancelled"])
        self.assertEqual(self.store.get(old.id).status, "cancelled")

    async def test_concurrent_updates_of_same_item_reject_stale_replacement(self):
        await self.collect()
        self.coordinator.user_turn("edit")
        plan = self.coordinator.plan
        submit = self.gateway.submit
        both_started = asyncio.Event()
        count = 0

        async def delayed_submit(*args):
            nonlocal count
            count += 1
            if count == 2:
                both_started.set()
            await both_started.wait()
            return await submit(*args)

        with patch.object(self.gateway, "submit", side_effect=delayed_submit):
            results = await asyncio.wait_for(asyncio.gather(*(
                self.coordinator.update(plan.id, 1, {"list_name": "购物清单", "items": [title]})
                for title in ("鸡蛋", "面包")
            )), 1)
        self.assertEqual(sum(result.ok for result in results), 1)
        self.assertEqual({result.reason for result in results}, {"updated", "operation_unavailable"})
        self.assertEqual(plan.version, 2)
        rows = self.store._db.execute("SELECT status FROM operations").fetchall()
        self.assertEqual(sorted(row["status"] for row in rows), ["cancelled", "cancelled", "pending"])

    async def test_every_control_result_contains_latest_version_and_detached_snapshot(self):
        await self.collect()
        plan = self.coordinator.plan
        denied = self.coordinator.remove(plan.id, 1)
        self.assertEqual(denied.reason, "no_user_turn")
        self.assertEqual(denied.version, plan.version)
        self.assertEqual(denied.snapshot, self.coordinator.snapshot())
        self.coordinator.user_turn("edit")
        updated = await self.coordinator.update(plan.id, 1, {
            "list_name": "购物清单", "items": ["鸡蛋"],
        })
        self.assertEqual(updated.version, 2)
        self.assertEqual(updated.snapshot["items"][0]["item_id"], 1)
        self.assertEqual(updated.snapshot, self.coordinator.snapshot())
        updated.snapshot["items"][0]["arguments"]["items"].append("面包")
        self.assertEqual(plan.items[0].arguments["items"], ["鸡蛋"])
        deleted = self.coordinator.remove(plan.id, 1)
        self.assertEqual(deleted.version, 3)
        self.assertEqual(deleted.snapshot["items"], [])
        cancelled = self.coordinator.cancel_plan(plan.id)
        self.assertEqual(cancelled.snapshot["status"], "cancelled")

    async def test_non_boolean_duplicate_intent_does_not_bypass_check(self):
        await self.collect()
        await self.collect()
        self.played()
        self.coordinator.user_turn("yes")
        for value in ("false", 1, None):
            self.assertEqual((await self.confirm(value)).reason, "invalid_arguments")
        self.executor.assert_not_awaited()

    async def test_interrupted_recap_cannot_be_confirmed(self):
        await self.collect()
        self.assertEqual(self.played(True).reason, "not_played")
        self.coordinator.user_turn("yes")
        self.assertEqual((await self.confirm()).reason, "not_played")
        self.executor.assert_not_awaited()

    async def test_model_cannot_confirm_without_user_input(self):
        await self.collect()
        self.played()
        self.assertEqual((await self.confirm()).reason, "no_user_turn")

    async def test_old_version_cannot_be_confirmed(self):
        await self.collect()
        self.played()
        self.coordinator.user_turn("yes")
        plan = self.coordinator.plan
        result = await self.coordinator.confirm(plan.id, plan.version - 1)
        self.assertEqual(result.reason, "version_mismatch")
        self.executor.assert_not_awaited()

    async def test_confirm_executes_each_item_once_and_consumes_turn(self):
        first, second = await self.collect(), await self.collect("鸡蛋")
        self.played()
        self.coordinator.user_turn("yes")
        result = await self.confirm()
        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "completed")
        self.assertEqual([op.id for op in result.operations], [first.id, second.id])
        self.assertEqual([op.status for op in result.operations], ["succeeded", "succeeded"])
        self.assertEqual(self.coordinator.plan.status, "done")
        self.assertEqual((await self.confirm()).reason, "turn_used")
        self.assertEqual(self.executor.await_count, 2)

    async def test_answered_question_cannot_be_reused_for_confirmation(self):
        await self.collect()
        self.played()
        self.coordinator.user_turn("what-is-item-two")
        self.coordinator.turn_answered("what-is-item-two")
        self.assertEqual((await self.confirm()).reason, "no_user_turn")
        self.executor.assert_not_awaited()

    async def test_update_cancels_old_record_changes_version_and_requires_new_recap(self):
        old = await self.collect()
        self.played()
        self.coordinator.user_turn("change-milk")
        plan = self.coordinator.plan
        version = plan.version
        result = await self.coordinator.update(plan.id, 1, {'list_name': '购物清单', 'items': ['鸡蛋']})
        self.assertTrue(result.ok)
        self.assertEqual(plan.version, version + 1)
        self.assertEqual(plan.status, "collecting")
        self.assertEqual(self.store.get(old.id).status, "cancelled")
        self.assertEqual(result.operations[0].status, "pending")
        self.assertEqual(plan.items[0].op_id, result.operations[0].id)
        self.assertEqual((await self.coordinator.confirm(plan.id, version)).reason, "version_mismatch")
        self.coordinator.user_turn("new-yes")
        self.assertEqual((await self.confirm()).reason, "not_played")
        # 复述之前的话也不能用于确认这次复述。
        self.played()
        self.assertEqual((await self.confirm()).reason, "no_user_turn")
        self.coordinator.user_turn("after-new-recap")
        self.assertTrue((await self.confirm()).ok)

    async def test_partial_failure_and_unknown_are_reported_as_actual_states(self):
        for second, expected in ((ExecutorResult(False, "权限不足"), "failed"),
                                 (TimeoutError("断线"), "unknown")):
            with self.subTest(expected=expected):
                await self.collect()
                await self.collect("鸡蛋")
                self.executor.side_effect = [ExecutorResult(True, "牛奶已添加"), second]
                self.played()
                self.coordinator.user_turn(f"yes-{expected}")
                result = await self.confirm()
                self.assertTrue(result.ok)  # 控制请求被接受，不表示全部成功。
                self.assertEqual([op.status for op in result.operations], ["succeeded", expected])
                self.assertEqual(self.coordinator.plan.status, "done")

    async def test_expiration_at_120_seconds_survives_explicit_expire_call(self):
        await self.collect()
        self.played()
        self.coordinator.user_turn("yes")
        self.clock.now += 120
        self.assertTrue(self.coordinator.expire())
        self.assertEqual(self.coordinator.plan.status, "collecting")
        self.assertEqual((await self.confirm()).reason, "expired")
        self.assertFalse(self.coordinator.expire())
        self.assertEqual((await self.confirm()).reason, "expired")
        self.played()
        self.assertEqual((await self.confirm()).reason, "no_user_turn")
        self.coordinator.user_turn("new-yes")
        self.assertTrue((await self.confirm()).ok)

    async def test_confirm_automatically_checks_timeout_and_accepts_before_deadline(self):
        await self.collect()
        self.played()
        self.clock.now += 121
        self.coordinator.user_turn("late")
        self.assertEqual((await self.confirm()).reason, "expired")
        self.played()
        self.clock.now += 119.9
        self.coordinator.user_turn("in-time")
        self.assertTrue((await self.confirm()).ok)

    async def test_invalid_update_does_not_cancel_old_or_change_plan(self):
        old = await self.collect()
        self.played()
        self.coordinator.user_turn("change")
        before = self.coordinator.snapshot()
        plan = self.coordinator.plan
        result = await self.coordinator.update(plan.id, 1, {"items": ["鸡蛋"]})
        self.assertEqual(result.reason, "invalid_arguments")
        self.assertEqual(self.store.get(old.id).status, "pending")
        self.assertEqual(self.coordinator.snapshot(), before)
        self.assertEqual(self.store._db.execute("SELECT COUNT(*) FROM operations").fetchone()[0], 1)
        self.assertTrue((await self.confirm()).ok)  # 校验失败不消耗这句话。

    async def test_old_plan_playback_callback_cannot_unlock_new_plan(self):
        await self.collect()
        old_id, old_version = self.coordinator.plan.id, self.coordinator.plan.version
        self.coordinator.user_turn("cancel-old")
        self.assertTrue(self.coordinator.cancel_plan(old_id).ok)
        await self.collect("鸡蛋")
        self.assertEqual(self.coordinator.plan.version, old_version)
        result = self.coordinator.recap_played(old_id, old_version, False)
        self.assertEqual(result.reason, "plan_mismatch")
        self.coordinator.user_turn("yes-new")
        self.assertEqual((await self.confirm()).reason, "not_played")

    async def test_quantity_ambiguity_requires_edit_and_new_confirmation(self):
        await self.collect()
        second = await self.collect()
        self.played()
        self.coordinator.user_turn("yes")
        self.assertEqual((await self.confirm()).reason, "quantity_ambiguous")
        self.executor.assert_not_awaited()
        plan = self.coordinator.plan
        self.coordinator.user_turn("only-once")
        self.assertTrue(self.coordinator.remove(plan.id, 2).ok)
        self.assertEqual(self.store.get(second.id).status, "cancelled")
        self.assertFalse(plan.has_duplicates)
        self.played()
        self.coordinator.user_turn("confirm-once")
        self.assertTrue((await self.confirm()).ok)
        self.executor.assert_awaited_once()

    async def test_concurrent_confirmation_claims_execution_once(self):
        await self.collect()
        self.played()
        self.coordinator.user_turn("yes")
        entered, release = asyncio.Event(), asyncio.Event()

        async def execute(*args):
            entered.set()
            await release.wait()
            return ExecutorResult(True, "已添加")

        self.executor.side_effect = execute
        task = asyncio.create_task(self.confirm())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertEqual(self.coordinator.plan.status, "executing")
            self.assertEqual((await self.confirm()).reason, "turn_used")
            plan = self.coordinator.plan
            self.coordinator.user_turn("another-turn")
            self.assertEqual(self.coordinator.remove(plan.id, 1).reason, "invalid_state")
            self.assertEqual(self.coordinator.cancel_plan(plan.id).reason, "invalid_state")
            self.assertEqual((await self.coordinator.update(plan.id, 1, {})).reason,
                             "invalid_state")
            release.set()
            self.assertTrue((await task).ok)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.executor.assert_awaited_once()

    async def test_approve_false_skips_external_cancelled_item(self):
        cancelled = await self.collect()
        await self.collect("鸡蛋")
        self.gateway.cancel(cancelled.id)
        self.played()
        self.coordinator.user_turn("yes")
        result = await self.confirm()
        self.assertEqual([op.status for op in result.operations], ["cancelled", "succeeded"])
        self.executor.assert_awaited_once()

    async def test_control_edits_need_a_current_unused_user_turn_even_when_collecting(self):
        await self.collect()
        plan = self.coordinator.plan
        self.assertEqual(self.coordinator.remove(plan.id, 1).reason, "no_user_turn")
        self.assertEqual(self.coordinator.cancel_plan(plan.id).reason, "no_user_turn")
        self.assertEqual((await self.coordinator.update(plan.id, 1, {})).reason,
                         "no_user_turn")
        self.coordinator.user_turn("remove")
        self.assertTrue(self.coordinator.remove(plan.id, 1).ok)
        self.assertTrue(self.coordinator.cancel_plan(plan.id).ok)

    async def test_cancelling_plan_cancels_all_operations_without_executing(self):
        await self.collect()
        await self.collect("鸡蛋")
        self.coordinator.user_turn("cancel")
        plan = self.coordinator.plan
        result = self.coordinator.cancel_plan(plan.id)
        self.assertTrue(result.ok)
        self.assertEqual(plan.status, "cancelled")
        self.assertEqual([op.status for op in result.operations], ["cancelled", "cancelled"])
        self.executor.assert_not_awaited()

    async def test_answered_or_replayed_turns_cannot_authorize_control_edits(self):
        await self.collect()
        plan = self.coordinator.plan
        self.coordinator.user_turn("question")
        self.coordinator.turn_answered("question")
        self.assertEqual(self.coordinator.user_turn("question").reason, "duplicate_turn")
        self.assertEqual(self.coordinator.cancel_plan(plan.id).reason, "no_user_turn")
        self.coordinator.user_turn("fresh")
        self.coordinator.turn_answered("question")  # 迟到的旧回答不作废新一句话。
        self.assertTrue(self.coordinator.cancel_plan(plan.id).ok)
        self.assertEqual(self.coordinator.user_turn("fresh").reason, "duplicate_turn")

    async def test_reconnect_preserves_contents_and_clears_confirmation_eligibility(self):
        await self.collect()
        self.played()
        self.coordinator.user_turn("before-reconnect")
        before = self.coordinator.snapshot()
        self.coordinator.on_reconnect()
        snapshot = self.coordinator.snapshot()
        self.assertEqual(snapshot["id"], before["id"])
        self.assertEqual(snapshot["version"], before["version"])
        self.assertEqual(snapshot["items"], before["items"])
        self.assertEqual(snapshot["status"], "collecting")
        self.assertEqual((await self.confirm()).reason, "not_played")
        self.played()
        self.assertEqual(self.coordinator.user_turn("before-reconnect").reason, "duplicate_turn")
        self.assertEqual((await self.confirm()).reason, "no_user_turn")
        self.coordinator.user_turn("after-reconnect")
        self.assertTrue((await self.confirm()).ok)

    async def test_snapshot_is_readonly_and_reports_live_operation_status(self):
        self.assertIsNone(self.coordinator.snapshot())
        op = await self.collect()
        self.gateway.cancel(op.id)
        snapshot = self.coordinator.snapshot()
        self.assertEqual(snapshot["items"][0]["status"], "cancelled")
        snapshot["items"][0]["arguments"]["items"].append("鸡蛋")
        self.assertEqual(self.coordinator.plan.items[0].arguments["items"], ["牛奶"])

    async def test_duplicate_playback_callback_does_not_extend_deadline(self):
        await self.collect()
        self.played()
        self.coordinator.user_turn("yes")
        self.clock.now += 100
        self.assertEqual(self.played().reason, "already_played")
        self.clock.now += 20
        self.assertEqual((await self.confirm()).reason, "expired")

    async def test_turn_before_recap_is_not_eligible_even_at_same_clock_value(self):
        await self.collect()
        self.coordinator.user_turn("too-early")
        self.played()
        self.assertEqual((await self.confirm()).reason, "no_user_turn")
        self.coordinator.user_turn("after-playback")
        self.assertTrue((await self.confirm()).ok)

    async def test_update_cleanup_when_old_operation_cannot_be_cancelled(self):
        old = await self.collect()
        self.coordinator.user_turn("change")
        plan = self.coordinator.plan
        cancel = self.gateway.cancel

        def cannot_cancel_old(op_id):
            return False if op_id == old.id else cancel(op_id)

        with patch.object(self.gateway, "cancel", side_effect=cannot_cancel_old):
            result = await self.coordinator.update(plan.id, 1, {'list_name': '购物清单', 'items': ['鸡蛋']})
        self.assertEqual(result.reason, "operation_unavailable")
        self.assertEqual(plan.items[0].op_id, old.id)
        self.assertEqual(plan.version, 1)
        rows = self.store._db.execute("SELECT status FROM operations ORDER BY rowid").fetchall()
        self.assertEqual([row["status"] for row in rows], ["pending", "cancelled"])

    async def test_input_changes_while_update_waits_discards_new_operation(self):
        old = await self.collect()
        self.coordinator.user_turn("change")
        plan = self.coordinator.plan
        entered, release = asyncio.Event(), asyncio.Event()
        submit = self.gateway.submit

        async def delayed_submit(*args):
            entered.set()
            await release.wait()
            return await submit(*args)

        with patch.object(self.gateway, "submit", side_effect=delayed_submit):
            task = asyncio.create_task(self.coordinator.update(plan.id, 1, {'list_name': '购物清单', 'items': ['鸡蛋']}))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                self.assertEqual(self.coordinator.recap_played(plan.id, plan.version, False).reason, "busy")
                self.assertEqual(self.coordinator.collect(old).reason, "busy")
                self.coordinator.user_turn("newer-user-input")
                release.set()
                self.assertEqual((await task).reason, "turn_changed")
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(plan.items[0].op_id, old.id)
        rows = self.store._db.execute("SELECT status FROM operations ORDER BY rowid").fetchall()
        self.assertEqual([row["status"] for row in rows], ["pending", "cancelled"])

    async def test_registration_failure_preserves_old_operation_and_turn(self):
        old = await self.collect()
        self.coordinator.user_turn("change")
        plan = self.coordinator.plan
        before = self.coordinator.snapshot()
        with patch.object(self.gateway, "submit", side_effect=RuntimeError("登记失败")):
            result = await self.coordinator.update(plan.id, 1, {'list_name': '购物清单', 'items': ['鸡蛋']})
        self.assertEqual(result.reason, "registration_failed")
        self.assertEqual(self.coordinator.snapshot(), before)
        self.assertEqual(self.store.get(old.id).status, "pending")
        self.assertTrue(self.coordinator.cancel_plan(plan.id).ok)

    async def test_reconnect_during_execution_keeps_running_and_blocks_new_confirmation(self):
        await self.collect()
        self.played()
        self.coordinator.user_turn("yes")
        entered, release = asyncio.Event(), asyncio.Event()

        async def execute(*args):
            entered.set()
            await release.wait()
            return ExecutorResult(True, "已添加")

        self.executor.side_effect = execute
        task = asyncio.create_task(self.confirm())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.coordinator.on_reconnect()
            self.assertEqual(self.coordinator.snapshot()["status"], "executing")
            self.assertEqual(self.coordinator.snapshot()["items"][0]["status"], "executing")
            self.coordinator.user_turn("after-reconnect")
            self.assertEqual((await self.confirm()).reason, "invalid_state")
            release.set()
            self.assertEqual((await task).operations[0].status, "succeeded")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.executor.assert_awaited_once()

    async def test_cancelled_confirmation_records_unknown_and_does_not_execute_remaining_items(self):
        first, second = await self.collect(), await self.collect("鸡蛋")
        self.played()
        self.coordinator.user_turn("yes")
        entered = asyncio.Event()

        async def execute(*args):
            entered.set()
            await asyncio.Future()

        self.executor.side_effect = execute
        task = asyncio.create_task(self.confirm())
        try:
            await asyncio.wait_for(entered.wait(), 1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.store.get(first.id).status, "unknown")
        self.assertEqual(self.store.get(second.id).status, "cancelled")
        self.assertEqual(self.coordinator.plan.status, "done")
        self.assertEqual((await self.confirm()).reason, "turn_used")
        self.executor.assert_awaited_once()

    async def test_invalid_ids_versions_indices_and_unregistered_operations(self):
        self.assertEqual((await self.coordinator.confirm("missing", 1)).reason, "no_plan")
        op = await self.collect()
        plan = self.coordinator.plan
        self.assertEqual(self.coordinator.collect(replace(op, id="not-registered")).reason,
                         "invalid_operation")
        self.assertEqual(self.coordinator.collect(op).reason, "duplicate_operation")
        self.coordinator.user_turn("edit")
        self.assertEqual(self.coordinator.cancel_plan('other').reason, "plan_mismatch")
        for index in (-1, 99, True):
            self.assertEqual(self.coordinator.remove(plan.id, index).reason, "operation_unavailable")
            self.assertEqual((await self.coordinator.update(plan.id, index, {})).reason,
                             "operation_unavailable")


if __name__ == "__main__":
    unittest.main()
