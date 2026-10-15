"""新批次逻辑的离线测试；不执行工具，不接语音和外部服务。"""

import unittest
from unittest.mock import patch

from core.task_batch import TaskChain
from integrations.operations import OperationStore
from integrations.tool_policy import TOOL_POLICY, ToolPolicy


class TaskBatchTests(unittest.TestCase):
    def setUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.chain = TaskChain()
        self.batch = self.chain.new_batch()

    def op(self, items=None, *, turn="turn"):
        return self.store.create(turn, "add_items", {
            "list_name": "购物清单", "items": items or ["牛奶"],
        })

    def add(self, items=None, *, batch=None):
        return self.chain.add((batch or self.batch).id, self.op(items))

    def queue(self, batch=None, **kwargs):
        batch = batch or self.batch
        return self.chain.queue(batch.id, version=self.chain.batch(batch.id).version, **kwargs)

    def test_fixed_ids_and_replacement_keeps_cancelled_history(self):
        milk, egg = self.add(), self.add(["鸡蛋"])
        soy = self.chain.replace(milk.task_id, self.op(["酱油"]))
        self.assertEqual(self.chain.task(milk.task_id).status, "cancelled")
        self.assertEqual(soy.replaces_task_id, milk.task_id)
        self.assertEqual(egg.task_id, 2)
        self.assertEqual(soy.task_id, 3)
        self.assertEqual(self.chain.batch(self.batch.id).version, 3)
        self.assertEqual([task.arguments["items"] for task in self.chain.effective_tasks(self.batch.id)],
                         [["鸡蛋"], ["酱油"]])
        # 纯逻辑不会偷偷取消旧数据库记录：以后由接线层同步 gateway。
        self.assertEqual(self.store.get(milk.operation_id).status, "pending")

    def test_invalid_replacement_does_not_damage_old_task_or_version(self):
        milk = self.add()
        bad = self.store.create("turn", "add_items", {"items": ["酱油"]})
        with self.assertRaises(ValueError):
            self.chain.replace(milk.task_id, bad)
        self.assertEqual(self.chain.task(milk.task_id), milk)
        self.assertEqual(self.chain.batch(self.batch.id).version, 1)

    def test_input_and_output_snapshots_are_independent(self):
        op = self.op()
        task = self.chain.add(self.batch.id, op)
        op.arguments["items"].append("鸡蛋")
        task.arguments["items"].clear()
        self.chain.tasks(self.batch.id)[0].arguments.clear()
        self.assertEqual(self.chain.task(task.task_id).arguments["items"], ["牛奶"])

    def test_duplicate_operation_rejected_but_equal_new_requests_retained(self):
        op = self.op()
        self.chain.add(self.batch.id, op)
        with self.assertRaises(ValueError):
            self.chain.add(self.batch.id, op)
        self.add()
        self.assertEqual(len(self.chain.tasks(self.batch.id)), 2)

    def test_reads_unknown_and_nonpending_operations_rejected(self):
        for name in ("list_items", "not_allowed"):
            with self.assertRaises(ValueError):
                self.chain.add(self.batch.id, self.store.create("turn", name, {}))
        op = self.op()
        self.store.transition(op.id, "pending", "approved")
        with self.assertRaises(ValueError):
            self.chain.add(self.batch.id, self.store.get(op.id))
        self.assertEqual(self.chain.batch(self.batch.id).version, 0)

    def test_device_writes_are_collected_without_executing(self):
        op = self.store.create("turn", "intent__HassTurnOff", {"name": "客厅灯"})
        task = self.chain.add(self.batch.id, op)
        self.assertEqual(task.status, "pending")
        self.assertEqual(self.store.get(op.id).status, "pending")

    def test_confirmation_counts_final_actions_threshold_three(self):
        first = self.add(["牛奶", "鸡蛋"])
        self.assertFalse(self.chain.requires_confirmation(self.batch.id))
        extra = self.add(["酱油"])
        self.assertTrue(self.chain.requires_confirmation(self.batch.id))
        self.assertFalse(self.queue())
        self.chain.cancel(extra.task_id)
        self.assertFalse(self.chain.requires_confirmation(self.batch.id))
        self.assertTrue(self.queue())
        self.assertEqual(self.chain.task(first.task_id).status, "queued")

    def test_one_high_risk_action_requires_confirmation(self):
        with patch.dict(TOOL_POLICY, {"send_mail": ToolPolicy(True, False, "confirm")}):
            self.chain.add(self.batch.id, self.store.create("turn", "send_mail", {"to": "a"}))
            self.assertTrue(self.chain.requires_confirmation(self.batch.id))
            self.assertFalse(self.queue())
            self.assertTrue(self.queue(confirmed=True))

    def test_queue_checks_version_empty_and_clarification(self):
        self.assertFalse(self.queue())
        self.add()
        self.assertFalse(self.chain.queue(self.batch.id, version=0))
        self.chain.set_collection_status(self.batch.id, "waiting_for_clarification")
        self.assertFalse(self.queue(confirmed=True))
        self.chain.set_collection_status(self.batch.id, "collecting")
        self.assertTrue(self.queue())
        self.assertFalse(self.queue())

    def test_edits_invalidate_confirmation(self):
        milk = self.add()
        for edit in (lambda: self.add(["鸡蛋"]),
                     lambda: self.chain.replace(milk.task_id, self.op(["酱油"])),
                     lambda: self.chain.cancel(2)):
            self.chain.set_collection_status(self.batch.id, "awaiting_confirmation")
            before = self.chain.batch(self.batch.id).version
            edit()
            self.assertEqual(self.chain.batch(self.batch.id).version, before + 1)
            self.assertEqual(self.chain.batch(self.batch.id).status, "collecting")

    def test_only_one_collecting_batch_and_no_add_to_submitted_batch(self):
        with self.assertRaises(ValueError):
            self.chain.new_batch()
        self.add()
        self.queue()
        with self.assertRaises(ValueError):
            self.add()
        next_batch = self.chain.new_batch()
        self.add(["鸡蛋"], batch=next_batch)
        self.assertEqual(self.chain.batch(self.batch.id).status, "queued")

    def test_cancellation_scope_precedes_command_across_batches_not_future_tasks(self):
        old = self.add()
        self.queue()
        second = self.chain.new_batch()
        earlier = self.add(batch=second)
        boundary = self.chain.mark_input()
        future = self.add(batch=second)
        targets = self.chain.targets_before(boundary, name="add_items")
        self.assertEqual([task.task_id for task in targets], [old.task_id, earlier.task_id])
        for task in targets:
            self.assertTrue(self.chain.cancel(task.task_id))
        self.assertEqual(self.chain.task(future.task_id).status, "pending")
        self.assertEqual(self.chain.batch(self.batch.id).status, "cancelled")

    def test_separate_chains_do_not_include_previous_history(self):
        self.add()
        other = TaskChain()
        boundary = other.mark_input()
        self.assertEqual(other.targets_before(boundary), ())
        with self.assertRaises(ValueError):
            self.chain.targets_before(999)

    def test_cancelled_number_not_reused_and_repeat_cancel_no_version_change(self):
        old = self.add()
        self.assertTrue(self.chain.cancel(old.task_id))
        version = self.chain.batch(self.batch.id).version
        self.assertFalse(self.chain.cancel(old.task_id))
        self.assertEqual(self.chain.batch(self.batch.id).version, version)
        self.assertEqual(self.add().task_id, old.task_id + 1)

    def test_cancel_executing_requires_verification_success_requires_compensation(self):
        old = self.add()
        self.queue()
        self.assertTrue(self.chain.transition_task(old.task_id, "executing"))
        self.assertEqual(self.chain.cancellation_effect(old.task_id), "verify")
        self.assertFalse(self.chain.cancel(old.task_id))
        self.chain.transition_task(old.task_id, "succeeded")
        self.assertEqual(self.chain.cancellation_effect(old.task_id), "compensate")
        self.assertFalse(self.chain.cancel(old.task_id))
        self.assertEqual(self.chain.batch(self.batch.id).status, "done")
        next_batch = self.chain.new_batch()
        with patch.dict(TOOL_POLICY, {"delete_task": ToolPolicy(True, False, "deferred")}):
            undo = self.store.create("turn", "delete_task", {
                "list_id": "exact-list-id", "task_id": "exact-google-id",
            })
            task = self.chain.add(next_batch.id, undo, compensates_task_id=old.task_id)
        self.assertEqual(task.compensates_task_id, old.task_id)
        self.assertEqual(self.chain.task(old.task_id).status, "succeeded")

    def test_unknown_requires_verification_and_cannot_be_retried(self):
        task = self.add()
        self.queue()
        self.chain.transition_task(task.task_id, "executing")
        self.chain.transition_task(task.task_id, "unknown")
        self.assertEqual(self.chain.cancellation_effect(task.task_id), "verify")
        self.assertFalse(self.chain.transition_task(task.task_id, "queued"))

    def test_cannot_compensate_pending_task(self):
        task = self.add()
        with self.assertRaises(ValueError):
            self.chain.add(self.batch.id, self.op(["酱油"]), compensates_task_id=task.task_id)
        self.assertEqual(self.chain.batch(self.batch.id).version, 1)

    def test_late_dependency_insertion_and_cycles_rejected_atomically(self):
        light, curtain = self.add(["开灯"]), self.add(["关窗帘"])
        self.chain.set_dependencies(light.task_id, (curtain.task_id,))
        self.assertEqual(self.chain.task(light.task_id).depends_on, (curtain.task_id,))
        before = self.chain.batch(self.batch.id)
        for deps in ((light.task_id,), (curtain.task_id,), (999,), (True,)):
            with self.assertRaises(ValueError):
                self.chain.set_dependencies(curtain.task_id, deps)
            self.assertEqual(self.chain.batch(self.batch.id), before)
        self.queue()
        self.assertFalse(self.chain.transition_task(light.task_id, "executing"))
        self.assertTrue(self.chain.transition_task(curtain.task_id, "executing"))
        self.chain.transition_task(curtain.task_id, "succeeded")
        self.assertTrue(self.chain.transition_task(light.task_id, "executing"))

    def test_failed_dependency_blocks_only_dependent_task(self):
        curtain = self.add(["窗帘"])
        light = self.chain.add(self.batch.id, self.op(["灯"]), depends_on=(curtain.task_id,))
        unrelated = self.add(["酱油"])
        self.assertTrue(self.queue(confirmed=True))
        self.chain.transition_task(curtain.task_id, "executing")
        self.chain.transition_task(curtain.task_id, "failed")
        self.assertFalse(self.chain.transition_task(light.task_id, "executing"))
        self.chain.transition_task(light.task_id, "skipped")
        self.assertTrue(self.chain.transition_task(unrelated.task_id, "executing"))
        self.chain.transition_task(unrelated.task_id, "succeeded")
        self.assertEqual(self.chain.batch(self.batch.id).status, "done")

    def test_dependency_on_future_batch_and_replacing_dependency_target_rejected(self):
        old = self.add()
        dependent = self.chain.add(self.batch.id, self.op(["灯"]), depends_on=(old.task_id,))
        with self.assertRaises(ValueError):
            self.chain.replace(old.task_id, self.op(["酱油"]))
        self.queue()
        second = self.chain.new_batch()
        new = self.add(batch=second)
        with self.assertRaises(ValueError):
            self.chain.set_dependencies(old.task_id, (new.task_id,))
        self.assertEqual(self.chain.task(dependent.task_id).depends_on, (old.task_id,))


if __name__ == "__main__":
    unittest.main()
