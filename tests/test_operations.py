"""策略与操作记录测试；只使用内存数据库，不访问任何外部服务。"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from integrations.operations import OperationStore
from integrations.tool_policy import get_policy


class OperationTests(unittest.TestCase):
    def test_default_arguments_match_old_and_new_records_in_both_directions(self):
        base = {'title': '开会', 'local_time': '2026-10-17T08:00:00'}
        schema = {'create_reminder': {'type': 'object', 'properties': {
            'minutes_before': {'type': 'integer', 'default': 0}}}}
        # 记录在注册schema之前写入，模拟已有数据库。
        old = self.create(base)
        self.succeed(old)
        self.store.set_tool_schemas(schema)
        self.assertEqual(self.store.recent_succeeded(old.name, {**base, 'minutes_before': 0}).id, old.id)
        self.assertEqual(self.store.get(old.id).arguments, base)
        self.assertIsNone(self.store.recent_succeeded(old.name, {**base, 'minutes_before': 5}))
        explicit = self.create({**base, 'minutes_before': 0})
        self.succeed(explicit)
        self.assertEqual(self.store.recent_succeeded(old.name, base).id, explicit.id)
        self.assertEqual(self.store.get(explicit.id).arguments['minutes_before'], 0)

    def test_unknown_default_arguments_also_match(self):
        self.store.set_tool_schemas({'create_reminder': {'properties': {'minutes_before': {'default': 0}}}})
        op = self.create({'title': '开会', 'minutes_before': 0})
        self.store.transition(op.id, 'pending', 'approved')
        self.store.transition(op.id, 'approved', 'executing')
        self.store.transition(op.id, 'executing', 'unknown')
        self.assertEqual(self.store.recent_unknown(op.name, {'title': '开会'}).id, op.id)

    def test_comparison_does_not_drop_required_unknown_keys_or_confuse_bool_and_zero(self):
        from integrations.tool_arguments import comparison_arguments
        schema = {'required': ['required'], 'properties': {
            'required': {'default': 0}, 'optional': {'default': 0},
            'items': {'items': {'properties': {'flag': {'default': False}}}}}}
        args = {'required': 0, 'optional': False, 'unknown': 0,
                'items': [{'name': '牛奶', 'flag': False}, {'name': '鸡蛋'}, {'name': '牛奶'}]}
        result = comparison_arguments(args, schema)
        self.assertEqual(result, {'required': 0, 'optional': False, 'unknown': 0,
                                 'items': [{'name': '牛奶'}, {'name': '鸡蛋'}, {'name': '牛奶'}]})
        self.assertFalse(args['items'][0]['flag'])

    def test_process_lock_prevents_competing_recovery_and_releases_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operations.db"
            first, second = OperationStore(path), OperationStore(path)
            try:
                first.acquire_process_lock()
                op = first.create("turn", "create_reminder", {"title": "开会"})
                first.transition(op.id, "pending", "approved")
                first.transition(op.id, "approved", "executing")
                with self.assertRaisesRegex(RuntimeError, "另一个 Alpha/Qwen"):
                    second.acquire_process_lock()
                self.assertEqual(second.get(op.id).status, "executing")
                first.close()
                second.acquire_process_lock()
                self.assertEqual(second.recover_on_startup()[0].status, "unknown")
            finally:
                first.close()
                second.close()

    def test_unknown_lookup_preserves_arguments_and_expires_without_changing_record(self):
        args = {"items": ["牛奶", "鸡蛋"], "list_name": "购物清单"}
        op = self.store.create("turn", "add_items", args)
        self.store.transition(op.id, "pending", "approved")
        self.store.transition(op.id, "approved", "executing")
        self.store.transition(op.id, "executing", "unknown")
        with patch("integrations.operations.time.time", return_value=op.created_at + 60):
            self.assertEqual(self.store.recent_unknown("add_items", dict(reversed(list(args.items())))).id, op.id)
            self.assertIsNone(self.store.recent_unknown("add_items", {**args, "items": ["鸡蛋", "牛奶"]}))
        with patch("integrations.operations.time.time", return_value=op.created_at + 601):
            self.assertIsNone(self.store.recent_unknown("add_items", args))
        self.assertEqual(self.store.get(op.id).status, "unknown")
        with self.assertRaises(ValueError):
            self.store.recent_unknown("add_items", args, within_s=-1)

    def setUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)

    def create(self, arguments=None):
        return self.store.create("turn-1", "create_reminder", arguments if arguments is not None else {
            "title": "开会", "local_time": "2026-10-15T08:00:00",
        })

    def succeed(self, operation, result="已创建"):
        self.assertTrue(self.store.transition(operation.id, "pending", "approved"))
        self.assertTrue(self.store.transition(operation.id, "approved", "executing"))
        self.assertTrue(self.store.transition(operation.id, "executing", "succeeded", result))

    def test_create_is_pending_and_copies_arguments(self):
        arguments = {"items": ["牛奶"]}
        op = self.create(arguments)
        self.assertEqual(op.status, "pending")
        self.assertEqual(op.turn_id, "turn-1")
        self.assertIsNone(op.result)
        self.assertEqual(op.created_at, op.updated_at)
        arguments["items"].append("鸡蛋")
        self.assertEqual(op.arguments, {"items": ["牛奶"]})
        self.assertEqual(self.store.get(op.id), op)

    def test_approved_to_executing_can_only_be_claimed_once(self):
        op = self.create()
        self.assertTrue(self.store.transition(op.id, "pending", "approved"))
        self.assertTrue(self.store.transition(op.id, "approved", "executing"))
        self.assertFalse(self.store.transition(op.id, "approved", "executing"))
        self.assertEqual(self.store.get(op.id).status, "executing")

    def test_pending_to_executing_is_rejected(self):
        op = self.create()
        self.assertFalse(self.store.transition(op.id, "pending", "executing"))
        self.assertEqual(self.store.get(op.id).status, "pending")

    def test_recent_success_preserves_list_order_and_duplicates(self):
        op = self.create({"items": ["牛奶", "鸡蛋", "牛奶"]})
        self.succeed(op, {"event_id": "event-1"})
        found = self.store.recent_succeeded(op.name, {"items": ["牛奶", "鸡蛋", "牛奶"]})
        self.assertEqual(found.id, op.id)
        self.assertEqual(found.result, {"event_id": "event-1"})
        self.assertIsNone(self.store.recent_succeeded(op.name, {"items": ["牛奶", "牛奶", "鸡蛋"]}))
        self.assertIsNone(self.store.recent_succeeded(op.name, {"items": ["牛奶", "鸡蛋"]}))
        self.assertIsNone(self.store.recent_succeeded("add_items", op.arguments))

    def test_recent_success_ignores_object_key_order_including_nested_objects(self):
        op = self.create({"title": "开会", "details": {"time": "08:00", "items": ["牛奶", "鸡蛋", "牛奶"]}})
        self.succeed(op)
        found = self.store.recent_succeeded(op.name, {
            "details": {"items": ["牛奶", "鸡蛋", "牛奶"], "time": "08:00"}, "title": "开会",
        })
        self.assertEqual(found.id, op.id)
        self.assertIsNone(self.store.recent_succeeded(op.name, {
            "title": " 开会 ", "details": {"time": "08:00", "items": ["牛奶", "鸡蛋", "牛奶"]},
        }))

    def test_transition_without_result_preserves_existing_result(self):
        op = self.create()
        self.store.transition(op.id, "pending", "approved")
        self.store.transition(op.id, "approved", "executing", "等待响应")
        self.assertTrue(self.store.transition(op.id, "executing", "succeeded"))
        self.assertEqual(self.store.get(op.id).result, "等待响应")

    def test_transition_explicit_result_replaces_existing_even_when_falsy(self):
        for result in ({"event_id": "event-1"}, "", False, 0, {}):
            with self.subTest(result=result):
                op = self.create()
                self.store.transition(op.id, "pending", "approved")
                self.store.transition(op.id, "approved", "executing", "等待响应")
                self.assertTrue(self.store.transition(op.id, "executing", "succeeded", result))
                self.assertEqual(self.store.get(op.id).result, result)

    def test_complete_items_is_deferred_and_not_retry_safe(self):
        policy = get_policy("complete_items")
        self.assertTrue(policy.mutates)
        self.assertFalse(policy.retry_safe)
        self.assertEqual(policy.commit, "deferred")

    def test_unknown_policy_is_not_allowed(self):
        self.assertIsNone(get_policy("不存在的工具"))

    def test_recover_executing_to_unknown_without_retrying(self):
        op = self.create()
        pending = self.create()
        self.assertTrue(self.store.transition(op.id, "pending", "approved"))
        self.assertTrue(self.store.transition(op.id, "approved", "executing", "等待响应"))
        recovered = self.store.recover_on_startup()
        self.assertEqual([record.id for record in recovered], [op.id])
        self.assertEqual(recovered[0].status, "unknown")
        self.assertEqual(recovered[0].result, "等待响应")
        self.assertEqual(self.store.get(op.id), recovered[0])
        self.assertEqual(self.store.get(pending.id).status, "pending")
        self.assertEqual(self.store.recover_on_startup(), [])
        self.assertIsNone(self.store.recent_succeeded(op.name, op.arguments))
        self.assertFalse(self.store.transition(op.id, "unknown", "executing"))

    def test_terminal_and_missing_operations_cannot_transition(self):
        for status in ("succeeded", "failed", "unknown"):
            with self.subTest(status=status):
                op = self.create()
                self.store.transition(op.id, "pending", "approved")
                self.store.transition(op.id, "approved", "executing")
                self.assertTrue(self.store.transition(op.id, "executing", status))
                self.assertFalse(self.store.transition(op.id, status, "pending"))
        op = self.create()
        self.assertTrue(self.store.transition(op.id, "pending", "cancelled"))
        self.assertFalse(self.store.transition(op.id, "cancelled", "approved"))
        self.assertFalse(self.store.transition("missing", "pending", "approved"))
        self.assertIsNone(self.store.get("missing"))

    def test_recent_success_uses_completion_time_window(self):
        with patch("integrations.operations.time.time", return_value=1000):
            op = self.create()
            self.succeed(op)
        with patch("integrations.operations.time.time", return_value=1599):
            self.assertEqual(self.store.recent_succeeded(op.name, op.arguments).id, op.id)
        with patch("integrations.operations.time.time", return_value=1601):
            self.assertIsNone(self.store.recent_succeeded(op.name, op.arguments))

    def test_deletion_backup_survives_final_result_and_database_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operations.db"
            store = OperationStore(path)
            op = store.create("turn", "delete_task", {"list_id": "list", "task_id": "old"})
            backup = {"title": "牛奶", "resource": {"title": "牛奶", "notes": "两瓶"}}
            store.save_deletion_backup(op.id, backup)
            store.transition(op.id, "pending", "approved")
            store.transition(op.id, "approved", "executing")
            store.transition(op.id, "executing", "succeeded", "删除成功")
            store.close()
            reopened = OperationStore(path)
            try:
                self.assertEqual(reopened.deletion_backup(op.id), backup)
                self.assertEqual(reopened.get(op.id).result, "删除成功")
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
