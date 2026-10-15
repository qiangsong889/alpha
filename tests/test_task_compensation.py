"""验证撤销只使用创建回执中的精确 ID。"""

from copy import deepcopy
from dataclasses import replace
import unittest

from core.task_compensation import created_tasks, undo_add_items
from integrations.operations import OperationStore


class TaskCompensationTests(unittest.TestCase):
    def setUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.op = replace(self.store.create("turn", "add_items", {
            "list_name": "购物清单", "items": ["牛奶", "牛奶", "鸡蛋"],
        }), status="succeeded", result={"text": "已添加", "structured_content": {
            "version": 1, "status": "succeeded", "list_id": "list-1", "list_name": "购物清单",
            "created": [{"task_id": "milk-1", "title": "牛奶"},
                        {"task_id": "milk-2", "title": "牛奶"},
                        {"task_id": "egg-1", "title": "鸡蛋"}],
        }})

    def test_equal_titles_remain_distinct_ids_and_original_unchanged(self):
        requests = undo_add_items(self.op)
        self.assertEqual([item.task_id for item in requests], ["milk-1", "milk-2", "egg-1"])
        self.assertTrue(all(item.source_operation_id == self.op.id for item in requests))
        self.assertTrue(all(item.name == "delete_task" for item in requests))
        self.assertEqual(requests[0].arguments, {"list_id": "list-1", "task_id": "milk-1"})
        self.assertEqual(self.op.status, "succeeded")

    def test_select_part_of_addition_and_cannot_delete_preexisting_milk(self):
        self.assertEqual([item.task_id for item in undo_add_items(self.op, task_ids=["milk-2"])],
                         ["milk-2"])
        with self.assertRaises(ValueError):
            undo_add_items(self.op, task_ids=["old-milk"])

    def test_unknown_failed_and_legacy_receipts_require_verification(self):
        for op in (replace(self.op, status="unknown"), replace(self.op, status="failed"),
                   replace(self.op, result="已添加：牛奶"), replace(self.op, name="create_reminder")):
            with self.assertRaises(ValueError):
                undo_add_items(op)

    def test_bad_receipts_are_rejected(self):
        for field, value in (("version", True), ("version", 2), ("status", "unknown"),
                             ("list_id", ""), ("list_name", "别的清单"), ("created", []),
                             ("created", [{"title": "牛奶"}] * 3),
                             ("created", [{"title": "牛奶", "task_id": "same"}] * 3)):
            result = deepcopy(self.op.result)
            result["structured_content"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                created_tasks(replace(self.op, result=result))

    def test_bad_selections_rejected_and_arguments_are_independent(self):
        for ids in ([], ["milk-1", "milk-1"], [None], "milk-1"):
            with self.assertRaises(ValueError):
                undo_add_items(self.op, task_ids=ids)
        request = undo_add_items(self.op)[0]
        request.arguments.clear()
        self.assertEqual(request.arguments["task_id"], "milk-1")
