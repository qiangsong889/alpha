"""计划逻辑测试：不启动语音、工具服务或任何外部请求。"""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from core.plan import Plan, action_count, needs_plan
from integrations.operations import OperationStore
from integrations.tool_policy import TOOL_POLICY, ToolPolicy


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.now = datetime(2026, 10, 15, 9, tzinfo=ZoneInfo("America/Los_Angeles"))
        self.plan = Plan()

    def add_items(self, items=None, list_name="购物清单"):
        return self.store.create("turn", "add_items", {
            "list_name": list_name, "items": items if items is not None else ["牛奶"],
        })

    def reminder(self, local_time="2026-10-16T08:00:00", **extra):
        return self.store.create("turn", "create_reminder", {
            "title": "开会", "local_time": local_time, **extra,
        })

    def test_action_counts_and_plan_threshold(self):
        three = self.add_items(["牛奶", "鸡蛋", "面包"])
        self.assertEqual(action_count(three), 3)
        self.assertTrue(needs_plan([three]))
        self.assertFalse(needs_plan([self.reminder()]))
        self.assertFalse(needs_plan([self.add_items()]))
        self.assertFalse(needs_plan([]))
        self.assertTrue(needs_plan([self.add_items(), self.reminder()]))

    def test_complete_items_uses_actual_parameter_and_keeps_repetitions(self):
        op = self.store.create("turn", "complete_items", {
            "list_name": "购物清单", "item_titles": ["牛奶", "牛奶"],
        })
        self.assertEqual(action_count(op), 2)
        self.assertTrue(needs_plan([op]))
        self.plan.add(op)
        self.assertIn("把购物清单里的牛奶、牛奶标为已完成", self.plan.recap(self.now))

    def test_confirm_policy_needs_plan_even_for_one_action(self):
        with patch.dict(TOOL_POLICY, {"send_mail": ToolPolicy(True, False, "confirm")}):
            op = self.store.create("turn", "send_mail", {"to": "test@example.com"})
            self.assertTrue(needs_plan([op]))
            self.plan.add(op)
            self.assertIn('send_mail {"to": "test@example.com"}', self.plan.recap(self.now))

    def test_unknown_read_only_immediate_and_nonpending_operations_rejected(self):
        for name in ("unknown", "list_items", "intent__HassTurnOn"):
            with self.subTest(name=name):
                op = self.store.create("turn", name, {})
                with self.assertRaises(ValueError):
                    needs_plan([self.add_items(["牛奶", "鸡蛋"]), op])
                with self.assertRaises(ValueError):
                    self.plan.add(op)
        op = self.add_items()
        self.store.transition(op.id, "pending", "approved")
        with self.assertRaises(ValueError):
            needs_plan([self.store.get(op.id)])

    def test_invalid_item_arrays_do_not_silently_count_as_zero(self):
        for items in ([], "牛奶", [""], [123]):
            with self.subTest(items=items):
                op = self.add_items(items)
                with self.assertRaises(ValueError):
                    action_count(op)
                with self.assertRaises(ValueError):
                    self.plan.add(op)
                self.assertEqual(self.plan.version, 0)

    def test_each_successful_edit_increments_version(self):
        original, replacement = self.add_items(), self.reminder()
        self.assertTrue(self.plan.add(original))
        self.assertEqual(self.plan.version, 1)
        self.assertTrue(self.plan.replace(0, replacement))
        self.assertEqual(self.plan.items[0].op_id, replacement.id)
        self.assertEqual(self.plan.version, 2)
        # 同项成功替换也算一次修改；Plan 不负责取消旧数据库记录。
        self.assertTrue(self.plan.replace(0, replacement))
        self.assertEqual(self.plan.version, 3)
        self.assertEqual(self.store.get(original.id).status, "pending")
        self.assertTrue(self.plan.remove(0))
        self.assertEqual(self.plan.version, 4)
        self.assertEqual(self.plan.items, ())

    def test_invalid_arguments_are_rejected_before_add_or_replace_changes_plan(self):
        invalid_calls = [
            ("add_items", {"items": ["牛奶"]}),
            ("add_items", {"items": ["牛奶"], "list_name": None}),
            ("complete_items", {"item_titles": ["牛奶"]}),
            ("complete_items", {"item_titles": ["牛奶"], "list_name": "  "}),
        ]
        valid_reminder = {"title": "开会", "local_time": "2026-10-16T08:00:00"}
        for field, values in (
            ("minutes_before", (-1, 40321, True, 1.5, "10", None)),
            ("local_time", (None, 123, "", "not-a-date", "2026-02-30T08:00:00",
                            "2026-10-16", "2026-10-16T08:00",
                            "2026-10-16T08:00:00-07:00", "2026-10-16T08:00:00Z")),
            ("title", (None, 123, "", "  ")),
        ):
            invalid_calls.extend(("create_reminder", {**valid_reminder, field: value})
                                 for value in values)
        for missing in ("title", "local_time"):
            invalid_calls.append(("create_reminder", {
                key: value for key, value in valid_reminder.items() if key != missing
            }))
        for name, args in invalid_calls:
            with self.subTest(name=name, args=args):
                bad = self.store.create("turn", name, args)
                empty = Plan()
                with self.assertRaises(ValueError):
                    empty.add(bad)
                self.assertEqual((empty.items, empty.version, empty.status), ((), 0, "collecting"))
                plan = Plan()
                plan.add(self.add_items())
                plan.status = "awaiting_confirmation"
                before = (plan.items, plan.version, plan.status, plan.recap(self.now))
                with self.assertRaises(ValueError):
                    plan.add(bad)
                with self.assertRaises(ValueError):
                    plan.replace(0, bad)
                self.assertEqual((plan.items, plan.version, plan.status, plan.recap(self.now)), before)

    def test_valid_reminder_limits_and_fractional_seconds_can_be_recapped(self):
        for minutes in (0, 40320):
            plan = Plan()
            self.assertTrue(plan.add(self.reminder("2026-10-16T08:00:01.123456", minutes_before=minutes)))
            self.assertIn("明天早上 8:00:01.123456", plan.recap(self.now))

    def test_locked_states_reject_all_edits_without_changing_version(self):
        self.plan.add(self.add_items())
        for status in ("executing", "done", "cancelled"):
            with self.subTest(status=status):
                self.plan.status = status
                self.assertFalse(self.plan.add(self.reminder()))
                self.assertFalse(self.plan.replace(0, self.reminder()))
                self.assertFalse(self.plan.remove(0))
                self.assertEqual(self.plan.version, 1)
                self.assertEqual(len(self.plan.items), 1)

    def test_each_edit_invalidates_awaiting_confirmation(self):
        self.plan.add(self.add_items())
        for edit in (lambda: self.plan.add(self.reminder()),
                     lambda: self.plan.replace(0, self.add_items(["鸡蛋"])),
                     lambda: self.plan.remove(0)):
            self.plan.status = "awaiting_confirmation"
            old_version = self.plan.version
            self.assertTrue(edit())
            self.assertEqual(self.plan.status, "collecting")
            self.assertEqual(self.plan.version, old_version + 1)

    def test_invalid_indices_do_not_change_plan(self):
        self.plan.add(self.add_items())
        self.plan.status = "awaiting_confirmation"
        for index in (-1, 1, 100, True):
            self.assertFalse(self.plan.replace(index, self.reminder()))
            self.assertFalse(self.plan.remove(index))
        self.assertEqual(self.plan.version, 1)
        self.assertEqual(self.plan.status, "awaiting_confirmation")

    def test_parameter_snapshots_cannot_be_changed_without_versioning(self):
        original = self.add_items()
        original.arguments["details"] = {"nested": ["original"]}
        self.plan.add(original)
        original.arguments["items"].append("鸡蛋")
        original.arguments["details"]["nested"].append("changed")
        exposed = self.plan.items
        exposed[0].arguments["items"].append("面包")
        exposed[0].arguments["details"]["nested"].clear()
        self.assertEqual(self.plan.items[0].arguments["items"], ["牛奶"])
        self.assertEqual(self.plan.items[0].arguments["details"], {"nested": ["original"]})
        self.assertEqual(self.plan.version, 1)

    def test_same_operation_id_cannot_be_added_twice_or_replace_another_item(self):
        first, second = self.add_items(), self.add_items(["鸡蛋"])
        self.plan.add(first)
        self.assertFalse(self.plan.add(first))
        self.plan.add(second)
        self.assertFalse(self.plan.replace(0, second))
        self.assertEqual(self.plan.version, 2)

    def test_reminder_recap_relative_date_and_advance_minutes(self):
        self.plan.add(self.reminder(minutes_before=10))
        text = self.plan.recap(self.now)
        self.assertIn("明天早上 8:00 提醒你开会", text)
        self.assertIn("提前 10 分钟提醒", text)

    def test_today_after_tomorrow_and_distant_dates(self):
        for target, expected in (("2026-10-15T12:00:00", "今天中午 12:00"),
                                 ("2026-10-17T15:30:00", "后天下午 3:30"),
                                 ("2026-11-03T08:00:00", "2026年11月03日早上 8:00")):
            with self.subTest(target=target):
                plan = Plan()
                plan.add(self.reminder(target))
                self.assertIn(expected, plan.recap(self.now))

    def test_now_is_converted_to_configured_timezone_before_comparing_dates(self):
        # UTC 已经是 10 月 16 日，但 LA 还是 15 日晚上。
        utc_now = datetime(2026, 10, 16, 1, tzinfo=timezone.utc)
        self.plan.add(self.reminder())
        self.assertIn("明天早上 8:00", self.plan.recap(utc_now))
        shanghai = Plan(timezone_name="Asia/Shanghai")
        shanghai.add(self.reminder())
        self.assertIn("今天早上 8:00", shanghai.recap(utc_now))
        with self.assertRaises(ValueError):
            self.plan.recap(datetime(2026, 10, 15, 9))

    def test_recap_does_not_mutate_status_version_or_items(self):
        self.plan.add(self.reminder())
        self.plan.status = "awaiting_confirmation"
        before = (self.plan.version, self.plan.status, self.plan.items)
        self.assertEqual(self.plan.recap(self.now), self.plan.recap(self.now))
        self.assertEqual((self.plan.version, self.plan.status, self.plan.items), before)

    def test_duplicates_keep_both_items_and_ask_quantity_once(self):
        first = self.add_items()
        second = self.store.create("turn", "add_items", {"items": ["牛奶"], "list_name": "购物清单"})
        self.plan.add(first)
        self.plan.add(second)
        text = self.plan.recap(self.now)
        self.assertEqual(len(self.plan.items), 2)
        self.assertTrue(self.plan.has_duplicates)
        self.assertEqual(text.count("说了 2 次"), 1)
        self.assertIn("要做 2 次还是 1 次", text)
        self.plan.remove(1)
        self.assertFalse(self.plan.has_duplicates)

    def test_duplicate_comparison_preserves_array_order_repetitions_and_strings(self):
        for items, name in ((["鸡蛋", "牛奶"], "购物清单"),
                            (["牛奶", "鸡蛋", "牛奶"], "购物清单"),
                            (["牛奶", "鸡蛋"], " 购物清单")):
            plan = Plan()
            plan.add(self.add_items(["牛奶", "鸡蛋"]))
            plan.add(self.add_items(items, name))
            self.assertFalse(plan.has_duplicates)

    def test_recap_orders_two_different_operations_and_handles_empty_plan(self):
        self.assertEqual(self.plan.recap(self.now), "当前没有待执行的计划。")
        self.plan.add(self.add_items(["牛奶", "鸡蛋"]))
        self.plan.add(self.reminder())
        text = self.plan.recap(self.now)
        self.assertTrue(text.startswith("我确认一下：① 把牛奶、鸡蛋加到购物清单"))
        self.assertIn("② 明天早上 8:00 提醒你开会", text)
        self.assertTrue(text.endswith("。对吗？"))
        self.assertNotEqual(self.plan.id, Plan().id)


if __name__ == "__main__":
    unittest.main()
