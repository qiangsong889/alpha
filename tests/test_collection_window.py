"""虚拟时钟测试：不真的等待两/五秒，不启动麦克风或外部服务。"""

import asyncio
import unittest
from unittest.mock import patch

from core.task_batch import TaskChain
from integrations.operations import OperationStore
from integrations.tool_policy import TOOL_POLICY, ToolPolicy
from voice.collection_window import CollectionWindow


class ManualClock:
    def __init__(self):
        self.now = 0.0
        self.waiters = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((self.now + seconds, future))
        try:
            await future
        finally:
            self.waiters = [(deadline, waiter) for deadline, waiter in self.waiters
                            if waiter is not future]

    async def advance(self, seconds):
        # 先让 create_task 的计时任务登记，再向前移动虚拟时间。
        await self.drain()
        self.now += seconds
        for deadline, future in tuple(self.waiters):
            if deadline <= self.now and not future.done():
                future.set_result(None)
        await self.drain()

    async def drain(self):
        for _ in range(4):
            await asyncio.sleep(0)


class CollectionWindowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = ManualClock()
        self.chain = TaskChain()
        self.batch = self.chain.new_batch()
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.window = CollectionWindow(self.chain, self.batch.id,
                                       clock=self.clock, sleep=self.clock.sleep)
        self.addAsyncCleanup(self.window.close)

    def add(self, items=None):
        op = self.store.create("user", "add_items", {
            "list_name": "购物清单", "items": items or ["牛奶"],
        })
        return self.chain.add(self.batch.id, op)

    def interpret(self, turn="u1", **kwargs):
        return self.window.interpreted(
            turn, version=self.chain.batch(self.batch.id).version, **kwargs,
        )

    def finish_input(self, turn="u1", **kwargs):
        self.window.speech_started(turn)
        self.window.speech_stopped(turn)
        self.assertTrue(self.interpret(turn, **kwargs))

    def event(self, kind):
        event = self.window.events.get_nowait()
        self.assertEqual(event.kind, kind)
        self.assertTrue(self.window.is_current(event))
        return event

    async def test_complete_task_ready_after_two_seconds_exactly_once(self):
        task = self.add()
        self.finish_input(complete=True)
        await self.clock.advance(1.9)
        self.assertTrue(self.window.events.empty())
        await self.clock.advance(.1)
        self.event("ready")
        await self.clock.advance(10)
        self.assertTrue(self.window.events.empty())
        self.assertEqual(self.chain.task(task.task_id).status, "pending")
        self.assertEqual(self.store.get(task.operation_id).status, "pending")

    async def test_no_submission_until_interpretation_finishes(self):
        self.add()
        self.window.speech_started("u1")
        self.window.speech_stopped("u1")
        await self.clock.advance(3)
        self.assertTrue(self.window.events.empty())
        self.assertTrue(self.interpret(complete=True))
        await self.clock.drain()
        self.event("ready")  # 不从模型处理完再额外等两秒。

    async def test_interpretation_before_speech_stopped_still_waits_for_silence(self):
        self.add()
        self.window.speech_started("u1")
        self.interpret(complete=True)
        await self.clock.advance(10)
        self.assertTrue(self.window.events.empty())
        self.window.speech_stopped("u1")
        await self.clock.advance(2)
        self.event("ready")

    async def test_incomplete_input_waits_five_seconds_then_asks_once(self):
        self.add()
        self.finish_input(complete=False, question="还要加什么？")
        await self.clock.advance(2)
        self.assertTrue(self.window.events.empty())
        await self.clock.advance(3)
        self.assertEqual(self.event("clarification_required").question, "还要加什么？")
        self.assertEqual(self.chain.batch(self.batch.id).status, "waiting_for_clarification")
        await self.clock.advance(20)
        self.assertTrue(self.window.events.empty())
        self.assertFalse(self.chain.queue(self.batch.id, version=1))

    async def test_clarification_answer_resumes_normal_window(self):
        self.add()
        self.finish_input(complete=False)
        await self.clock.advance(5)
        old = self.event("clarification_required")
        self.finish_input("u2", complete=True)
        self.assertFalse(self.window.is_current(old))
        await self.clock.advance(2)
        self.event("ready")

    async def test_new_speech_immediately_pauses_until_new_input_interpreted(self):
        self.add()
        self.finish_input(complete=True)
        await self.clock.advance(1)
        self.window.speech_started("u2")
        await self.clock.advance(5)
        self.assertTrue(self.window.events.empty())
        self.window.speech_stopped("u2")
        self.add(["鸡蛋"])
        await self.clock.advance(5)
        self.assertTrue(self.window.events.empty())
        self.interpret("u2", complete=True)
        await self.clock.drain()
        self.assertEqual(self.event("ready").version, 2)

    async def test_noise_resumes_previous_decision_and_preserves_meaningful_turn(self):
        self.add()
        self.finish_input(complete=True)
        await self.clock.advance(1)
        self.window.speech_started("noise")
        await self.clock.advance(5)
        self.assertTrue(self.window.events.empty())
        self.assertTrue(self.window.speech_ignored("noise"))
        await self.clock.advance(1)
        self.assertFalse(self.window.speech_ignored("noise"))
        await self.clock.advance(1)
        self.assertEqual(self.event("ready").turn_id, "u1")

    async def test_noise_while_waiting_for_clarification_does_not_submit_or_repeat_question(self):
        self.add()
        self.finish_input(complete=False)
        await self.clock.advance(5)
        self.event("clarification_required")
        self.window.speech_started("noise")
        self.window.speech_ignored("noise")
        await self.clock.advance(10)
        self.assertTrue(self.window.events.empty())
        self.assertEqual(self.chain.batch(self.batch.id).status, "waiting_for_clarification")

    async def test_initial_noise_does_not_create_a_submission(self):
        self.window.speech_started("noise")
        self.window.speech_ignored("noise")
        await self.clock.advance(10)
        self.assertTrue(self.window.events.empty())

    async def test_noise_does_not_bypass_pending_interpretation_of_previous_input(self):
        self.add()
        self.finish_input(complete=True)
        self.window.speech_started("u2")
        self.window.speech_stopped("u2")
        self.window.speech_started("noise")
        self.window.speech_ignored("noise")
        await self.clock.advance(10)
        self.assertTrue(self.window.events.empty())
        self.assertTrue(self.interpret("u2", complete=False, question="还要改哪一项？"))
        await self.clock.drain()
        event = self.event("clarification_required")
        self.assertEqual(event.turn_id, "u2")

    async def test_execute_now_skips_delay_but_not_confirmation(self):
        self.add(["牛奶", "鸡蛋", "酱油"])
        self.finish_input(complete=True, execute_now=True)
        await self.clock.drain()
        self.event("confirmation_required")
        self.assertEqual(self.chain.batch(self.batch.id).status, "collecting")
        self.assertFalse(self.chain.queue(self.batch.id, version=1))

    async def test_execute_now_with_complete_simple_task_is_ready_immediately(self):
        self.add()
        self.finish_input(complete=True, execute_now=True)
        await self.clock.drain()
        self.event("ready")

    async def test_execute_now_cannot_force_an_incomplete_task(self):
        self.add()
        self.finish_input(complete=False, execute_now=True)
        await self.clock.advance(2)
        self.assertTrue(self.window.events.empty())
        await self.clock.advance(3)
        self.event("clarification_required")

    async def test_one_high_risk_task_requests_confirmation(self):
        with patch.dict(TOOL_POLICY, {"send_mail": ToolPolicy(True, False, "confirm")}):
            self.chain.add(self.batch.id, self.store.create("user", "send_mail", {}))
            self.finish_input(complete=True)
            await self.clock.advance(2)
            self.event("confirmation_required")

    async def test_queued_event_is_invalidated_by_new_speech(self):
        self.add()
        self.finish_input(complete=True)
        await self.clock.advance(2)
        event = self.event("ready")
        self.window.speech_started("u2")
        self.assertFalse(self.window.is_current(event))

    async def test_changed_batch_requires_fresh_interpretation(self):
        self.add()
        self.finish_input(complete=True)
        self.add(["鸡蛋"])
        self.assertFalse(self.window.interpreted("u1", version=1, complete=True))
        await self.clock.advance(2)
        self.assertTrue(self.window.events.empty())
        self.interpret(complete=True)
        await self.clock.drain()
        self.event("ready")

    async def test_version_change_invalidates_already_queued_event(self):
        self.add()
        self.finish_input(complete=True)
        await self.clock.advance(2)
        event = self.event("ready")
        self.add(["鸡蛋"])
        self.assertFalse(self.window.is_current(event))

    async def test_old_turn_and_duplicate_stop_cannot_reset_latest_window(self):
        self.add()
        self.finish_input(complete=True)
        self.window.speech_started("u2")
        self.assertFalse(self.interpret("u1", complete=True))
        self.assertFalse(self.window.speech_stopped("u1"))
        self.assertFalse(self.window.speech_started("u1"))
        self.window.speech_stopped("u2")
        self.interpret("u2", complete=True)
        await self.clock.advance(1)
        self.assertFalse(self.window.speech_stopped("u2"))
        self.interpret("u2", complete=True)  # 重复模型结果不能延长计时。
        await self.clock.advance(1)
        self.event("ready")

    async def test_empty_batch_does_not_create_work(self):
        task = self.add()
        self.chain.cancel(task.task_id)
        self.finish_input(complete=True)
        await self.clock.advance(2)
        self.event("empty")

    async def test_submitted_batch_does_not_emit_or_accept_more_speech(self):
        self.add()
        self.finish_input(complete=True)
        self.assertTrue(self.chain.queue(self.batch.id, version=1))
        await self.clock.advance(2)
        self.assertTrue(self.window.events.empty())
        self.assertFalse(self.window.speech_started("u2"))

    async def test_close_cancels_timer_and_invalidates_events_without_cancelling_tasks(self):
        task = self.add()
        self.finish_input(complete=True)
        await self.window.close()
        await self.clock.advance(10)
        self.assertTrue(self.window.events.empty())
        self.assertFalse(self.window.speech_started("u2"))
        self.assertEqual(self.chain.task(task.task_id).status, "pending")
        self.assertEqual(self.clock.waiters, [])

    async def test_custom_windows_and_invalid_configuration(self):
        for kwargs in ({"window_s": -1}, {"incomplete_s": float("inf")},
                       {"window_s": float("nan")}, {"window_s": 3, "incomplete_s": 2}):
            with self.assertRaises(ValueError):
                CollectionWindow(self.chain, self.batch.id, **kwargs)
        custom = CollectionWindow(self.chain, self.batch.id, window_s=1, incomplete_s=3,
                                  clock=self.clock, sleep=self.clock.sleep)
        self.addAsyncCleanup(custom.close)
        self.add()
        custom.speech_started("u1")
        custom.speech_stopped("u1")
        custom.interpreted("u1", version=1, complete=False)
        await self.clock.advance(2)
        self.assertTrue(custom.events.empty())
        await self.clock.advance(1)
        self.assertEqual(custom.events.get_nowait().kind, "clarification_required")


if __name__ == "__main__":
    unittest.main()
