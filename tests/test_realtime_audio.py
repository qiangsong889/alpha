"""Offline regression checks: no microphone, network, or paid model calls."""

import asyncio
import base64
import json
import sqlite3
import time
import unittest
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from mcp.types import CallToolResult, TextContent

from voice.audio import SpeechActivity, SpeechCandidate, StreamingPlayer, is_meaningful
import voice.qwen_realtime as qwen
import integrations.tool_executor as executor_module
from integrations.operations import OperationStore
from integrations.tool_gateway import ToolGateway


class AudioTests(unittest.TestCase):
    def test_next_item_preserves_unplayed_preamble_and_frame_offset(self):
        player = StreamingPlayer()
        player.ready = True
        player.append(np.full(600, 16384, dtype="<i2").tobytes(), "preamble", 0)
        first = np.zeros((240, 1), dtype=np.float32)
        player.callback(first, 240, None, None)
        player.append(np.full(480, 8192, dtype="<i2").tobytes(), "result", 0)
        self.assertEqual(player.offset, 240)
        self.assertEqual(player.item_id, "preamble")
        remaining = np.zeros((840, 1), dtype=np.float32)
        player.callback(remaining, 840, None, None)
        np.testing.assert_allclose(remaining[:360], .5)
        np.testing.assert_allclose(remaining[360:], .25)
        self.assertEqual(player.item_id, "result")
        self.assertEqual(player.played_frames, 480)
        self.assertFalse(player.buffers)
        self.assertFalse(player._buffer_items)

    def test_actual_interruption_discards_all_queued_items(self):
        player = StreamingPlayer()
        player.ready = True
        player.append(b"\x00\x40" * 600, "preamble", 0)
        player.append(b"\x00\x20" * 600, "result", 0)
        player.pause()
        self.assertEqual(player.clear(), ("preamble", 0, 0))
        self.assertFalse(player.buffers)
        self.assertFalse(player._buffer_items)
        output = np.ones((240, 1), dtype=np.float32)
        player.callback(output, 240, None, None)
        np.testing.assert_array_equal(output, 0)
        player.append(b"\x00\x20" * 240, "new", 0)
        player.callback(output, 240, None, None)
        np.testing.assert_allclose(output, .25)

    def test_clear_before_next_item_starts_reports_zero_heard_for_that_item(self):
        player = StreamingPlayer()
        player.ready = True
        player.append(b"\x00\x40" * 240, "preamble", 0)
        player.callback(np.zeros((240, 1), dtype=np.float32), 240, None, None)
        player.append(b"\x00\x20" * 240, "result", 1)
        self.assertEqual(player.clear(), ("result", 1, 0))

    def test_pause_preserves_audio_and_resume_continues(self):
        player = StreamingPlayer()
        player.ready = True
        player.append(np.full(2400, 16384, dtype="<i2").tobytes(), "answer", 0)
        output = np.zeros((240, 1), dtype=np.float32)
        player.duck(True)
        player.callback(output, 240, None, None)
        np.testing.assert_allclose(output, .1)
        player.pause()
        player.callback(output, 240, None, None)
        self.assertEqual(player.played_frames, 240)
        self.assertEqual(len(player.buffers), 1)
        np.testing.assert_array_equal(output, 0)
        player.resume()
        player.callback(output, 240, None, None)
        np.testing.assert_allclose(output, .5)
        self.assertEqual(player.clear(), ("answer", 0, 20))
        self.assertFalse(player.buffers)

    def test_vad_both_rates_and_short_gap(self):
        for rate in (16000, 24000):
            activity = SpeechActivity(rate)
            activity.vad = Mock()
            activity.vad.is_speech.return_value = True
            frame = np.zeros(rate // 50, dtype="<i2").tobytes()
            for _ in range(20):
                activity.observe(frame)
            self.assertEqual(activity.voiced_ms, 400)
            self.assertEqual(len(activity.vad.is_speech.call_args.args[0]), 640)
            activity.vad.is_speech.return_value = False
            for _ in range(4):
                activity.observe(frame)
            self.assertEqual(activity.voiced_ms, 400)
            activity.observe(frame)
            self.assertEqual(activity.voiced_ms, 0)

    def test_meaningful(self):
        for noise in ("", "。", "咳", "嗯，呃", "cough"):
            self.assertFalse(is_meaningful(noise))
        for text in ("好", "是", "等一下", "关客厅灯"):
            self.assertTrue(is_meaningful(text))


class PlaybackCompletionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.player = StreamingPlayer()
        self.player.ready = True
        self.waiters = []
        self.addAsyncCleanup(self.stop_waiters)

    async def stop_waiters(self):
        self.player.clear()
        for task in self.waiters:
            task.cancel()
        await asyncio.gather(*self.waiters, return_exceptions=True)

    def append(self, item_id="answer", frames=480, content_index=0):
        self.player.append(b"\x00\x40" * frames, item_id, content_index)

    def callback(self, frames):
        output = np.zeros((frames, 1), dtype=np.float32)
        self.player.callback(output, frames, None, None)
        return output

    async def waiting(self, item_id="answer"):
        task = asyncio.create_task(self.player.wait_played(item_id))
        self.waiters.append(task)
        await asyncio.sleep(0)
        return task

    async def test_finish_and_callback_drain_are_both_required(self):
        self.append()
        self.player.finish("answer")
        task = await self.waiting()
        self.callback(240)
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.callback(240)
        self.assertTrue(await asyncio.wait_for(task, 1))

    async def test_empty_buffer_without_finish_is_not_completion(self):
        self.append()
        task = await self.waiting()
        self.callback(480)
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        # 生成慢时出现临时空队列；后续音频仍必须全部取完。
        self.append(frames=240)
        self.player.finish("answer")
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.callback(240)
        self.assertTrue(await asyncio.wait_for(task, 1))

    async def test_finish_after_buffer_was_drained_completes(self):
        self.append()
        task = await self.waiting()
        self.callback(480)
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.player.finish("answer")
        self.assertTrue(await asyncio.wait_for(task, 1))

    async def test_clear_halfway_returns_false(self):
        self.append()
        self.player.finish("answer")
        task = await self.waiting()
        self.callback(240)
        self.player.clear()
        self.assertFalse(await asyncio.wait_for(task, 1))

    async def test_pause_keeps_waiting_and_resume_completes(self):
        self.append()
        self.player.finish("answer")
        task = await self.waiting()
        self.player.pause()
        output = self.callback(480)
        np.testing.assert_array_equal(output, 0)
        self.assertEqual(self.player.played_frames, 0)
        self.assertFalse(task.done())
        self.player.resume()
        self.callback(480)
        self.assertTrue(await asyncio.wait_for(task, 1))

    async def test_each_item_finishes_independently(self):
        self.append("first", frames=240)
        self.append("second", frames=480)
        self.player.finish("first")
        self.player.finish("second")
        first, second = await self.waiting("first"), await self.waiting("second")
        self.callback(240)
        self.assertTrue(await asyncio.wait_for(first, 1))
        self.assertFalse(second.done())
        self.callback(480)
        self.assertTrue(await asyncio.wait_for(second, 1))

    async def test_output_latency_is_waited_after_last_callback(self):
        self.player.latency = .05
        self.append()
        self.player.finish("answer")
        task = await self.waiting()
        started = time.perf_counter()
        self.callback(480)
        await asyncio.sleep(.005)
        self.assertFalse(task.done())
        self.assertTrue(await asyncio.wait_for(task, 1))
        self.assertGreaterEqual(time.perf_counter() - started, .05)

    async def test_clear_during_output_latency_returns_false(self):
        self.player.latency = .1
        self.append()
        self.player.finish("answer")
        task = await self.waiting()
        self.callback(480)
        await asyncio.sleep(.005)
        self.assertFalse(task.done())
        self.player.clear()
        self.assertFalse(await asyncio.wait_for(task, 1))

    async def test_clear_before_finish_stays_false_when_finish_arrives_late(self):
        self.append()
        task = await self.waiting()
        self.player.clear()
        self.player.finish("answer")
        self.assertFalse(await asyncio.wait_for(task, 1))
        self.assertFalse(await self.player.wait_played("answer"))
        self.append()  # 旧回答晚到的块不能重新加入缓冲区。
        self.assertFalse(self.player.buffers)

    async def test_items_without_audio_return_false_without_waiting(self):
        self.assertFalse(await asyncio.wait_for(self.player.wait_played("unknown"), .1))
        self.player.finish("empty")
        self.assertFalse(await asyncio.wait_for(self.player.wait_played("empty"), .1))
        self.player.append(b"", "empty-chunk", 0)
        self.player.finish("empty-chunk")
        self.assertFalse(await self.player.wait_played("empty-chunk"))

    async def test_unavailable_device_does_not_report_partial_audio_as_complete(self):
        self.player.ready = False
        self.append()
        self.player.ready = True
        self.append()
        self.player.finish("answer")
        self.assertFalse(await self.player.wait_played("answer"))
        self.assertFalse(self.player.buffers)

    async def test_multiple_waiters_and_cancelled_waiter_do_not_interfere(self):
        self.append()
        first, second = await self.waiting(), await self.waiting()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        self.player.finish("answer")
        self.callback(480)
        self.assertTrue(await asyncio.wait_for(second, 1))
        self.assertTrue(await self.player.wait_played("answer"))

    async def test_callback_from_another_thread_resolves_on_event_loop(self):
        self.append()
        self.player.finish("answer")
        task = await self.waiting()
        await asyncio.to_thread(self.callback, 480)
        self.assertTrue(await asyncio.wait_for(task, 1))

    async def test_clear_before_queued_completion_notification_returns_false(self):
        self.append()
        self.player.finish("answer")
        task = await self.waiting()
        self.callback(480)
        self.player.clear()  # True 通知仍在事件循环队列中，不能覆盖这次打断。
        self.assertFalse(await asyncio.wait_for(task, 1))

    async def test_finished_item_remains_true_after_later_items_are_cleared(self):
        self.append("first", frames=240)
        self.player.finish("first")
        first = await self.waiting("first")
        self.callback(240)
        self.assertTrue(await asyncio.wait_for(first, 1))
        self.append("second")
        second = await self.waiting("second")
        self.player.clear()
        self.assertFalse(await asyncio.wait_for(second, 1))
        self.assertTrue(await self.player.wait_played("first"))

    async def test_recent_terminal_records_are_limited_to_200_in_item_order(self):
        for index in range(250):
            self.player.finish(f"empty-{index}")
        self.assertEqual(len(self.player._playbacks), 200)
        self.assertEqual(list(self.player._playbacks), [f"empty-{index}" for index in range(50, 250)])

    async def test_recent_cleared_record_still_blocks_late_audio(self):
        self.append("cleared")
        self.player.clear()
        for index in range(199):
            self.player.finish(f"empty-{index}")
        self.assertIn("cleared", self.player._playbacks)
        self.append("cleared")
        self.assertFalse(self.player.buffers)
        self.assertFalse(await self.player.wait_played("cleared"))

    async def test_old_active_record_can_exceed_limit_then_prunes_after_playback(self):
        self.append("active")
        self.player.finish("active")
        for index in range(250):
            self.player.finish(f"empty-{index}")
        self.assertEqual(len(self.player._playbacks), 201)
        self.assertIn("active", self.player._playbacks)
        task = await self.waiting("active")
        self.callback(480)
        self.assertTrue(await asyncio.wait_for(task, 1))
        self.assertNotIn("active", self.player._playbacks)
        self.assertEqual(len(self.player._playbacks), 200)

    async def test_terminal_record_is_retained_until_its_waiter_exits(self):
        self.append("old")
        task = await self.waiting("old")
        self.player.clear()
        # clear 的 False 通知还在事件循环队列里；此时不能删掉正在等待的记录。
        for index in range(250):
            self.player.finish(f"empty-{index}")
        self.assertIn("old", self.player._playbacks)
        self.assertEqual(len(self.player._playbacks), 201)
        self.assertFalse(await asyncio.wait_for(task, 1))
        self.assertNotIn("old", self.player._playbacks)
        self.assertEqual(len(self.player._playbacks), 200)

    async def test_unwaited_completed_audio_does_not_accumulate_forever(self):
        self.append("old")
        self.player.finish("old")
        self.callback(480)
        for index in range(250):
            self.player.finish(f"empty-{index}")
        self.assertNotIn("old", self.player._playbacks)
        self.assertEqual(len(self.player._playbacks), 200)

    async def test_clear_prunes_old_active_records_after_clearing_their_audio(self):
        for index in range(210):
            self.append(f"active-{index}")
        self.assertEqual(len(self.player._playbacks), 210)
        self.player.clear()
        self.assertFalse(self.player.buffers)
        self.assertEqual(len(self.player._playbacks), 200)
        self.assertEqual(list(self.player._playbacks)[0], "active-10")

    async def test_old_device_failure_record_is_not_pruned_while_audio_is_still_queued(self):
        self.append("old")
        self.player.ready = False
        self.append("old")
        self.player.ready = True
        for index in range(250):
            self.player.finish(f"empty-{index}")
        self.assertIn("old", self.player._playbacks)
        self.callback(480)  # 队列里仍有音频时，callback 必须还能找到它的记录。
        self.assertNotIn("old", self.player._playbacks)


class QwenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ws = Mock()
        self.ws.send = AsyncMock()
        self.player = StreamingPlayer()
        self.player.ready = True
        self.activity = SpeechActivity(16000)
        self.receiver = self.make_receiver(self.ws, self.player, self.activity,
                                          qwen.CostTracker("singapore"), "session")

    def make_receiver(self, ws, player, activity, costs, session_id, worker=None, tools=None,
                      *, ha_url="", http_client=None):
        store = OperationStore(":memory:")
        self.addCleanup(store.close)
        gateway = ToolGateway(store, executor_module.make_executor(
            ha_url=ha_url, http_client=http_client, worker=worker,
        ))
        window = qwen.CommitWindow(gateway, window_s=.05)
        self.addAsyncCleanup(window.close)
        receiver = qwen.QwenReceiver(ws, player, activity, costs, session_id, gateway=gateway,
                                     commit_window=window, coordinator=qwen.PlanCoordinator(gateway), tools=tools)
        self.addAsyncCleanup(receiver.close_playback_tasks)
        return receiver

    def sent(self):
        return [json.loads(call.args[0]) for call in self.ws.send.call_args_list]

    async def emit(self, kind, **fields):
        await self.receiver.handle({"type": kind, **fields}, None)

    async def plan_tool_round(self, name, arguments, response_id, tasks):
        receiver = self.receiver
        if not receiver.response_pending:
            receiver.response_pending = True
            receiver.requested_candidate = receiver.candidate
        await receiver.handle({"type": "response.created", "response": {"id": response_id}}, tasks)
        await receiver.handle({"type": "response.function_call_arguments.done", "response_id": response_id,
                               "call_id": f"call-{response_id}", "name": name,
                               "arguments": json.dumps(arguments, ensure_ascii=False)}, tasks)
        await receiver.handle({"type": "response.done", "response": {
            "id": response_id, "status": "completed",
        }}, tasks)

    async def prepare_qwen_plan(self):
        receiver = self.receiver
        receiver.gateway.executor = AsyncMock(return_value=executor_module.ExecutorResult(True, "已添加牛奶和鸡蛋"))
        receiver.tool_names = {"add_items"}
        receiver.candidate = SpeechCandidate("request", 0, meaningful=True, committed=True, responded=True)
        receiver.last_meaningful_candidate = receiver.candidate
        receiver.coordinator.user_turn(receiver.coordinator_turn(receiver.candidate))
        async with asyncio.TaskGroup() as tasks:
            await self.plan_tool_round("add_items", {"list_name": "购物清单", "items": ["牛奶", "鸡蛋"]},
                                       "register", tasks)
        # 工具完成后直接请求模板复述，不再生成普通确认回答。
        return receiver.coordinator.plan

    async def recap_audio(self, response_id, tasks, *, consume=True):
        receiver = self.receiver
        await receiver.handle({"type": "response.created", "response": {"id": response_id}}, tasks)
        await receiver.handle({"type": "response.audio.delta", "response_id": response_id,
                               "item_id": f"audio-{response_id}",
                               "delta": base64.b64encode(b"\x00\x40" * 480).decode()}, tasks)
        await receiver.handle({"type": "response.audio.done", "response_id": response_id,
                               "item_id": f"audio-{response_id}"}, tasks)
        await receiver.handle({"type": "response.done", "response": {
            "id": response_id, "status": "completed", "output": [{
                "id": f"audio-{response_id}", "type": "message", "content": [{
                    "type": "audio", "transcript": receiver.recaps[response_id].text,
                }],
            }],
        }}, tasks)
        await asyncio.sleep(0)
        if consume:
            self.player.callback(np.zeros((480, 1), dtype=np.float32), 480, None, None)

    async def plan_user_input(self, text, item_id, tasks):
        await self.receiver.handle({"type": "input_audio_buffer.speech_started", "item_id": item_id,
                                   "audio_start_ms": 0}, tasks)
        await self.receiver.handle({"type": "input_audio_buffer.committed", "item_id": item_id}, tasks)
        await self.receiver.handle({"type": "conversation.item.input_audio_transcription.completed",
                                   "item_id": item_id, "transcript": text}, tasks)

    def last_tool_output(self):
        return next(json.loads(event["item"]["output"]) for event in reversed(self.sent())
                    if event.get("item", {}).get("type") == "function_call_output")

    async def test_multi_action_plan_only_executes_after_recap_playback_and_user_confirmation(self):
        plan = await self.prepare_qwen_plan()
        self.assertIsNotNone(plan)
        self.receiver.gateway.executor.assert_not_awaited()
        self.assertEqual(len(self.receiver.commit_window.pending), 0)
        self.assertEqual(plan.status, "collecting")
        texts = [event["item"]["content"][0]["text"] for event in self.sent()
                 if event.get("item", {}).get("role") == "system"]
        expected = plan.recap(datetime.now(self.receiver.timezone))
        self.assertTrue(any(text.endswith(expected) and "逐字念出" in text for text in texts))
        async with asyncio.TaskGroup() as tasks:
            await self.recap_audio("recap", tasks, consume=False)
            self.assertEqual(plan.status, "collecting")
            self.player.callback(np.zeros((480, 1), dtype=np.float32), 480, None, None)
        self.assertEqual(plan.status, "awaiting_confirmation")
        async with asyncio.TaskGroup() as tasks:
            await self.plan_user_input("对", "yes", tasks)
            await self.plan_tool_round("confirm_plan", {"plan_id": plan.id, "version": plan.version},
                                       "confirm", tasks)
        self.receiver.gateway.executor.assert_awaited_once_with(
            "add_items", {"list_name": "购物清单", "items": ["牛奶", "鸡蛋"]},
        )
        self.assertEqual(plan.status, "done")
        output = self.last_tool_output()
        self.assertEqual(output["reason"], "completed")
        self.assertEqual(output["operations"][0]["status"], "succeeded")
        self.assertEqual(output["version"], plan.version)
        requests = sum(event["type"] == "response.create" for event in self.sent())
        with patch("builtins.print") as printed:
            await self.emit("response.created", response={"id": "confirmed-answer"})
            await self.emit("response.done", response={"id": "confirmed-answer", "status": "completed",
                            "output": [{"type": "message", "content": [{
                                "type": "text", "text": "牛奶和鸡蛋已经加好了",
                            }]}]})
        self.assertFalse(any("疑似虚构" in str(call) for call in printed.call_args_list))
        self.assertEqual(sum(event["type"] == "response.create" for event in self.sent()), requests)

    async def test_shopping_and_reminder_batch_continues_directly_with_recap_in_either_event_order(self):
        for tools_first in (False, True):
            with self.subTest(tools_first=tools_first):
                self.ws.send.reset_mock()
                receiver = self.make_receiver(self.ws, self.player, self.activity,
                                             self.receiver.costs, f"session-{tools_first}")
                receiver.gateway.executor = AsyncMock()
                receiver.tool_names = {"add_items", "create_reminder"}
                receiver.candidate = SpeechCandidate("request", 0, meaningful=True, committed=True, responded=True)
                receiver.last_meaningful_candidate = receiver.requested_candidate = receiver.candidate
                receiver.coordinator.user_turn(receiver.coordinator_turn(receiver.candidate))
                receiver.response_pending = True
                async with asyncio.TaskGroup() as tasks:
                    await receiver.handle({"type": "response.created", "response": {"id": "tools"}}, tasks)
                    for name, arguments in (
                        ("add_items", {"list_name": "购物清单", "items": ["牛奶", "鸡蛋"]}),
                        ("create_reminder", {"title": "开会", "local_time": "2026-10-16T08:00:00"}),
                    ):
                        await receiver.handle({"type": "response.function_call_arguments.done",
                                               "response_id": "tools", "call_id": name, "name": name,
                                               "arguments": json.dumps(arguments)}, tasks)
                    if tools_first:
                        await asyncio.sleep(0)
                        self.assertFalse(receiver.response_pending)
                    await receiver.handle({"type": "response.done", "response": {
                        "id": "tools", "status": "completed", "output": [{
                            "type": "message", "content": [{"type": "text", "text": "提醒已经设好了"}],
                        }],
                    }}, tasks)
                receiver.gateway.executor.assert_not_awaited()
                self.assertEqual(len(receiver.coordinator.plan.items), 2)
                self.assertEqual(receiver.pending_recap.version, 2)
                self.assertEqual(receiver.correction_counts, {})
                self.assertEqual(sum(event["type"] == "response.create" for event in self.sent()), 1)
                texts = [event["item"]["content"][0]["text"] for event in self.sent()
                         if event.get("item", {}).get("role") == "system"]
                self.assertTrue(any("逐字念出" in text and "牛奶、鸡蛋" in text and "开会" in text for text in texts))
                statuses = receiver.gateway.store._db.execute("SELECT status FROM operations").fetchall()
                self.assertEqual([row["status"] for row in statuses], ["pending", "pending"])

    async def test_recap_interrupted_halfway_does_not_open_confirmation(self):
        plan = await self.prepare_qwen_plan()
        async with asyncio.TaskGroup() as tasks:
            await self.recap_audio("recap", tasks, consume=False)
            self.player.callback(np.zeros((240, 1), dtype=np.float32), 240, None, None)
            await self.plan_user_input("对", "early-yes", tasks)
            await self.plan_tool_round("confirm_plan", {"plan_id": plan.id, "version": plan.version},
                                       "early-confirm", tasks)
        self.assertEqual(self.last_tool_output()["reason"], "not_played")
        self.receiver.gateway.executor.assert_not_awaited()

    async def test_tools_in_recap_response_are_rejected_including_confirmation(self):
        plan = await self.prepare_qwen_plan()
        async with asyncio.TaskGroup() as tasks:
            await self.plan_tool_round("confirm_plan", {"plan_id": plan.id, "version": plan.version},
                                       "recap", tasks)
        output = self.last_tool_output()
        self.assertEqual(output["reason"], "recap_only")
        self.assertTrue(output["is_error"])
        self.assertEqual(plan.status, "collecting")
        self.receiver.gateway.executor.assert_not_awaited()
        self.assertEqual(len(self.receiver.gateway.store._db.execute("SELECT * FROM operations").fetchall()), 1)
        self.assertTrue(self.receiver.silent_response_pending)

    async def test_waiting_plan_collects_new_operation_and_recaps_new_version(self):
        plan = await self.prepare_qwen_plan()
        async with asyncio.TaskGroup() as tasks:
            await self.recap_audio("recap", tasks)
        version = plan.version
        async with asyncio.TaskGroup() as tasks:
            await self.plan_user_input("再加面包", "more", tasks)
            await self.plan_tool_round("add_items", {"list_name": "购物清单", "items": ["面包"]},
                                       "add-more", tasks)
        self.assertEqual(plan.version, version + 1)
        self.assertEqual(plan.status, "collecting")
        self.assertEqual(len(plan.items), 2)
        self.assertEqual(self.receiver.pending_recap.version, plan.version)
        self.receiver.gateway.executor.assert_not_awaited()

    async def test_control_update_and_remove_in_one_response_trigger_one_new_recap(self):
        plan = await self.prepare_qwen_plan()
        async with asyncio.TaskGroup() as tasks:
            await self.recap_audio("recap", tasks)
        # 再收集一个独立条目，供同一轮修改和删除。
        extra = await self.receiver.gateway.submit("setup", "add_items", {"list_name": "购物清单", "items": ["面包"]})
        self.receiver.coordinator.collect(extra)
        async with asyncio.TaskGroup() as tasks:
            await self.plan_user_input("第一项改成牛奶，第二项不要了", "edit", tasks)
            await self.emit("response.created", response={"id": "edit-tools"})
            for call_id, name, arguments in (
                ("update", "update_plan_item", {"plan_id": plan.id, "item_id": 1,
                                               "arguments": {"list_name": "购物清单", "items": ["牛奶"]}}),
                ("remove", "remove_plan_item", {"plan_id": plan.id, "item_id": 2}),
            ):
                await self.receiver.handle({"type": "response.function_call_arguments.done",
                                            "response_id": "edit-tools", "call_id": call_id, "name": name,
                                            "arguments": json.dumps(arguments)}, tasks)
            await self.receiver.handle({"type": "response.done", "response": {
                "id": "edit-tools", "status": "completed",
            }}, tasks)
        self.assertEqual(len(plan.items), 1)
        self.assertEqual(plan.items[0].arguments["items"], ["牛奶"])
        self.assertEqual(self.receiver.pending_recap.version, plan.version)
        outputs = [json.loads(event["item"]["output"]) for event in self.sent()
                   if event.get("item", {}).get("call_id") in {"update", "remove"}]
        self.assertTrue(all(not output["is_error"] for output in outputs))
        self.receiver.gateway.executor.assert_not_awaited()

    async def test_reconnect_reuses_plan_and_requires_new_recap_playback(self):
        plan = await self.prepare_qwen_plan()
        async with asyncio.TaskGroup() as tasks:
            await self.recap_audio("recap", tasks)
        self.assertEqual(plan.status, "awaiting_confirmation")
        coordinator = self.receiver.coordinator
        coordinator.on_reconnect()
        ws = Mock(send=AsyncMock())
        receiver = qwen.QwenReceiver(ws, self.player, self.activity, self.receiver.costs, "new-session",
                                     gateway=self.receiver.gateway, commit_window=self.receiver.commit_window,
                                     coordinator=coordinator)
        self.addAsyncCleanup(receiver.close_playback_tasks)
        await receiver.maybe_respond()
        self.assertIs(receiver.coordinator.plan, plan)
        self.assertEqual(plan.status, "collecting")
        self.assertEqual(receiver.pending_recap.plan_id, plan.id)
        events = [json.loads(call.args[0]) for call in ws.send.call_args_list]
        self.assertEqual(events[-1]["type"], "response.create")
        self.assertTrue(any(plan.id in json.dumps(event, ensure_ascii=False) for event in events))
        self.receiver.gateway.executor.assert_not_awaited()

    async def test_invalid_plan_operation_is_cancelled_and_reported(self):
        receiver = self.receiver
        receiver.candidate = SpeechCandidate("bad", 0, meaningful=True, responded=True)
        receiver.finished_responses["final"] = "completed"
        receiver.response_turns["final"] = "bad"
        op = await receiver.gateway.submit("bad", "add_items", {"items": ["牛奶", "鸡蛋"]})
        receiver.commit_window.add(op)
        await receiver.maybe_finish_turn()
        self.assertEqual(receiver.gateway.store.get(op.id).status, "cancelled")
        self.assertIsNone(receiver.coordinator.plan)
        texts = [event["item"]["content"][0]["text"] for event in self.sent()
                 if event.get("item", {}).get("role") == "system"]
        self.assertTrue(any("这一项没有加进计划" in text and "invalid_arguments" in text for text in texts))

    async def test_plan_busy_cancels_newly_registered_operation_without_affecting_execution(self):
        plan = await self.prepare_qwen_plan()
        async with asyncio.TaskGroup() as tasks:
            await self.recap_audio("recap", tasks)
        receiver = self.receiver
        entered, release = asyncio.Event(), asyncio.Event()

        async def execute(*args):
            entered.set()
            await release.wait()
            return executor_module.ExecutorResult(True, "已添加")

        receiver.gateway.executor.side_effect = execute
        receiver.coordinator.user_turn("confirmed")
        execution = asyncio.create_task(receiver.coordinator.confirm(plan.id, plan.version))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            receiver.candidate = SpeechCandidate("more", 100, meaningful=True, responded=True)
            receiver.finished_responses["more-final"] = "completed"
            receiver.response_turns["more-final"] = "more"
            op = await receiver.gateway.submit("more", "add_items", {"list_name": "购物清单", "items": ["面包"]})
            receiver.commit_window.add(op)
            await receiver.maybe_finish_turn()
            self.assertEqual(receiver.gateway.store.get(op.id).status, "cancelled")
            texts = [event["item"]["content"][0]["text"] for event in self.sent()
                     if event.get("item", {}).get("role") == "system"]
            self.assertTrue(any("这一项没有加进计划（busy）" in text for text in texts))
            release.set()
            self.assertTrue((await execution).ok)
        finally:
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)

    async def test_confirm_schema_errors_include_snapshot_without_gateway_execution(self):
        plan = await self.prepare_qwen_plan()
        async with asyncio.TaskGroup() as tasks:
            await self.recap_audio("recap", tasks)
        async with asyncio.TaskGroup() as tasks:
            await self.plan_user_input("对", "yes", tasks)
            await self.plan_tool_round("confirm_plan", {"plan_id": plan.id, "version": True},
                                       "wrong-type", tasks)
        output = self.last_tool_output()
        self.assertEqual(output["reason"], "invalid_arguments")
        self.assertEqual(output["version"], plan.version)
        self.assertEqual(output["snapshot"]["id"], plan.id)
        self.receiver.gateway.executor.assert_not_awaited()

    async def test_two_short_utterances_transfer_window_actions_into_one_plan(self):
        receiver = self.receiver
        receiver.tool_names = {"add_items"}
        receiver.gateway.executor = AsyncMock(return_value=executor_module.ExecutorResult(True, "已添加"))
        for item_id, title in (("milk", "牛奶"), ("eggs", "鸡蛋")):
            async with asyncio.TaskGroup() as tasks:
                await self.plan_user_input(f"加{title}", item_id, tasks)
                await self.plan_tool_round("add_items", {"list_name": "购物清单", "items": [title]},
                                           f"tool-{item_id}", tasks)
            if receiver.pending_recap is None:
                await self.emit("response.created", response={"id": f"ack-{item_id}"})
                await self.emit("response.done", response={"id": f"ack-{item_id}", "status": "completed"})
        self.assertEqual(len(receiver.coordinator.plan.items), 2)
        self.assertEqual(receiver.commit_window.pending, [])
        receiver.gateway.executor.assert_not_awaited()

    async def test_claim_without_tool_warns_and_requests_one_correction(self):
        response = {"id": "made-up", "status": "completed", "output": [{
            "type": "message", "role": "assistant", "content": [{"type": "audio", "transcript": "已经设好了。"}],
        }]}
        with patch("builtins.print") as printed:
            await self.emit("response.done", response=response)
            self.assertTrue(any("[⚠️ 疑似虚构]" in str(call) for call in printed.call_args_list))
            events = self.sent()
            self.assertEqual(sum(event["type"] == "response.create" for event in events), 1)
            correction = events[0]["item"]
            self.assertEqual(correction["role"], "system")
            self.assertIn("没有成功的写入工具结果", correction["content"][0]["text"])
            # 重复 done 不会重复纠正；纠正回答仍然瞎说时，也不会无限请求。
            await self.emit("response.done", response=response)
            await self.emit("response.created", response={"id": "correction"})
            await self.emit("response.done", response={**response, "id": "correction"})
            self.assertEqual(sum(event["type"] == "response.create" for event in self.sent()), 1)

    async def test_claim_can_be_detected_from_transcript_events(self):
        await self.emit("response.audio_transcript.delta", response_id="r", item_id="answer", delta="设置")
        await self.emit("response.audio_transcript.done", response_id="r", item_id="answer", transcript="设置好了")
        with patch("builtins.print") as printed:
            await self.emit("response.done", response={"id": "r", "status": "completed"})
            self.assertTrue(any("[⚠️ 疑似虚构]" in str(call) for call in printed.call_args_list))
        self.assertEqual(self.sent()[-1]["type"], "response.create")

    async def test_real_tool_success_allows_completion_across_followup_response(self):
        worker = Mock(closed=False, call=AsyncMock(return_value=CallToolResult(
            content=[TextContent(type="text", text="已添加牛奶")],
        )))
        receiver = self.make_receiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker,
                                     [{"name": "add_items", "input_schema": {}}])
        async with asyncio.TaskGroup() as tasks:
            await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "tool-response",
                                   "call_id": "call", "name": "add_items",
                                   "arguments": '{"list_name":"购物清单","items":["牛奶"]}'}, tasks)
            await receiver.handle({"type": "response.done", "response": {
                "id": "tool-response", "status": "completed",
            }}, tasks)
        await receiver.handle({"type": "response.created", "response": {"id": "answer"}}, None)
        # pending 的工具后续回答只确认登记；真正的成功报告在窗口执行后才生成。
        await receiver.handle({"type": "response.done", "response": {
            "id": "answer", "status": "completed", "output": [{
                "type": "message", "content": [{"type": "text", "text": "好的，记下了"}],
            }],
        }}, None)
        worker.call.assert_not_awaited()
        report = await asyncio.wait_for(receiver.commit_window.reports.get(), 1)
        receiver.commit_window.reports.put_nowait(report)
        await receiver.maybe_respond()
        await receiver.handle({"type": "response.created", "response": {"id": "report"}}, None)
        with patch("builtins.print") as printed:
            await receiver.handle({"type": "response.done", "response": {
                "id": "report", "status": "completed", "output": [{
                    "type": "message", "content": [{"type": "text", "text": "已经加了牛奶"}],
                }],
            }}, None)
            self.assertFalse(any("[⚠️ 疑似虚构]" in str(call) for call in printed.call_args_list))
        self.assertEqual(sum(event["type"] == "response.create" for event in self.sent()), 2)
        # 下一句用户请求不能借用上一句的成功记录。
        receiver.candidate = SpeechCandidate("next-user", 100, committed=True, meaningful=True, responded=True)
        receiver.requested_candidate = receiver.candidate
        with patch("builtins.print") as printed:
            await receiver.handle({"type": "response.done", "response": {
                "id": "new-made-up", "status": "completed", "output": [{
                    "type": "message", "content": [{"type": "text", "text": "已经设好了"}],
                }],
            }}, None)
            self.assertTrue(any("[⚠️ 疑似虚构]" in str(call) for call in printed.call_args_list))

    async def test_pending_write_claim_correction_states_facts_without_repeating_tool(self):
        worker = Mock(closed=False, call=AsyncMock())
        receiver = self.make_receiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker,
                                     [{"name": "create_reminder", "input_schema": {}}])
        with patch("builtins.print") as printed:
            async with asyncio.TaskGroup() as tasks:
                await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                       "call_id": "call", "name": "create_reminder", "arguments": "{}"}, tasks)
                await receiver.handle({"type": "response.done", "response": {
                    "id": "r", "status": "completed", "output": [{
                        "type": "message", "content": [{"text": "已经设好了"}],
                    }],
                }}, tasks)
            self.assertTrue(any("[⚠️ 疑似虚构]" in str(call) for call in printed.call_args_list))
        corrections = [event["item"] for event in self.sent() if event.get("item", {}).get("role") == "system"]
        self.assertIn("还没有执行", corrections[0]["content"][0]["text"])
        self.assertIn("不得再次调用工具", corrections[0]["content"][0]["text"])
        worker.call.assert_not_awaited()
        self.assertEqual(sum(event["type"] == "response.create" for event in self.sent()), 1)

    async def test_delayed_report_waits_for_current_response_and_preserves_failures(self):
        receiver = self.receiver
        receiver.gateway.executor = AsyncMock(side_effect=TimeoutError("响应丢失"))
        op = await receiver.gateway.submit("session:<no-user-item>", "create_reminder", {})
        receiver.commit_window.add(op)
        receiver.commit_window.turn_finished()
        report = await asyncio.wait_for(receiver.commit_window.reports.get(), 1)
        receiver.commit_window.reports.put_nowait(report)
        receiver.response_pending = True
        watcher = asyncio.create_task(receiver.watch_reports())
        self.addAsyncCleanup(self.cancel_task, watcher)
        await asyncio.sleep(0)
        self.assertEqual(self.sent(), [])
        # 先前已请求的正常回答不能被误认为是报告回答。
        await self.emit("response.created", response={"id": "normal"})
        self.assertIsNone(receiver.report_response)
        await self.emit("response.done", response={"id": "normal", "status": "completed"})
        await asyncio.sleep(0)
        events = self.sent()
        self.assertEqual(sum(e["type"] == "response.create" for e in events), 1)
        text = events[0]["item"]["content"][0]["text"]
        self.assertIn("unknown", text)
        self.assertIn("不要重试", text)
        self.assertIn("只根据这份执行结果", text)

    @staticmethod
    async def cancel_task(task):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_noise_transcription_after_final_done_still_starts_commit_window(self):
        receiver = self.receiver
        receiver.gateway.executor = AsyncMock(return_value=executor_module.ExecutorResult(True, "已添加"))
        original = SpeechCandidate("user", 0, meaningful=True, committed=True, responded=True)
        receiver.candidate = receiver.last_meaningful_candidate = original
        receiver.requested_candidate = original
        op = await receiver.gateway.submit("session:user", "add_items", {"items": ["牛奶"]})
        receiver.commit_window.add(op)
        receiver.current_response = "ack"
        receiver.response_turns["ack"] = "user"
        async with asyncio.TaskGroup() as tasks:
            # 回答结束前出现一声噪音，转写比 response.done 更晚到达。
            await receiver.handle({"type": "input_audio_buffer.speech_started",
                                   "item_id": "noise", "audio_start_ms": 100}, tasks)
            await self.emit("response.done", response={"id": "ack", "status": "completed"})
            await asyncio.sleep(.08)
            receiver.gateway.executor.assert_not_awaited()
            await self.emit("conversation.item.input_audio_transcription.completed",
                            item_id="noise", transcript="咳")
            report = await asyncio.wait_for(receiver.commit_window.reports.get(), .3)
            self.assertEqual(report.operations[0].status, "succeeded")

    async def test_old_response_done_does_not_resume_window_after_new_meaningful_input(self):
        receiver = self.receiver
        executor = AsyncMock(return_value=executor_module.ExecutorResult(True, "成功"))
        receiver.gateway.executor = executor
        op = await receiver.gateway.submit("session:old", "add_items", {})
        receiver.commit_window.add(op)
        receiver.commit_window.turn_finished()
        receiver.commit_window.speech_started()
        receiver.commit_window.speech_accepted()
        receiver.candidate = SpeechCandidate("new", 100, meaningful=True)
        receiver.response_turns["old-answer"] = "old"
        await self.emit("response.done", response={"id": "old-answer", "status": "completed"})
        await asyncio.sleep(.08)
        executor.assert_not_awaited()

    async def test_disconnect_returns_unfinished_report_to_shared_window(self):
        receiver = self.receiver
        receiver.gateway.executor = AsyncMock(return_value=executor_module.ExecutorResult(True, "已添加牛奶"))
        op = await receiver.gateway.submit("turn", "add_items", {})
        receiver.commit_window.add(op)
        receiver.commit_window.turn_finished()
        report = await asyncio.wait_for(receiver.commit_window.reports.get(), 1)
        receiver.active_report = report
        entered = asyncio.Event()

        class Socket:
            send = AsyncMock()

            def __aiter__(self):
                return self

            async def __anext__(self):
                entered.set()
                await asyncio.Future()

        receiver.ws = Socket()
        task = asyncio.create_task(receiver.run())
        await asyncio.wait_for(entered.wait(), 1)
        await self.cancel_task(task)
        self.assertIsNone(receiver.active_report)
        self.assertIs(receiver.commit_window.reports.get_nowait(), report)
        self.assertEqual(receiver.gateway.store.get(op.id).status, "succeeded")

    async def test_speech_events_pause_noise_restart_and_accept_pause_window(self):
        receiver = self.receiver
        receiver.gateway.executor = AsyncMock(return_value=executor_module.ExecutorResult(True, "成功"))
        op = await receiver.gateway.submit("turn", "add_items", {})
        receiver.commit_window.add(op)
        receiver.commit_window.turn_finished()
        async with asyncio.TaskGroup() as tasks:
            await receiver.handle({"type": "input_audio_buffer.speech_started",
                                   "item_id": "noise", "audio_start_ms": 0}, tasks)
            await asyncio.sleep(.08)
            receiver.gateway.executor.assert_not_awaited()
            await receiver.handle({"type": "conversation.item.input_audio_transcription.completed",
                                   "item_id": "noise", "transcript": "咳"}, tasks)
            report = await asyncio.wait_for(receiver.commit_window.reports.get(), 1)
            self.assertEqual(report.operations[0].status, "succeeded")
            await receiver.handle({"type": "input_audio_buffer.speech_started",
                                   "item_id": "user", "audio_start_ms": 100}, tasks)
            await receiver.handle({"type": "conversation.item.input_audio_transcription.completed",
                                   "item_id": "user", "transcript": "再加一个鸡蛋"}, tasks)
            other = await receiver.gateway.submit("turn2", "add_items", {"items": ["鸡蛋"]})
            receiver.commit_window.add(other)
            await asyncio.sleep(.08)
            self.assertEqual(receiver.gateway.executor.await_count, 1)
            receiver.commit_window.turn_finished()
            self.assertEqual((await asyncio.wait_for(receiver.commit_window.reports.get(), 1)).operations[0].id,
                             other.id)

    async def test_read_only_success_does_not_prove_a_write_happened(self):
        receiver = self.receiver
        receiver.candidate = SpeechCandidate("user", 0, responded=True, meaningful=True)
        receiver.requested_candidate = receiver.candidate
        worker = Mock(closed=False, call=AsyncMock(return_value=CallToolResult(
            content=[TextContent(type="text", text="清单里有牛奶")],
        )))
        receiver.gateway.executor = executor_module.make_executor(ha_url="", http_client=None, worker=worker)
        receiver.tool_names = {"list_items"}
        with patch("builtins.print") as printed:
            async with asyncio.TaskGroup() as tasks:
                await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                       "call_id": "read", "name": "list_items", "arguments": "{}"}, tasks)
                await receiver.handle({"type": "response.done", "response": {
                    "id": "r", "status": "completed", "output": [{
                        "type": "message", "content": [{"text": "已添加牛奶"}],
                    }],
                }}, tasks)
            self.assertTrue(any("[⚠️ 疑似虚构]" in str(call) for call in printed.call_args_list))

    async def test_tool_result_audio_does_not_cut_off_earlier_response_audio(self):
        for response_id, item_id, pcm in (("first", "preamble", b"\x00\x40" * 240),
                                           ("second", "result", b"\x00\x20" * 240)):
            self.receiver.response_pending = True
            await self.emit("response.created", response={"id": response_id})
            await self.emit("response.audio.delta", response_id=response_id, item_id=item_id,
                            delta=base64.b64encode(pcm).decode())
            await self.emit("response.done", response={"id": response_id, "status": "completed"})
        output = np.zeros((480, 1), dtype=np.float32)
        self.player.callback(output, 480, None, None)
        np.testing.assert_allclose(output[:240], .5)
        np.testing.assert_allclose(output[240:], .25)

    async def test_noise_resumes_and_deletes_without_cancelling(self):
        target = SpeechCandidate("noise", 0, committed=True)
        self.receiver.candidate = target
        self.receiver.current_response = "old"
        self.player.append(b"\x00\x40" * 2400, "answer", 0)
        self.player.duck(True)
        self.activity.voiced_ms = 400
        self.receiver.pause_candidate(target)
        self.assertTrue(self.player.paused)
        await self.emit("conversation.item.input_audio_transcription.completed",
                        item_id="noise", transcript="咳")
        self.assertFalse(self.player.paused)
        self.assertEqual(self.player.gain, 1)
        self.assertTrue(self.player.buffers)
        self.assertEqual([event["type"] for event in self.sent()], ["conversation.item.delete"])

    async def test_meaningful_cancels_truncates_then_waits_for_old_done(self):
        self.receiver.candidate = SpeechCandidate("question", 0, committed=True, stopped=True)
        self.receiver.current_response = "old"
        self.player.append(b"\x00\x40" * 2400, "answer", 0)
        self.player.callback(np.zeros((480, 1), dtype=np.float32), 480, None, None)
        await self.emit("conversation.item.input_audio_transcription.completed",
                        item_id="question", transcript="等一下")
        events = self.sent()
        self.assertEqual([event["type"] for event in events],
                         ["response.cancel", "conversation.item.truncate"])
        self.assertNotIn("response_id", events[0])
        self.assertEqual(events[1]["audio_end_ms"], 20)
        await self.emit("response.audio.delta", response_id="old", item_id="answer",
                        delta=base64.b64encode(b"\x00\x40" * 240).decode())
        self.assertFalse(self.player.buffers)
        await self.emit("response.done", response={"id": "old", "status": "cancelled"})
        self.assertEqual(self.sent()[-1]["type"], "response.create")
        await self.receiver.maybe_respond()
        self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)

    async def test_transcript_before_commit_does_not_answer_early(self):
        self.receiver.candidate = SpeechCandidate("question", 0, stopped=True)
        await self.emit("conversation.item.input_audio_transcription.completed",
                        item_id="question", transcript="你好")
        self.assertEqual(self.sent(), [])
        await self.emit("input_audio_buffer.committed", item_id="question")
        self.assertEqual(self.sent()[-1]["type"], "response.create")
        await self.emit("response.created", response={"id": "new"})
        self.assertEqual(len(self.sent()), 2)

    async def test_tool_call_runs_in_background_and_preserves_call_id(self):
        worker = Mock(closed=False)
        entered, release = asyncio.Event(), asyncio.Event()

        async def call(name, arguments):
            entered.set()
            await release.wait()
            return CallToolResult(content=[TextContent(type="text", text="已完成：牛奶")])

        worker.call = AsyncMock(side_effect=call)
        tool = {"name": "list_items", "input_schema": {"type": "object"}}
        receiver = self.make_receiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker, [tool])
        event = {"type": "response.function_call_arguments.done", "response_id": "r1",
                 "name": "list_items", "call_id": "call_milk",
                 "arguments": '{"list_name":"购物清单","titles":["牛奶"]}'}
        async with asyncio.TaskGroup() as tasks:
            await receiver.handle(event, tasks)
            await asyncio.wait_for(entered.wait(), 1)
            # worker 尚未完成时，接收循环仍能处理其他事件。
            await receiver.handle({"type": "response.done", "response": {
                "id": "r1", "status": "completed", "output": [{
                    "type": "function_call", "name": event["name"], "call_id": event["call_id"],
                    "arguments": event["arguments"],
                }],
            }}, tasks)
            self.assertEqual(self.sent(), [])
            release.set()
        worker.call.assert_awaited_once_with("list_items", {"list_name": "购物清单", "titles": ["牛奶"]})
        events = self.sent()
        result = events[0]["item"]
        self.assertEqual(result["type"], "function_call_output")
        self.assertEqual(result["call_id"], "call_milk")
        self.assertEqual(json.loads(result["output"]), {"is_error": False, "text": "已完成：牛奶"})
        self.assertEqual(events[-1]["type"], "response.create")
        clock = events[-2]["item"]
        self.assertEqual(clock["role"], "system")
        self.assertIn("[当前时间：", clock["content"][0]["text"])
        self.assertIn("America/Los_Angeles", clock["content"][0]["text"])

    async def test_tool_errors_are_returned_without_executing_forbidden_calls(self):
        for name, arguments, message in (("delete_all", "{}", "白名单"),
                                         ("list_items", "[]", "JSON 对象"),
                                         ("list_items", "{", "JSONDecodeError")):
            worker = Mock(closed=False, call=AsyncMock())
            self.ws.send.reset_mock()
            receiver = self.make_receiver(self.ws, self.player, self.activity,
                                         self.receiver.costs, "session", worker,
                                         [{"name": "list_items", "input_schema": {}}])
            async with asyncio.TaskGroup() as tasks:
                await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                       "name": name, "call_id": "call", "arguments": arguments}, tasks)
            worker.call.assert_not_awaited()
            output = json.loads(self.sent()[0]["item"]["output"])
            self.assertTrue(output["is_error"])
            self.assertIn(message, output["text"])

    async def test_tool_mcp_error_is_sent_to_model(self):
        worker = Mock(closed=False, call=AsyncMock(return_value=CallToolResult(
            is_error=True, content=[TextContent(type="text", text="找不到清单")],
        )))
        receiver = self.make_receiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker,
                                     [{"name": "list_items", "input_schema": {}}])
        async with asyncio.TaskGroup() as tasks:
            await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                   "name": "list_items", "call_id": "call", "arguments": "{}"}, tasks)
        self.assertEqual(json.loads(self.sent()[0]["item"]["output"]),
                         {"is_error": True, "text": "找不到清单"})

    async def test_deferred_duplicate_is_cancelled_and_does_not_borrow_previous_success(self):
        worker = Mock(closed=False, call=AsyncMock(return_value=CallToolResult(
            content=[TextContent(type="text", text="已添加牛奶")],
        )))
        receiver = self.make_receiver(self.ws, self.player, self.activity,
                                      self.receiver.costs, "session", worker,
                                      [{"name": "add_items", "input_schema": {}}])
        for call_id in ("original", "duplicate"):
            async with asyncio.TaskGroup() as tasks:
                await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                       "name": "add_items", "call_id": call_id,
                                       "arguments": '{"items":["牛奶"],"list_name":"购物清单"}'}, tasks)
            if call_id == "original":
                receiver.commit_window.turn_finished()
                await asyncio.wait_for(receiver.commit_window.reports.get(), 1)
        worker.call.assert_awaited_once()
        outputs = [json.loads(event["item"]["output"]) for event in self.sent()]
        self.assertIn("尚未执行", outputs[0]["text"])
        self.assertIn("本次没有执行", outputs[1]["text"])
        self.assertIn("那是之前的那次", outputs[1]["text"])
        self.assertIn("先向用户确认", outputs[1]["text"])
        self.assertFalse(receiver.turn_successes.get("<no-user-item>"))
        self.assertNotIn("duplicate", receiver.turn_attempts.get("<no-user-item>", set()))
        rows = receiver.gateway.store._db.execute("SELECT status FROM operations ORDER BY rowid").fetchall()
        self.assertEqual([row["status"] for row in rows], ["succeeded", "cancelled"])

    async def test_policy_missing_even_loaded_tool_is_refused(self):
        self.receiver.tool_names.add("not-in-policy")
        async with asyncio.TaskGroup() as tasks:
            await self.receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                        "name": "not-in-policy", "call_id": "forbidden", "arguments": "{}"}, tasks)
        output = json.loads(self.sent()[0]["item"]["output"])
        self.assertTrue(output["is_error"])
        self.assertIn("白名单", output["text"])
        self.assertEqual(self.receiver.gateway.store._db.execute("SELECT COUNT(*) FROM operations").fetchone()[0], 0)

    async def test_ha_calls_route_to_http_instead_of_personal_worker(self):
        worker = Mock(closed=False, call=AsyncMock())
        http_client = Mock(is_closed=False)
        tools = [{"name": name, "input_schema": {}} for name in qwen.HA_DEVICE_TOOLS]
        receiver = self.make_receiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker, tools,
                                     ha_url="http://ha/api/mcp", http_client=http_client)
        result = CallToolResult(content=[TextContent(type="text", text="操作成功")])
        with patch.object(executor_module, "call_ha_tool", AsyncMock(return_value=result)) as call_ha:
            async with asyncio.TaskGroup() as tasks:
                for name in sorted(qwen.HA_DEVICE_TOOLS):
                    await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                           "name": name, "call_id": name, "arguments": '{"name":"客厅灯"}'}, tasks)
            for name in qwen.HA_DEVICE_TOOLS:
                call_ha.assert_any_await(name, {"name": "客厅灯"}, url="http://ha/api/mcp", http_client=http_client, classify_not_sent=True)
            self.assertEqual(call_ha.await_count, 4)
        worker.call.assert_not_awaited()
        self.assertEqual({e["item"]["call_id"] for e in self.sent()}, qwen.HA_DEVICE_TOOLS)

    async def test_ha_error_and_undiscovered_tool_are_returned_to_model(self):
        tool = {"name": "intent__HassTurnOn", "input_schema": {}}
        for tools in ([tool], []):
            self.ws.send.reset_mock()
            receiver = self.make_receiver(self.ws, self.player, self.activity,
                                         self.receiver.costs, "session", tools=tools,
                                         ha_url="http://ha/api/mcp", http_client=Mock(is_closed=False))
            with patch.object(executor_module, "call_ha_tool", AsyncMock(side_effect=OSError("HA 正在重启"))) as call_ha:
                async with asyncio.TaskGroup() as tasks:
                    await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                           "name": tool["name"], "call_id": "ha-call", "arguments": "{}"}, tasks)
                if tools:
                    call_ha.assert_awaited_once()
                else:
                    call_ha.assert_not_awaited()
            output = self.sent()[0]["item"]
            self.assertEqual(output["call_id"], "ha-call")
            result = json.loads(output["output"])
            self.assertTrue(result["is_error"])
            self.assertIn("HA 正在重启" if tools else "白名单", result["text"])

    async def test_mixed_ha_and_personal_tools_continue_once(self):
        result = CallToolResult(content=[TextContent(type="text", text="完成")])
        worker = Mock(closed=False, call=AsyncMock(return_value=result))
        tools = [{"name": name, "input_schema": {}} for name in ("list_items", "intent__HassTurnOff")]
        receiver = self.make_receiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker, tools,
                                     ha_url="http://ha/api/mcp", http_client=Mock(is_closed=False))
        with patch.object(executor_module, "call_ha_tool", AsyncMock(return_value=result)) as call_ha:
            async with asyncio.TaskGroup() as tasks:
                for tool in tools:
                    await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                           "name": tool["name"], "call_id": tool["name"], "arguments": "{}"}, tasks)
                await receiver.handle({"type": "response.done", "response": {
                    "id": "r", "status": "completed",
                }}, tasks)
            call_ha.assert_awaited_once()
        worker.call.assert_awaited_once_with("list_items", {})
        self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)

    async def test_successful_device_actions_drain_silently_and_next_turn_still_works(self):
        names = ["intent__HassTurnOn", "intent__HassTurnOff", "light__HassLightSet"]
        result = CallToolResult(content=[TextContent(type="text", text=json.dumps({
            "response": {"response_type": "action_done", "data": {}},
        }))])
        for tools_finish_first in (True, False):
            self.ws.send.reset_mock()
            receiver = self.make_receiver(self.ws, self.player, self.activity,
                                         self.receiver.costs, "session",
                                         tools=[{"name": name, "input_schema": {}} for name in names],
                                         ha_url="http://ha/api/mcp", http_client=Mock(is_closed=False))
            with patch.object(executor_module, "call_ha_tool", AsyncMock(return_value=result)):
                async with asyncio.TaskGroup() as tasks:
                    for name in names:
                        await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                               "name": name, "call_id": name, "arguments": "{}"}, tasks)
                    if tools_finish_first:
                        await asyncio.sleep(0)
                    await receiver.handle({"type": "response.done", "response": {
                        "id": "r", "status": "completed",
                    }}, tasks)
            outputs = [event["item"] for event in self.sent() if event["type"] == "conversation.item.create"]
            self.assertEqual(len(outputs), 3)
            self.assertTrue(all(item["type"] == "function_call_output" for item in outputs))
            self.assertTrue(receiver.response_pending)
            self.assertTrue(receiver.silent_response_pending)
            self.assertEqual(self.sent()[-1]["type"], "response.create")
            await receiver.handle({"type": "response.created", "response": {"id": "silent"}}, None)
            await receiver.handle({"type": "response.audio.delta", "response_id": "silent", "item_id": "answer",
                                   "delta": base64.b64encode(b"\x00\x40" * 240).decode()}, None)
            self.assertFalse(self.player.buffers)
            await receiver.handle({"type": "response.done", "response": {
                "id": "silent", "status": "completed",
            }}, None)
            self.assertFalse(receiver.response_pending)
            # 静默结束只跳过本轮确认，不影响下一句指令。
            receiver.candidate = SpeechCandidate("next", 100, committed=True, meaningful=True)
            await receiver.maybe_respond()
            self.assertEqual(self.sent()[-1]["type"], "response.create")

    async def test_twelve_consecutive_silent_actions_complete_each_followup(self):
        name = "intent__HassTurnOff"
        result = CallToolResult(content=[], structured_content={"response_type": "action_done"})
        receiver = self.make_receiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", tools=[{"name": name, "input_schema": {}}],
                                     ha_url="http://ha/api/mcp", http_client=Mock(is_closed=False))
        with patch.object(executor_module, "call_ha_tool", AsyncMock(return_value=result)) as call_ha:
            for index in range(12):
                response_id = f"action-{index}"
                async with asyncio.TaskGroup() as tasks:
                    await receiver.handle({"type": "response.function_call_arguments.done", "response_id": response_id,
                                           "name": name, "call_id": f"call-{index}", "arguments": "{}"}, tasks)
                    await receiver.handle({"type": "response.done", "response": {
                        "id": response_id, "status": "completed",
                    }}, tasks)
                await receiver.handle({"type": "response.created", "response": {"id": f"silent-{index}"}}, None)
                await receiver.handle({"type": "response.done", "response": {
                    "id": f"silent-{index}", "status": "completed",
                }}, None)
                self.assertFalse(receiver.response_pending)
                self.assertIsNone(receiver.current_response)
            self.assertEqual(call_ha.await_count, 12)
        self.assertEqual(sum(event["type"] == "response.create" for event in self.sent()), 12)
        self.assertFalse(self.player.buffers)

    async def test_silent_followup_does_not_repeat_device_write(self):
        receiver = self.receiver
        receiver.response_pending = True
        receiver.silent_response_pending = True
        await self.emit("response.created", response={"id": "silent"})
        event = {"type": "response.function_call_arguments.done", "response_id": "silent",
                 "name": "intent__HassTurnOff", "call_id": "duplicate", "arguments": "{}"}
        with patch.object(executor_module, "call_ha_tool", AsyncMock()) as call_ha:
            async with asyncio.TaskGroup() as tasks:
                await receiver.handle(event, tasks)
            self.assertNotIn("duplicate", receiver.tool_calls)
            with self.assertRaises(qwen.SessionEnded):
                await self.emit("response.done", response={"id": "silent", "status": "completed", "output": [{
                    "type": "function_call", "name": event["name"], "call_id": event["call_id"], "arguments": "{}",
                }]})
            call_ha.assert_not_awaited()

    async def test_failed_uncertain_and_query_results_keep_model_response(self):
        cases = [
            ("intent__HassTurnOff", CallToolResult(is_error=True, content=[TextContent(type="text", text="失败")])),
            ("intent__HassTurnOff", CallToolResult(content=[TextContent(type="text", text="操作成功")])),
            ("intent__HassTurnOff", CallToolResult(content=[], structured_content={
                "response_type": "action_done", "data": {"failed": ["light.living_room"]},
            })),
            ("homeassistant__GetLiveContext", CallToolResult(content=[TextContent(type="text", text="客厅灯已关")])),
        ]
        for name, result in cases:
            self.ws.send.reset_mock()
            receiver = self.make_receiver(self.ws, self.player, self.activity,
                                         self.receiver.costs, "session",
                                         tools=[{"name": name, "input_schema": {}}],
                                         ha_url="http://ha/api/mcp", http_client=Mock(is_closed=False))
            with patch.object(executor_module, "call_ha_tool", AsyncMock(return_value=result)):
                async with asyncio.TaskGroup() as tasks:
                    await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                           "name": name, "call_id": "call", "arguments": "{}"}, tasks)
                    await receiver.handle({"type": "response.done", "response": {
                        "id": "r", "status": "completed",
                    }}, tasks)
            self.assertEqual(self.sent()[-1]["type"], "response.create")

    async def test_parallel_tools_create_one_continuation(self):
        worker = Mock(closed=False, call=AsyncMock(return_value=CallToolResult(
            content=[TextContent(type="text", text="完成")],
        )))
        receiver = self.make_receiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker,
                                     [{"name": "list_items", "input_schema": {}}])
        async with asyncio.TaskGroup() as tasks:
            for call_id in ("first", "second"):
                await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                       "name": "list_items", "call_id": call_id, "arguments": "{}"}, tasks)
            await receiver.handle({"type": "response.done", "response": {
                "id": "r", "status": "completed",
            }}, tasks)
        outputs = [e["item"] for e in self.sent() if e.get("item", {}).get("type") == "function_call_output"]
        self.assertEqual({item["call_id"] for item in outputs}, {"first", "second"})
        self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)

    async def test_worker_exception_is_returned_with_call_id(self):
        worker = Mock(closed=False, call=AsyncMock(side_effect=TimeoutError("服务超时")))
        receiver = self.make_receiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker,
                                     [{"name": "list_items", "input_schema": {}}])
        async with asyncio.TaskGroup() as tasks:
            await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                   "name": "list_items", "call_id": "failed", "arguments": "{}"}, tasks)
        result = self.sent()[0]["item"]
        self.assertEqual(result["call_id"], "failed")
        output = json.loads(result["output"])
        self.assertTrue(output["is_error"])
        self.assertIn("服务超时", output["text"])

    async def test_new_meaningful_turn_suppresses_old_tool_continuation(self):
        old, new = SpeechCandidate("old", 0), SpeechCandidate("new", 100)
        receiver = self.receiver
        receiver.candidate = new
        receiver.last_meaningful_candidate = new
        receiver.tool_candidates["old-response"] = old
        receiver.tool_pending["old-response"] = 0
        receiver.finished_responses["old-response"] = "completed"
        await receiver.maybe_respond()
        self.assertEqual(self.sent(), [])

    async def test_pending_answer_cancelled_after_new_meaningful_input(self):
        self.receiver.candidate = SpeechCandidate("first", 0, committed=True, meaningful=True)
        await self.receiver.maybe_respond()
        self.receiver.candidate = SpeechCandidate("second", 100, committed=True, stopped=True)
        await self.emit("conversation.item.input_audio_transcription.completed",
                        item_id="second", transcript="换个问题")
        await self.emit("response.created", response={"id": "first_answer"})
        self.assertEqual(self.sent()[-1]["type"], "response.cancel")

    async def test_unrequested_response_is_reported(self):
        with self.assertRaises(qwen.QwenProtocolError):
            await self.emit("response.created", response={"id": "automatic"})

    async def test_reconnect_reapplies_session_configuration(self):
        sockets = []
        tools = [{"name": "intent__HassTurnOn", "input_schema": {}}]
        http_client = Mock(is_closed=False)

        class Socket:
            def __init__(self):
                self.send = AsyncMock()
                self.recv = AsyncMock(side_effect=[
                    json.dumps({"type": "session.created", "session": {"id": str(len(sockets))}}),
                    json.dumps({"type": "session.updated", "session": qwen.session_config("Tina", tools, "客厅灯")}),
                ])

            async def __aenter__(self):
                sockets.append(self)
                return self

            async def __aexit__(self, *args):
                return False

        async def receive(receiver):
            self.assertIs(receiver.gateway, self.receiver.gateway)
            self.assertIs(receiver.commit_window, self.receiver.commit_window)
            self.assertIs(receiver.coordinator, self.receiver.coordinator)
            self.assertIn("intent__HassTurnOn", receiver.tool_names)
            if len(sockets) == 1:
                qwen.raise_api_error({"message": "Turn queue overflow: capacity=8"})
            second_session.set()
            await asyncio.Future()

        second_session = asyncio.Event()
        original_sleep = asyncio.sleep

        async def quick_retry(seconds):
            await original_sleep(0 if seconds == 1 else seconds)

        with patch.object(qwen, "connect", side_effect=lambda *a, **kw: Socket()), \
                patch.object(qwen, "send_microphone", side_effect=lambda *a: asyncio.Future()), \
                patch.object(qwen.QwenReceiver, "run", receive), \
                patch.object(qwen.asyncio, "sleep", quick_retry):
            task = asyncio.create_task(qwen.reconnect_loop(
                "wss://example", "test", self.player, self.receiver.costs, "Tina", "singapore",
                tools=tools, assist_prompt="客厅灯", gateway=self.receiver.gateway,
                commit_window=self.receiver.commit_window, coordinator=self.receiver.coordinator))
            await asyncio.wait_for(second_session.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(sockets), 2)
        for socket in sockets:
            event = json.loads(socket.send.call_args.args[0])
            self.assertEqual(event["type"], "session.update")
            self.assertEqual(event["session"]["tools"][0]["function"]["name"], "intent__HassTurnOn")
            self.assertIn("客厅灯", event["session"]["instructions"])

    async def test_main_loads_both_catalogs_and_closes_clients(self):
        personal = {"name": "list_items", "input_schema": {}}
        ha = {"name": "intent__HassTurnOn", "input_schema": {}}
        prompt = [{"type": "text", "text": "基础规则\nHome Assistant Assist prompt：\n客厅灯"}]
        db_dir = self.enterContext(TemporaryDirectory())
        db_path = Path(db_dir) / "operations.db"
        previous = OperationStore(db_path)
        interrupted = previous.create("old-turn", "create_reminder", {})
        previous.transition(interrupted.id, "pending", "approved")
        previous.transition(interrupted.id, "approved", "executing")
        previous.close()
        for configured, available in ((True, True), (True, False), (False, False)):
            worker = Mock(run=AsyncMock())
            environment = {"DASHSCOPE_API_KEY": "test", "QWEN_WORKSPACE_ID": "test",
                           "REMINDER_TIMEZONE": "Asia/Shanghai"}
            if configured:
                environment.update(HA_URL="http://ha", HA_TOKEN="test")
            catalog = ([ha], prompt) if available else None
            with patch.object(qwen, "OPERATIONS_DB", db_path), \
                    patch("builtins.print") as printed, \
                    patch.dict(qwen.os.environ, environment, clear=True), \
                    patch.object(qwen, "load_dotenv"), \
                    patch.object(qwen, "PersonalWorker", return_value=worker), \
                    patch.object(qwen, "stop_worker", AsyncMock()) as stop, \
                    patch.object(qwen, "load_personal_catalog", AsyncMock(return_value=[personal])), \
                    patch.object(qwen, "load_ha_catalog", AsyncMock(return_value=catalog)) as load_ha, \
                    patch.object(qwen.StreamingPlayer, "run", AsyncMock()), \
                    patch.object(qwen, "reconnect_loop", AsyncMock(side_effect=qwen.QwenProtocolError("test stop"))) as reconnect:
                with self.assertRaises(ExceptionGroup):
                    await qwen.main()
                if configured:
                    load_ha.assert_awaited_once()
                else:
                    load_ha.assert_not_awaited()
                args, kwargs = reconnect.await_args
                self.assertEqual([t["name"] for t in args[7]],
                                 ["list_items", "intent__HassTurnOn"] if available else ["list_items"])
                self.assertEqual(kwargs["assist_prompt"], "客厅灯" if available else "")
                self.assertEqual(kwargs["coordinator"]._timezone_name, "Asia/Shanghai")
                self.assertIs(kwargs["coordinator"].gateway, kwargs["gateway"])
                with self.assertRaises(sqlite3.ProgrammingError):
                    kwargs["gateway"].store.get("closed")
                stop.assert_awaited_once()
                checker = OperationStore(db_path)
                try:
                    self.assertEqual(checker.get(interrupted.id).status, "unknown")
                finally:
                    checker.close()
                if available:
                    self.assertTrue(any("[恢复]" in str(call) and interrupted.id in str(call)
                                        for call in printed.call_args_list))


class CostAndProtocolTests(unittest.TestCase):
    def test_simple_controls_are_local_and_verified_in_session_configuration(self):
        config = qwen.session_config("Tina", [])
        names = {tool["function"]["name"] for tool in config["tools"]}
        self.assertEqual(names, set(qwen.CONTROL_TOOLS))
        self.assertTrue(all(qwen.get_policy(name) is None for name in names))
        qwen.verify_session(config, [])
        with self.assertRaises(qwen.QwenProtocolError):
            qwen.verify_session({**config, "tools": config["tools"][:-1]}, [])
        self.assertEqual(names, {"cancel_pending", "undo_recent", "delete_item"})
        self.assertNotIn("finish_collection", config["instructions"])
        self.assertNotIn("confirm_plan", names)

    def test_completion_phrases_exclude_negation_and_quotes(self):
        for text in ("已经设好了", "已经加了", "已添加牛奶", "设置好了", "标记好了", "提醒已经帮你设置好了"):
            self.assertTrue(qwen.claims_completion(text), text)
        for text in ("还没有设置好", "无法确认已经设好了", "不能说已经加了", "请不要说“已经设好了”", "我查一下", "购物清单有牛奶"):
            self.assertFalse(qwen.claims_completion(text), text)

    def test_ha_tools_prompt_and_service_specific_fallback(self):
        tools = [{"name": "intent__HassTurnOn", "description": "开灯", "input_schema": {}}]
        config = qwen.session_config("Tina", tools, "客厅灯属于客厅")
        self.assertEqual(config["tools"][0]["function"]["name"], "intent__HassTurnOn")
        self.assertIn("客厅灯属于客厅", config["instructions"])
        self.assertIn("个人服务暂时不可用", config["instructions"])
        self.assertNotIn("家居设备暂时不可用", config["instructions"])
        qwen.verify_session(config, tools)
        with self.assertRaises(qwen.QwenProtocolError):
            qwen.verify_session({**config, "tools": []}, tools)
        personal_only = qwen.session_config("Tina", [{"name": "list_items", "input_schema": {}}])
        self.assertIn("家居设备暂时不可用", personal_only["instructions"])
        self.assertNotIn("个人服务暂时不可用", personal_only["instructions"])

    def test_tools_use_qwen_schema_and_are_checked_on_session_update(self):
        tools = [{"name": "list_items", "description": "列出条目", "input_schema": {
            "type": "object", "properties": {"list_name": {"type": "string"}},
        }}]
        config = qwen.session_config("Tina", tools)
        self.assertEqual(config["tools"][0], {"type": "function", "function": {
            "name": "list_items", "description": "列出条目", "parameters": tools[0]["input_schema"],
        }})
        qwen.verify_session(config, tools)
        with self.assertRaises(qwen.QwenProtocolError):
            qwen.verify_session({**config, "tools": []}, tools)
        self.assertNotIn("[当前时间：", config["instructions"])

    def test_price_detail_and_deduplication(self):
        response = {"id": "reply", "usage": {
            "input_tokens_details": {"text_tokens": 960, "audio_tokens": 168},
            "output_tokens_details": {"text_tokens": 16, "audio_tokens": 60},
        }}
        costs = qwen.CostTracker("singapore")
        self.assertEqual(costs.record("s1", response), Decimal("0.00050044"))
        self.assertIsNone(costs.record("s1", response))
        costs.record("s2", response)
        self.assertEqual(costs.total, Decimal("0.00100088"))
        self.assertIsNone(costs.record("s2", {"id": "missing"}))
        self.assertEqual(costs.unpriced, 1)

    def test_session_validation_and_error_classification(self):
        qwen.verify_session(qwen.session_config("Tina"))
        with self.assertRaises(qwen.QwenProtocolError):
            qwen.verify_session({"turn_detection": {"type": "semantic_vad"}})
        with self.assertRaises(qwen.SessionEnded):
            qwen.raise_api_error({"code": "session_expired"})
        with self.assertRaises(qwen.SessionEnded) as overflow:
            qwen.raise_api_error({"message": "Turn queue overflow: capacity=8"})
        self.assertTrue(qwen.reconnectable(overflow.exception))
        with self.assertRaises(qwen.QwenProtocolError):
            qwen.raise_api_error({"code": "invalid_value", "message": "invalid session parameter"})
        self.assertTrue(qwen.reconnectable(ExceptionGroup("network", [OSError("offline")])) )
        self.assertFalse(qwen.reconnectable(ExceptionGroup("bad config", [qwen.QwenProtocolError("bad")])) )


if __name__ == "__main__":
    unittest.main()
