"""共享播放器和Omni后备传输的离线回归测试。"""

import asyncio
import os
import time
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
import numpy as np
from voice.audio import SpeechActivity, StreamingPlayer, is_meaningful
import voice.omni_transport as qwen


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


class MicrophoneCongestionTests(unittest.IsolatedAsyncioTestCase):
    def stream_factory(self, blocks, closed):
        def factory(**kwargs):
            class Stream:
                active = True

                def __enter__(self):
                    for _ in range(blocks):
                        kwargs["callback"](np.zeros((320, 1), dtype="int16"), 320, None, None)
                    return self

                def __exit__(self, *args):
                    closed.set()
            return Stream()
        return factory

    async def test_queue_overflow_discards_old_audio_and_is_reconnectable(self):
        closed = asyncio.Event()
        ws = Mock(send=AsyncMock())
        with patch.object(qwen.sd, "InputStream", side_effect=self.stream_factory(101, closed)):
            with self.assertRaises(qwen.SessionEnded) as raised:
                await asyncio.wait_for(qwen.send_microphone(ws, Mock()), 1)
        self.assertTrue(qwen.reconnectable(raised.exception))
        self.assertTrue(closed.is_set())
        self.assertIn("收到 101 块、已发送 0 块", str(raised.exception))
        self.assertIn("回调调度延迟", str(raised.exception))
        ws.send.assert_not_awaited()  # 不把积压的两秒音频依次补发。

    async def test_stalled_websocket_send_has_bounded_wait_and_closes_microphone(self):
        closed = asyncio.Event()
        async def stalled(*args):
            await asyncio.Future()
        ws = Mock(send=AsyncMock(side_effect=stalled))
        with patch.object(qwen.sd, "InputStream", side_effect=self.stream_factory(1, closed)), \
                patch.object(qwen, "AUDIO_SEND_TIMEOUT_S", .01):
            with self.assertRaises(qwen.SessionEnded) as raised:
                await asyncio.wait_for(qwen.send_microphone(ws, Mock()), 1)
        self.assertIn("音频发送超时", str(raised.exception))
        self.assertTrue(closed.is_set())
        self.assertTrue(qwen.reconnectable(ExceptionGroup("microphone", [raised.exception])))


class CostAndProtocolTests(unittest.TestCase):
    def test_server_vad_and_silence_are_configurable_and_verified(self):
        with patch.dict(os.environ, {"QWEN_VAD": "server_vad", "QWEN_SILENCE_MS": "800"}):
            config = qwen.session_config("Tina")
            detection = config["turn_detection"]
            self.assertEqual(detection, {"type": "server_vad", "silence_duration_ms": 800,
                                         "create_response": False, "interrupt_response": False})
            qwen.verify_session(config)
            with self.assertRaises(qwen.QwenProtocolError):
                qwen.verify_session({**config, "turn_detection": {**detection, "silence_duration_ms": 1000}})
        with patch.dict(os.environ, {"QWEN_VAD": "semantic_vad"}):
            self.assertEqual(qwen.session_config("Tina")["turn_detection"]["type"], "semantic_vad")
            self.assertNotIn("silence_duration_ms", qwen.session_config("Tina")["turn_detection"])

    def test_invalid_vad_settings_fail_before_connecting(self):
        for values in ({"QWEN_VAD": "invalid"}, {"QWEN_VAD": "server_vad", "QWEN_SILENCE_MS": "bad"},
                       {"QWEN_VAD": "server_vad", "QWEN_SILENCE_MS": "199"},
                       {"QWEN_VAD": "server_vad", "QWEN_SILENCE_MS": "6001"}):
            with self.subTest(values=values), patch.dict(os.environ, values):
                with self.assertRaises(ValueError):
                    qwen.session_config("Tina")






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
