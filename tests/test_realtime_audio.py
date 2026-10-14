"""Offline regression checks: no microphone, network, or paid model calls."""

import asyncio
import base64
import json
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from voice.audio import SpeechActivity, SpeechCandidate, StreamingPlayer, is_meaningful
import voice.qwen_realtime as qwen


class AudioTests(unittest.TestCase):
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


class QwenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ws = Mock()
        self.ws.send = AsyncMock()
        self.player = StreamingPlayer()
        self.player.ready = True
        self.activity = SpeechActivity(16000)
        self.receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                          qwen.CostTracker("singapore"), "session")

    def sent(self):
        return [json.loads(call.args[0]) for call in self.ws.send.call_args_list]

    async def emit(self, kind, **fields):
        await self.receiver.handle({"type": kind, **fields}, None)

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
        self.assertEqual(self.sent()[0]["type"], "response.create")
        await self.emit("response.created", response={"id": "new"})
        self.assertEqual(len(self.sent()), 1)

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

        class Socket:
            def __init__(self):
                self.send = AsyncMock()
                self.recv = AsyncMock(side_effect=[
                    json.dumps({"type": "session.created", "session": {"id": str(len(sockets))}}),
                    json.dumps({"type": "session.updated", "session": qwen.session_config("Tina")}),
                ])

            async def __aenter__(self):
                sockets.append(self)
                return self

            async def __aexit__(self, *args):
                return False

        async def receive(receiver):
            if len(sockets) == 1:
                raise qwen.SessionEnded("session expired")
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
                "wss://example", "test", self.player, self.receiver.costs, "Tina", "singapore"))
            await asyncio.wait_for(second_session.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(sockets), 2)
        for socket in sockets:
            self.assertEqual(json.loads(socket.send.call_args.args[0])["type"], "session.update")


class CostAndProtocolTests(unittest.TestCase):
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
        self.assertTrue(qwen.reconnectable(ExceptionGroup("network", [OSError("offline")])) )
        self.assertFalse(qwen.reconnectable(ExceptionGroup("bad config", [qwen.QwenProtocolError("bad")])) )


if __name__ == "__main__":
    unittest.main()
