import asyncio
import base64
import json
import os
from types import SimpleNamespace as NS
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from voice.upload_gate import UploadGate, make_upload_gate
from voice.qwen_realtime import send_microphone

SILENCE = bytes(640)
VOICE = bytes([1]) * 640


def gate(**kwargs):
    kwargs.setdefault('onset_ms', 20)  # Explicit original threshold for buffer/tail tests.
    detector = NS(is_speech=lambda frame, rate: frame == VOICE)
    return UploadGate(detector=detector, **kwargs)


class GateTests(TestCase):
    def test_idle_does_not_upload_and_buffer_is_bounded(self):
        g = gate()
        for _ in range(10000):
            self.assertIsNone(g.feed(SILENCE))
        self.assertEqual(len(g.before), 21)
        self.assertEqual(g.forwarded_bytes, 0)

    def test_first_voiced_frame_includes_pre_roll_exactly_once(self):
        g = gate()
        for _ in range(30):
            g.feed(SILENCE)
        self.assertEqual(g.feed(VOICE), SILENCE * 20 + VOICE)
        self.assertEqual(g.feed(VOICE), VOICE)

    def test_legacy_20ms_onset_remains_available(self):
        g = gate()
        self.assertEqual(g.feed(VOICE), VOICE)
        for _ in range(49):
            self.assertEqual(g.feed(SILENCE), SILENCE)
        self.assertTrue(g.open)
        self.assertEqual(g.feed(SILENCE), SILENCE)
        self.assertFalse(g.open)
        self.assertIsNone(g.feed(SILENCE))

    def test_pause_below_tail_stays_open_and_voice_restarts_tail(self):
        g = gate()
        g.feed(VOICE)
        for _ in range(40):
            self.assertEqual(g.feed(SILENCE), SILENCE)
        self.assertEqual(g.feed(VOICE), VOICE)
        for _ in range(49):
            g.feed(SILENCE)
        self.assertTrue(g.open)
        g.feed(SILENCE)
        self.assertFalse(g.open)

    def test_next_utterance_does_not_replay_previous_tail(self):
        g = gate()
        g.feed(VOICE)
        for _ in range(50):
            g.feed(SILENCE)
        for _ in range(3):
            g.feed(SILENCE)
        self.assertEqual(g.feed(VOICE), SILENCE * 3 + VOICE)

    def test_reset_drops_old_audio(self):
        g = gate()
        for _ in range(20):
            g.feed(SILENCE)
        g.reset()
        self.assertEqual(g.feed(VOICE), VOICE)
        self.assertEqual(g.captured_bytes, 640)

    def test_default_confirmation_keeps_pre_roll_and_initial_syllable(self):
        g = gate(onset_ms=80)
        for _ in range(30):
            self.assertIsNone(g.feed(SILENCE))
        for _ in range(3):
            self.assertIsNone(g.feed(VOICE))
        self.assertEqual(g.feed(VOICE), SILENCE * 20 + VOICE * 4)
        self.assertTrue(g.open)

    def test_isolated_loud_spikes_do_not_trigger_upload(self):
        g = gate(onset_ms=80)
        for _ in range(50):
            for _ in range(3):
                self.assertIsNone(g.feed(VOICE))
            self.assertIsNone(g.feed(SILENCE))
        self.assertEqual(g.forwarded_bytes, 0)

    def test_low_volume_vad_false_positive_never_opens(self):
        g = UploadGate(detector=NS(is_speech=lambda *_: True))
        quiet_noise = np.full(320, 30, dtype='<i2').tobytes()
        for _ in range(1000):
            self.assertIsNone(g.feed(quiet_noise))
        self.assertFalse(g.open)
        self.assertEqual(g.forwarded_bytes, 0)

    def test_below_floor_frames_close_stream_even_if_raw_vad_remains_true(self):
        g = UploadGate(detector=NS(is_speech=lambda *_: True))
        for _ in range(4):
            g.feed(VOICE)
        self.assertTrue(g.open)
        quiet_noise = np.full(320, 30, dtype='<i2').tobytes()
        for _ in range(50):
            g.feed(quiet_noise)
        self.assertFalse(g.open)
        self.assertIsNone(g.feed(quiet_noise))

    def test_invalid_frames_and_settings(self):
        for values in ({'pre_ms': 401}, {'tail_ms': 1010}, {'mode': 4}, {'onset_ms': 0}, {'min_rms': float('nan')}):
            with self.assertRaises(ValueError):
                gate(**values)
        with self.assertRaises(ValueError):
            gate().feed(bytes(1280))

    def test_rollback_and_tail_must_cover_server_silence(self):
        with patch.dict(os.environ, {'QWEN_ALPHA_LOCAL_VAD': '0'}, clear=True):
            self.assertIsNone(make_upload_gate())
        with patch.dict(os.environ, {'QWEN_ALPHA_LOCAL_VAD': '1', 'QWEN_SILENCE_MS': '1300'}, clear=True):
            self.assertEqual(make_upload_gate().tail_ms, 1500)
        with patch.dict(os.environ, {'QWEN_ALPHA_LOCAL_VAD': '1', 'QWEN_SILENCE_MS': '1300',
                                     'QWEN_ALPHA_VAD_TAIL_MS': '1000'}, clear=True):
            with self.assertRaises(ValueError):
                make_upload_gate()
        with patch.dict(os.environ, {'QWEN_ALPHA_LOCAL_VAD': '1', 'QWEN_VAD': 'semantic_vad'}, clear=True):
            with self.assertRaises(ValueError):
                make_upload_gate()


class SendIntegrationTests(IsolatedAsyncioTestCase):
    async def run_audio(self, frames, g, expected_calls):
        finished = asyncio.Event()
        closed = asyncio.Event()
        ws = NS(send=AsyncMock())
        sent = Mock()
        async def send(message):
            if ws.send.await_count >= expected_calls:
                finished.set()
        ws.send.side_effect = send
        def factory(**kwargs):
            class Stream:
                active = True
                def __enter__(self):
                    for frame in frames:
                        kwargs['callback'](np.frombuffer(frame, dtype='<i2').reshape(320, 1), 320, None, None)
                    return self
                def __exit__(self, *args):
                    closed.set()
            return Stream()
        with patch('voice.qwen_realtime.sd.InputStream', side_effect=factory):
            task = asyncio.create_task(send_microphone(ws, Mock(), on_sent=sent, upload_gate=g))
            try:
                if expected_calls:
                    await asyncio.wait_for(finished.wait(), 1)
                    await asyncio.sleep(.02)  # 等发送器进入空队列等待，避免 wait_for 完成/取消竞态。
                else:
                    await asyncio.sleep(.03)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, .3)
        self.assertTrue(closed.is_set())
        return ws, sent

    async def test_silence_neither_sent_nor_billed_but_microphone_is_consumed(self):
        ws, sent = await self.run_audio([SILENCE] * 50, gate(), 0)
        ws.send.assert_not_awaited()
        sent.assert_not_called()

    async def test_only_uploaded_bytes_reach_meter(self):
        ws, sent = await self.run_audio([SILENCE]*20 + [VOICE]*5 + [SILENCE]*20,
                                        gate(pre_ms=100, tail_ms=200, onset_ms=80), 12)
        self.assertEqual(ws.send.await_count, 12)
        payload = b''.join(base64.b64decode(json.loads(call.args[0])['audio']) for call in ws.send.await_args_list)
        self.assertEqual(payload, SILENCE*5 + VOICE*5 + SILENCE*10)
        self.assertEqual(b''.join(call.args[0] for call in sent.call_args_list), payload)

    async def test_disabled_gate_retains_continuous_upload(self):
        ws, sent = await self.run_audio([SILENCE]*10, None, 10)
        self.assertEqual(ws.send.await_count, 10)
        self.assertEqual(sent.call_count, 10)
