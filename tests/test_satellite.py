"""ESPHome 语音卫星（J9）：唤醒、推流、一轮的阶段和结束时机、打断、设备播放。不连真实设备。"""

import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from aioesphomeapi import VoiceAssistantEventType as Event
import numpy as np

from voice import satellite as module
from voice.satellite import FLUSH, Resampler, SatelliteLink, SatellitePlayer

WITH_SPEAKER, WITHOUT_SPEAKER = 7, 5  # VOICE_ASSISTANT | SPEAKER | API_AUDIO；旧固件没有 SPEAKER


class FakeClient:
    instances = []
    flags = WITH_SPEAKER

    def __init__(self, host, port, password, *, client_info, noise_psk):
        self.host, self.noise_psk = host, noise_psk
        self.events, self.audio, self.handlers, self.on_stop = [], [], None, None
        self.disconnect = AsyncMock()
        FakeClient.instances.append(self)

    async def connect(self, on_stop, login):
        self.on_stop = on_stop

    async def device_info(self):
        return NS(name='alpha-xvf', esphome_version='2026.9.1', voice_assistant_feature_flags=self.flags)

    def subscribe_voice_assistant(self, **handlers):
        self.handlers = handlers
        return Mock()

    def send_voice_assistant_event(self, kind, data):
        self.events.append((kind, data))

    def subscribe_service_calls(self, callback):
        self.service_callback = callback

    def send_voice_assistant_audio(self, data):
        self.audio.append(data)

    def kinds(self):
        return [kind for kind, _ in self.events]


class FakeAdapter:
    def __init__(self):
        self.fed = []
        self.segmenter = NS(active=None)
        self.receiver = NS(active=False, current_report=None, interrupt=AsyncMock())

    async def process(self, pcm):
        self.fed.append(pcm)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class LinkTestCase(unittest.IsolatedAsyncioTestCase):
    flags = WITH_SPEAKER

    async def asyncSetUp(self):
        FakeClient.instances.clear()
        FakeClient.flags = self.flags
        self.clock = Clock()
        self.link = SatelliteLink('192.168.1.115', noise_psk='key', listen_s=8, client_factory=FakeClient,
                                  clock=self.clock)
        self.adapter = FakeAdapter()
        self.tasks = [asyncio.create_task(self.link.run()), asyncio.create_task(self.link.feed(self.adapter))]
        await self.settle()
        self.client = FakeClient.instances[0]

    async def asyncTearDown(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def wake(self):
        return await self.client.handlers['handle_start']('conv', 0, None, 'Hey Jarvis')

    async def settle(self):
        await asyncio.sleep(0.05)


class ListeningTests(LinkTestCase):
    async def test_wake_streams_audio_over_api_into_the_segmenter(self):
        self.assertEqual(self.client.noise_psk, 'key')
        self.assertTrue(self.link.speaker)
        self.assertEqual(await self.wake(), 0)  # 0：音频走 API 连接
        self.assertEqual(self.client.kinds(), [Event.VOICE_ASSISTANT_RUN_START, Event.VOICE_ASSISTANT_STT_START])
        await self.client.handlers['handle_audio'](b'\x01\x00' * 256, None)
        await self.client.handlers['handle_audio'](b'\x02\x00' * 256, None)
        await self.settle()
        self.assertEqual(self.adapter.fed, [b'\x01\x00' * 256, b'\x02\x00' * 256])

    async def test_audio_is_amplified_by_the_volume_multiplier_the_device_asks_for(self):
        await self.client.handlers['handle_start']('conv', 0, NS(volume_multiplier=16.0), 'Hey Jarvis')
        await self.client.handlers['handle_audio'](np.array([100, -100, 3000], '<i2').tobytes(), None)
        await self.settle()
        self.assertEqual(np.frombuffer(self.adapter.fed[0], '<i2').tolist(), [1600, -1600, 32767])  # 削顶不溢出
        await self.client.handlers['handle_start']('conv', 0, NS(volume_multiplier=1.0), 'Hey Jarvis')
        await self.client.handlers['handle_audio'](np.array([100], '<i2').tobytes(), None)
        await self.settle()
        self.assertEqual(np.frombuffer(self.adapter.fed[-1], '<i2').tolist(), [100])

    async def test_audio_outside_listening_is_dropped(self):
        await self.client.handlers['handle_audio'](b'\x01\x00' * 256, None)
        await self.wake()
        self.link.end_listening('submitted')
        await self.client.handlers['handle_audio'](b'\x01\x00' * 256, None)
        await self.settle()
        self.assertEqual(self.adapter.fed, [FLUSH])

    async def test_no_voice_for_listen_seconds_ends_the_run(self):
        await self.wake()
        self.clock.now += 7
        await self.client.handlers['handle_audio'](b'\x00\x00', None)
        await self.settle()
        self.assertNotIn(Event.VOICE_ASSISTANT_RUN_END, self.client.kinds())
        # 说话中刷新计时：说完 8 秒内都不结束。
        self.adapter.segmenter.active = object()
        await self.client.handlers['handle_audio'](b'\x00\x00', None)
        await self.settle()
        self.adapter.segmenter.active = None
        self.clock.now += 7.9
        await self.client.handlers['handle_audio'](b'\x00\x00', None)
        await self.settle()
        self.assertNotIn(Event.VOICE_ASSISTANT_RUN_END, self.client.kinds())
        self.clock.now += 0.2
        await asyncio.sleep(0.6)  # 没有新音频也要按时结束
        self.assertEqual(self.client.kinds()[-1], Event.VOICE_ASSISTANT_RUN_END)
        self.assertEqual(self.link.phase, 'idle')

    async def test_wake_during_playback_interrupts_it(self):
        self.adapter.receiver.active = True
        await self.wake()
        self.adapter.receiver.interrupt.assert_awaited_once()
        self.adapter.receiver.active = False
        await self.wake()
        self.adapter.receiver.interrupt.assert_awaited_once()

    async def test_disconnect_ends_the_run_and_reconnects(self):
        await self.wake()
        await self.client.on_stop(False)
        await self.settle()
        self.assertEqual(self.link.phase, 'idle')
        self.client.disconnect.assert_awaited_once()
        self.assertNotIn(Event.VOICE_ASSISTANT_RUN_END, self.client.kinds())  # 连接已断，不再发


class ResponseTests(LinkTestCase):
    async def test_submission_stops_the_mic_but_keeps_the_run_until_the_answer_is_played(self):
        await self.wake()
        self.link.end_listening('submitted')
        await self.settle()
        self.assertEqual(self.client.kinds()[-1], Event.VOICE_ASSISTANT_STT_VAD_END)
        self.assertEqual((self.link.phase, self.adapter.fed), ('awaiting', [FLUSH]))
        self.assertTrue(self.link.start_response())
        self.assertTrue(self.link.start_response())  # 第二段（例如“我查一下”之后的回答）不重复开流
        self.assertEqual(self.client.events[-3:], [
            (Event.VOICE_ASSISTANT_TTS_START, {'text': 'Alpha'}), (Event.VOICE_ASSISTANT_TTS_END, {'url': 'api'}),
            (Event.VOICE_ASSISTANT_TTS_STREAM_START, None)])
        self.link.send_audio(b'pcm')
        self.link.end_response()
        self.assertEqual(self.client.audio, [b'pcm'])
        self.assertEqual(self.client.kinds()[-2:], [Event.VOICE_ASSISTANT_TTS_STREAM_END,
                                                    Event.VOICE_ASSISTANT_RUN_END])
        self.assertEqual(self.link.phase, 'idle')
        self.link.end_response()
        self.assertEqual(self.client.kinds().count(Event.VOICE_ASSISTANT_RUN_END), 1)

    async def test_no_response_for_a_long_time_ends_the_run(self):
        await self.wake()
        self.link.end_listening('submitted')
        self.clock.now += module.RESPONSE_TIMEOUT_S
        await asyncio.sleep(0.6)
        self.assertEqual(self.client.kinds()[-1], Event.VOICE_ASSISTANT_RUN_END)

    async def test_device_stop_during_playback_stops_it_without_ending_the_next_run(self):
        await self.wake()
        self.link.end_listening('submitted')
        self.link.start_response()
        self.adapter.receiver.active = True
        await self.client.handlers['handle_stop'](False)
        self.adapter.receiver.interrupt.assert_awaited_once()
        self.assertEqual(self.link.phase, 'idle')
        self.assertNotIn(Event.VOICE_ASSISTANT_RUN_END, self.client.kinds())

    async def barge_in_event(self, service='esphome.alpha_barge_in', is_event=True):
        self.client.service_callback(NS(service=service, is_event=is_event, data={}))
        await self.settle()

    async def test_wake_during_playback_event_stops_playback_and_ends_the_run(self):
        await self.wake()
        self.link.end_listening('submitted')
        self.link.start_response()
        self.adapter.receiver.active = True
        await self.barge_in_event()
        self.adapter.receiver.interrupt.assert_awaited_once()
        self.assertEqual(self.client.kinds()[-2:], [Event.VOICE_ASSISTANT_TTS_STREAM_END,
                                                    Event.VOICE_ASSISTANT_RUN_END])
        self.assertEqual(self.link.phase, 'idle')

    async def test_wake_again_while_listening_restarts_the_run(self):
        await self.wake()
        await self.barge_in_event()
        self.assertEqual(self.client.kinds()[-1], Event.VOICE_ASSISTANT_RUN_END)
        self.assertEqual(self.link.phase, 'idle')

    async def test_other_service_calls_and_idle_barge_in_are_ignored(self):
        await self.barge_in_event()  # 空闲时
        await self.wake()
        await self.barge_in_event(service='esphome.something_else')
        await self.barge_in_event(is_event=False)
        self.assertEqual(self.link.phase, 'listening')
        self.assertNotIn(Event.VOICE_ASSISTANT_RUN_END, self.client.kinds())

    async def test_cannot_play_outside_a_run(self):
        self.assertFalse(self.link.start_response())
        await self.wake()
        self.assertFalse(self.link.start_response())  # 还在收音


class DumpTests(LinkTestCase):
    async def asyncSetUp(self):
        import tempfile
        self.dump = tempfile.TemporaryDirectory()
        self.addCleanup(self.dump.cleanup)
        await super().asyncSetUp()
        self.link.dump_dir = __import__('pathlib').Path(self.dump.name)

    async def test_each_run_is_saved_as_raw_audio_before_gain(self):
        import wave
        await self.client.handlers['handle_start']('conv', 0, NS(volume_multiplier=16.0), 'Hey Jarvis')
        await self.client.handlers['handle_audio'](np.array([100, -100], '<i2').tobytes(), None)
        self.link.end_listening('submitted')
        files = list(__import__('pathlib').Path(self.dump.name).glob('*-submitted.wav'))
        self.assertEqual(len(files), 1)
        with wave.open(str(files[0])) as saved:
            self.assertEqual(np.frombuffer(saved.readframes(10), '<i2').tolist(), [100, -100])


class WithoutSpeakerTests(LinkTestCase):
    flags = WITHOUT_SPEAKER

    async def test_submission_ends_the_run_and_answers_play_on_the_mac(self):
        self.assertFalse(self.link.speaker)
        await self.wake()
        self.link.end_listening('submitted')
        await self.settle()
        self.assertEqual(self.client.kinds()[-1], Event.VOICE_ASSISTANT_RUN_END)
        self.assertEqual(self.adapter.fed, [FLUSH])
        self.assertEqual(self.link.phase, 'idle')


class ResamplerTests(unittest.TestCase):
    def tone(self, freq, seconds, rate=24000):
        t = np.arange(int(rate * seconds)) / rate
        return (np.sin(2 * np.pi * freq * t) * 10000).astype('<i2').tobytes()

    def test_keeps_speech_band_and_drops_what_16k_cannot_hold(self):
        out = np.frombuffer(Resampler()(self.tone(1000, 1.0)), '<i2').astype(float)
        self.assertAlmostEqual(len(out), 16000, delta=2)
        middle = out[1000:-1000]
        spectrum = np.abs(np.fft.rfft(middle))
        self.assertAlmostEqual(np.argmax(spectrum) * 16000 / len(middle), 1000, delta=5)
        self.assertGreater(np.sqrt(np.mean(middle ** 2)), 6000)  # 1 kHz 基本不衰减（原始 RMS 约 7071）
        high = np.frombuffer(Resampler()(self.tone(10000, 1.0)), '<i2').astype(float)
        self.assertLess(np.sqrt(np.mean(high[1000:-1000] ** 2)), 300)  # 10 kHz 混叠被滤掉

    def test_chunked_equals_one_shot_even_with_odd_byte_chunks(self):
        audio = self.tone(440, 0.3)
        whole = Resampler()(audio)
        resampler = Resampler()
        parts = b''.join(resampler(audio[i:i + 333]) for i in range(0, len(audio), 333))
        self.assertEqual(parts, whole)


class PlayerTests(unittest.IsolatedAsyncioTestCase):
    def make(self, *, speaker=True, phase='awaiting'):
        self.link = NS(speaker=speaker, phase=phase, sent=[], ended=0)

        def start_response():
            if self.link.phase in ('awaiting', 'speaking'):
                self.link.phase = 'speaking'
            return self.link.phase == 'speaking'

        self.link.start_response = start_response
        self.link.send_audio = self.link.sent.append
        self.link.end_response = lambda: setattr(self.link, 'ended', self.link.ended + 1)
        self.fallback = Mock(gain=1.0, wait_played=AsyncMock(return_value=True), run=AsyncMock(),
                             wait_started=AsyncMock(return_value=1.0))
        self.player = SatellitePlayer(self.link, self.fallback)
        self.task = asyncio.create_task(self.player.run())
        self.addAsyncCleanup(self.stop)

    async def stop(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)

    async def test_device_playback_is_resampled_paced_and_reports_start_and_end(self):
        self.make()
        audio = bytes(24000 * 2 * 3 // 10)  # 0.3 秒 24 kHz
        began = asyncio.get_running_loop().time()
        self.player.append(audio, 'tts_1', 0)
        self.player.finish('tts_1')
        self.assertIsNotNone(await self.player.wait_started('tts_1'))
        self.assertTrue(await self.player.wait_played('tts_1'))
        self.assertGreaterEqual(asyncio.get_running_loop().time() - began, 0.3)  # 播完才算
        self.assertEqual(self.link.phase, 'speaking')
        self.assertAlmostEqual(sum(map(len, self.link.sent)), 16000 * 2 * 3 // 10, delta=4)
        self.assertTrue(self.player.playback_finished('tts_1'))
        self.fallback.append.assert_not_called()
        self.player.end_response()
        self.assertEqual(self.link.ended, 1)

    async def test_never_runs_far_ahead_of_the_device_buffer(self):
        self.make()
        self.player.append(bytes(24000 * 2 * 3), 'tts_1', 0)  # 3 秒
        await asyncio.sleep(0.2)
        sent_s = sum(map(len, self.link.sent)) / 2 / 16000
        self.assertLess(sent_s, 0.2 + SatellitePlayer.LEAD + 0.05)

    async def test_clear_drops_pending_audio_and_marks_it_unplayed(self):
        self.make()
        self.player.append(bytes(24000 * 2 * 3), 'tts_1', 0)
        await asyncio.sleep(0.05)
        self.player.clear()
        self.assertFalse(await self.player.wait_played('tts_1'))
        self.assertFalse(self.player.playback_finished('tts_1'))
        sent = len(self.link.sent)
        await asyncio.sleep(0.1)
        self.assertEqual(len(self.link.sent), sent)
        self.fallback.clear.assert_called_once()

    async def test_without_device_speaker_everything_goes_to_the_mac_player(self):
        self.make(speaker=False)
        self.player.append(b'pcm', 'tts_1', 0)
        self.player.finish('tts_1')
        self.assertTrue(await self.player.wait_played('tts_1'))
        self.fallback.append.assert_called_once_with(b'pcm', 'tts_1', 0)
        self.fallback.finish.assert_called_once_with('tts_1')
        self.assertEqual(self.link.sent, [])

    async def test_audio_outside_a_run_is_skipped(self):
        self.make(phase='idle')
        self.player.append(bytes(4800), 'tts_1', 0)
        self.player.finish('tts_1')
        self.assertFalse(await self.player.wait_played('tts_1'))
        self.assertEqual(self.link.sent, [])


class MakeSatelliteTests(unittest.TestCase):
    def test_only_enabled_with_a_host_and_only_for_asr_input(self):
        from voice.qwen_alpha import make_satellite
        with patch.dict('os.environ', {'ALPHA_SATELLITE_HOST': ''}):
            self.assertIsNone(make_satellite('asr'))
        with patch.dict('os.environ', {'ALPHA_SATELLITE_HOST': '192.168.1.115', 'ALPHA_SATELLITE_KEY': ''}):
            link = make_satellite('asr')
            self.assertEqual((link.host, link.noise_psk, link.listen_s), ('192.168.1.115', None, 8.0))
            with self.assertRaises(ValueError):
                make_satellite('omni')


if __name__ == '__main__':
    unittest.main()
