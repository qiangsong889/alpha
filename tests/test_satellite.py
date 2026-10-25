"""ESPHome 语音卫星（J9）：唤醒、推流、一轮的阶段和结束时机、打断、设备播放。不连真实设备。"""

import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from aioesphomeapi import VoiceAssistantEventType as Event
import numpy as np

from voice import satellite as module
from voice.satellite import FLUSH, Resampler, SatelliteLink, SatellitePlayer, split_dismissal

WITH_SPEAKER, WITHOUT_SPEAKER = 7, 5  # VOICE_ASSISTANT | SPEAKER | API_AUDIO；旧固件没有 SPEAKER


class FakeClient:
    instances = []
    flags = WITH_SPEAKER

    def __init__(self, host, port, password, *, client_info, noise_psk):
        self.host, self.noise_psk = host, noise_psk
        self.events, self.audio, self.handlers, self.on_stop = [], [], None, None
        self.services_called = []
        self.disconnect = AsyncMock()
        FakeClient.instances.append(self)

    async def connect(self, on_stop, login):
        self.on_stop = on_stop

    async def device_info(self):
        return NS(name='alpha-xvf', esphome_version='2026.9.1', voice_assistant_feature_flags=self.flags)

    async def list_entities_services(self):
        return [], [NS(name='dismiss', key=7)]

    async def execute_service(self, service, data):
        self.services_called.append(service.name)

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
    followup_s = 0.0  # 默认关掉跟进窗口；FollowupTests 单独测

    async def asyncSetUp(self):
        FakeClient.instances.clear()
        FakeClient.flags = self.flags
        self.clock = Clock()
        self.link = SatelliteLink('192.168.1.115', noise_psk='key', listen_s=8, followup_s=self.followup_s,
                                  client_factory=FakeClient, clock=self.clock)
        self.adapter = FakeAdapter()
        self.tasks = [asyncio.create_task(self.link.run()), asyncio.create_task(self.link.feed(self.adapter))]
        await self.settle()
        self.client = FakeClient.instances[0]

    async def asyncTearDown(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def wake(self):
        # 唤醒词触发的一轮带 USE_WAKE_WORD（2）；设备自动开的跟进和诊断按钮不带
        return await self.client.handlers['handle_start']('conv', 2, None, 'Hey Jarvis')

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
        self.assertEqual(self.client.services_called, ['dismiss'])  # 唤醒后没开口的超时也响退出提示音

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


class DismissTests(LinkTestCase):
    async def test_dismissal_words(self):
        cases = {
            '退下': '', '退下吧。': '', '没事了': '', '没事的。': '', '拜拜！': '',  # 没事的：ASR 常这样转
            'Jarvis。；拜拜。': '', '拜拜；退下': '',
            '没有用啊。；退下。': '没有用啊',  # 跟进窗口把前面的话连在了一起（2026-10-25 实测）
            '把灯关了，拜拜': '把灯关了', '把灯关了；拜拜。': '把灯关了',
            '可以了。': '', '好的，可以了': '好的',
            '再见。': '', '结束。': '', '结束对话吧': '', 'Finish.': '', 'Never mind.': '', 'never mind': '', 'Turn off the lights. Never mind.': 'Turn off the lights',
        }
        for text, rest in cases.items():
            with self.subTest(text=text):
                self.assertEqual(split_dismissal(text), (rest, True))
        for text in ('没事', '拜拜，明天见', '退下之前把灯关了', '现在几点', '', '可以了吗', '这样就可以了吗？', 'Never mind the lights', 'Finish the email', '调到 25.5 度', '结束了吗', '会议几点结束'):
            with self.subTest(text=text):
                self.assertEqual(split_dismissal(text), (text, False))

    async def test_dismiss_ends_the_run_without_follow_up_and_plays_the_exit_beep(self):
        self.link.followup_s = 6.0
        await self.wake()
        self.assertTrue(self.link.dismiss())
        await self.settle()
        self.assertEqual(self.client.kinds()[-1], Event.VOICE_ASSISTANT_RUN_END)
        self.assertNotIn(Event.VOICE_ASSISTANT_INTENT_END, self.client.kinds())
        self.assertEqual(self.client.services_called, ['dismiss'])
        self.assertEqual(self.link.phase, 'idle')
        self.assertFalse(self.link.dismiss())  # 不在收音时什么都不做

    async def test_route_dismisses_unless_alpha_just_asked_a_question(self):
        from voice.qwen_alpha import route_satellite_submission
        bridge = NS(last_reply='好的。', last_reply_at=0.0)
        await self.wake()
        self.assertIsNone(route_satellite_submission('拜拜', self.link, bridge, clock=lambda: 1000.0))
        self.assertEqual(self.link.phase, 'idle')
        # 刚问过“确认发送吗？”：没事了是在回答，整句照常交给大脑
        bridge = NS(last_reply='确认发送吗？', last_reply_at=990.0)
        await self.wake()
        self.assertEqual(route_satellite_submission('没事了', self.link, bridge, clock=lambda: 1000.0), '没事了')
        self.assertEqual(self.link.phase, 'awaiting')
        # 很久以前的问句不算
        bridge = NS(last_reply='确认发送吗？', last_reply_at=0.0)
        self.link.end_run('test')
        await self.wake()
        self.assertIsNone(route_satellite_submission('没事了', self.link, bridge, clock=lambda: 1000.0))
        # 普通请求照常提交
        await self.wake()
        self.assertEqual(route_satellite_submission('现在几点', self.link, bridge, clock=lambda: 1000.0), '现在几点')
        self.assertFalse(self.link.dismiss_after_answer)

    async def test_request_then_bye_is_answered_then_the_run_ends_with_the_exit_beep(self):
        from voice.qwen_alpha import route_satellite_submission
        self.link.followup_s = 6.0
        bridge = NS(last_reply='', last_reply_at=0.0)
        await self.wake()
        self.assertEqual(route_satellite_submission('把灯关了，拜拜', self.link, bridge), '把灯关了')
        self.link.start_response()
        self.link.end_response()
        self.assertEqual(self.client.kinds()[-2:], [Event.VOICE_ASSISTANT_TTS_STREAM_END,
                                                    Event.VOICE_ASSISTANT_RUN_END])
        self.assertNotIn(Event.VOICE_ASSISTANT_INTENT_END, self.client.kinds())  # 不开跟进窗口
        self.assertEqual(self.client.services_called, [])  # 等设备放完回答再响
        await asyncio.sleep(0.7)
        self.assertEqual(self.client.services_called, ['dismiss'])
        await self.wake()
        self.assertFalse(self.link.dismiss_after_answer)  # 只管那一轮


class FollowupTests(LinkTestCase):
    followup_s = 6.0

    async def answer(self):
        await self.wake()
        self.link.end_listening('submitted')
        self.link.start_response()
        self.link.end_response()

    async def test_after_the_answer_the_device_is_asked_to_keep_listening_without_run_end(self):
        await self.answer()
        self.assertEqual(self.client.events[-2:], [
            (Event.VOICE_ASSISTANT_INTENT_END, {'continue_conversation': '1'}),
            (Event.VOICE_ASSISTANT_TTS_STREAM_END, None)])
        self.assertNotIn(Event.VOICE_ASSISTANT_RUN_END, self.client.kinds())  # 晚到会结束掉设备新开的一轮
        self.assertEqual(self.link.phase, 'idle')

    async def test_the_device_started_follow_up_uses_the_short_timeout(self):
        await self.answer()
        self.clock.now += 1
        await self.client.handlers['handle_start']('conv', 0, None, 'Hey Jarvis')  # 跟进会沿用上次的唤醒词名字
        self.assertTrue(self.link.followup)
        self.clock.now += 5.9
        await asyncio.sleep(0.6)
        self.assertEqual(self.link.phase, 'listening')
        self.assertEqual(self.client.services_called, [])
        self.clock.now += 0.2
        await asyncio.sleep(0.6)
        self.assertEqual(self.link.phase, 'idle')
        self.assertEqual(self.client.kinds()[-1], Event.VOICE_ASSISTANT_RUN_END)
        self.assertEqual(self.client.services_called, ['dismiss'])  # 跟进窗口结束响退出提示音

    async def test_wake_word_runs_and_late_or_button_starts_are_not_follow_ups(self):
        await self.answer()
        await self.wake()  # 跟进窗口里说了唤醒词：普通的一轮
        self.assertFalse(self.link.followup)
        self.link.end_run('test')
        await self.client.handlers['handle_start']('conv', 0, None, None)  # 诊断按钮：没有紧跟在回答后面
        self.assertFalse(self.link.followup)
        self.link.end_run('test')
        await self.answer()
        self.clock.now += 6  # 设备太久才来 start，不算跟进
        await self.client.handlers['handle_start']('conv', 0, None, None)
        self.assertFalse(self.link.followup)

    async def test_barge_in_does_not_ask_for_a_follow_up(self):
        await self.wake()
        self.link.end_listening('submitted')
        self.link.start_response()
        self.adapter.receiver.active = True
        await self.barge_in_event()
        self.assertNotIn(Event.VOICE_ASSISTANT_INTENT_END, self.client.kinds())
        self.assertEqual(self.client.kinds()[-1], Event.VOICE_ASSISTANT_RUN_END)

    async def barge_in_event(self):
        self.client.service_callback(NS(service='esphome.alpha_barge_in', is_event=True, data={}))
        await self.settle()


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
            self.assertEqual((link.host, link.noise_psk, link.listen_s, link.followup_s),
                             ('192.168.1.115', None, 8.0, 6.0))
            with self.assertRaises(ValueError):
                make_satellite('omni')


if __name__ == '__main__':
    unittest.main()
