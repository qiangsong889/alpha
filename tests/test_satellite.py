"""ESPHome 语音卫星（J9）：唤醒、推流、一轮的结束时机和打断。不连真实设备。"""

import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock

from aioesphomeapi import VoiceAssistantEventType as Event

from voice.satellite import FLUSH, SatelliteLink


class FakeClient:
    instances = []

    def __init__(self, host, port, password, *, client_info, noise_psk):
        self.host, self.noise_psk = host, noise_psk
        self.events, self.handlers, self.on_stop = [], None, None
        self.disconnect = AsyncMock()
        FakeClient.instances.append(self)

    async def connect(self, on_stop, login):
        self.on_stop = on_stop

    async def device_info(self):
        return NS(name='alpha-xvf', esphome_version='2026.9.1')

    def subscribe_voice_assistant(self, **handlers):
        self.handlers = handlers
        return Mock()

    def send_voice_assistant_event(self, kind, data):
        self.events.append(kind)


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


class SatelliteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        FakeClient.instances.clear()
        self.clock = Clock()
        self.link = SatelliteLink('192.168.1.115', noise_psk='key', listen_s=8, client_factory=FakeClient,
                                  clock=self.clock)
        self.adapter = FakeAdapter()
        self.tasks = [asyncio.create_task(self.link.run()), asyncio.create_task(self.link.feed(self.adapter))]
        for _ in range(5):
            await asyncio.sleep(0)
        self.client = FakeClient.instances[0]

    async def asyncTearDown(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def wake(self):
        return await self.client.handlers['handle_start']('conv', 0, None, 'Hey Jarvis')

    async def settle(self):
        await asyncio.sleep(0.05)

    async def test_wake_streams_audio_over_api_into_the_segmenter(self):
        self.assertEqual(self.client.noise_psk, 'key')
        self.assertEqual(await self.wake(), 0)  # 0：音频走 API 连接
        self.assertEqual(self.client.events, [Event.VOICE_ASSISTANT_RUN_START, Event.VOICE_ASSISTANT_STT_START])
        await self.client.handlers['handle_audio'](b'\x01\x00' * 256, None)
        await self.client.handlers['handle_audio'](b'\x02\x00' * 256, None)
        await self.settle()
        self.assertEqual(self.adapter.fed, [b'\x01\x00' * 256, b'\x02\x00' * 256])

    async def test_audio_outside_a_run_is_dropped(self):
        await self.client.handlers['handle_audio'](b'\x01\x00' * 256, None)
        await self.settle()
        self.assertEqual(self.adapter.fed, [])

    async def test_submission_ends_the_run_and_flushes_the_segmenter(self):
        await self.wake()
        self.link.end_run('submitted')
        await self.settle()
        self.assertEqual(self.client.events[-1], Event.VOICE_ASSISTANT_RUN_END)
        self.assertEqual(self.adapter.fed, [FLUSH])
        # 结束后设备晚到的音频不再进断句器，也不会重复发 RUN_END。
        await self.client.handlers['handle_audio'](b'\x01\x00', None)
        self.link.end_run('submitted')
        await self.settle()
        self.assertEqual(self.adapter.fed, [FLUSH])
        self.assertEqual(self.client.events.count(Event.VOICE_ASSISTANT_RUN_END), 1)

    async def test_no_voice_for_listen_seconds_ends_the_run(self):
        await self.wake()
        self.clock.now += 7
        await self.client.handlers['handle_audio'](b'\x00\x00', None)
        await self.settle()
        self.assertNotIn(Event.VOICE_ASSISTANT_RUN_END, self.client.events)
        # 说话中刷新计时：说完 8 秒内都不结束。
        self.adapter.segmenter.active = object()
        await self.client.handlers['handle_audio'](b'\x00\x00', None)
        await self.settle()
        self.adapter.segmenter.active = None
        self.clock.now += 7.9
        await self.client.handlers['handle_audio'](b'\x00\x00', None)
        await self.settle()
        self.assertNotIn(Event.VOICE_ASSISTANT_RUN_END, self.client.events)
        self.clock.now += 0.2
        await asyncio.sleep(0.6)  # 没有新音频也要按时结束
        self.assertEqual(self.client.events[-1], Event.VOICE_ASSISTANT_RUN_END)

    async def test_device_stop_ends_the_run(self):
        await self.wake()
        await self.client.handlers['handle_stop'](False)
        await self.settle()
        self.assertEqual(self.client.events[-1], Event.VOICE_ASSISTANT_RUN_END)
        self.assertFalse(self.link.active)

    async def test_wake_during_playback_interrupts_it(self):
        self.adapter.receiver.active = True
        await self.wake()
        self.adapter.receiver.interrupt.assert_awaited_once()
        self.adapter.receiver.active = False
        self.link.end_run('submitted')
        await self.wake()
        self.adapter.receiver.interrupt.assert_awaited_once()

    async def test_disconnect_ends_the_run_and_reconnects(self):
        await self.wake()
        await self.client.on_stop(False)
        await self.settle()
        self.assertFalse(self.link.active)
        self.client.disconnect.assert_awaited_once()
        self.assertNotIn(Event.VOICE_ASSISTANT_RUN_END, self.client.events)  # 连接已断，不再发



class MakeSatelliteTests(unittest.TestCase):
    def test_only_enabled_with_a_host_and_only_for_asr_input(self):
        from unittest.mock import patch
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
