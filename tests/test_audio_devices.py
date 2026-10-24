"""跟随系统默认音频设备：先关闭所有流再重新初始化 PortAudio，再用新设备重开。不打开真实声卡。"""

import asyncio
from types import SimpleNamespace as NS
import unittest

import numpy as np
from unittest.mock import Mock, patch

from voice import asr_input, audio, audio_devices
from voice import qwen_alpha
from voice.asr_input import AsrInput
from voice.audio import StreamingPlayer
from voice.audio_devices import AudioRoute, CoreAudioDefaults
from core.settings import VOICE, check_settings


class FakeStream:
    """记录打开/关闭；active 一直为真，模拟设备没有出错（旧设备仍然可用）。"""

    def __init__(self, log, kind, device_name):
        self.log, self.kind, self.device_name = log, kind, device_name
        self.active, self.latency = True, 0.01

    def __enter__(self):
        self.log.append(('open', self.kind, self.device_name))
        return self

    def __exit__(self, *exc):
        self.log.append(('close', self.kind, self.device_name))
        return False


async def until(condition, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError('等待超时')
        await asyncio.sleep(0.01)


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_switch_waits_until_every_stream_is_closed_before_reinitializing(self):
        log = []
        route = AudioRoute(follow_input=True, follow_output=True, probe=lambda: ('mic', 'spk'),
                           reinitialize=lambda: log.append('reinitialize'))

        async def owner(name):
            with route.stream():
                await route.changed.wait()
                await asyncio.sleep(0.05 if name == 'slow' else 0)
                log.append(('closed', name))
        owners = [asyncio.create_task(owner('fast')), asyncio.create_task(owner('slow'))]
        await asyncio.sleep(0)
        self.assertTrue(await route.switch('测试'))
        await asyncio.gather(*owners)
        self.assertEqual(log, [('closed', 'fast'), ('closed', 'slow'), 'reinitialize'])
        self.assertFalse(route.changed.is_set())  # 切换结束，允许重新打开

    async def test_a_stream_that_does_not_close_cancels_the_switch_instead_of_reinitializing(self):
        reinitialize = Mock()
        route = AudioRoute(follow_input=True, follow_output=True, probe=lambda: ('mic', 'spk'),
                           reinitialize=reinitialize, close_timeout_s=0.05)
        with route.stream():  # 不配合关闭的流（例如旧的 Omni 收音）
            self.assertFalse(await route.switch('测试'))
        reinitialize.assert_not_called()   # 流还开着时重新初始化可能让进程崩溃
        self.assertFalse(route.changed.is_set())

    async def test_watch_only_reacts_to_followed_devices_and_does_not_retry_every_second(self):
        readings = [('mic', 'spk'), ('mic2', 'spk'),        # 输入变了，但没有跟随输入
                    ('mic2', None),                          # 切换中暂时读不到：跳过
                    ('mic2', 'airpods'), ('mic2', 'airpods'), ('mic2', 'airpods')]
        calls = []

        def probe():
            calls.append(1)
            return readings[min(len(calls), len(readings)) - 1]
        route = AudioRoute(follow_input=False, follow_output=True, probe=probe, interval_s=0)
        switches = []

        async def switch(reason):
            switches.append(reason)
            return False  # 即使这次放弃了，同一个变化也不再每秒重试
        route.switch = switch
        task = asyncio.create_task(route.watch())
        await until(lambda: len(calls) > len(readings) + 3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(len(switches), 1)
        self.assertIn('输出 spk → airpods', switches[0])
        self.assertNotIn('输入', switches[0])

    async def test_route_is_disabled_for_fixed_devices_and_the_old_omni_input(self):
        with patch.object(qwen_alpha, 'setting', side_effect=lambda name: {
                'REALTIME_INPUT_DEVICE': '', 'REALTIME_OUTPUT_DEVICE': 'USB Speaker'}[name]):
            asr = qwen_alpha.audio_route('asr')
            omni = qwen_alpha.audio_route('omni')
        self.assertEqual((asr.follow_input, asr.follow_output), (True, False))
        self.assertFalse(omni.enabled)
        await asyncio.wait_for(omni.watch(), 0.1)  # 不跟随时直接返回，不读系统设备

    def test_core_audio_reader_returns_nothing_outside_macos(self):
        with patch.object(audio_devices.sys, 'platform', 'linux'):
            self.assertEqual(CoreAudioDefaults()(), (None, None))


class StreamOwnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.log = []
        self.default = {'output': '旧扬声器', 'input': '旧麦克风'}
        self.devices = patch.object(audio.sd, 'query_devices', side_effect=lambda device, kind: {
            'name': self.default[kind], 'default_samplerate': 48000, 'max_input_channels': 2})
        self.devices.start()  # asr_input 和 audio 用的是同一个 sounddevice 模块
        self.addCleanup(self.devices.stop)
        self.route = AudioRoute(follow_input=True, follow_output=True,
                                probe=lambda: (self.default['input'], self.default['output']),
                                reinitialize=lambda: self.log.append('reinitialize'))

    async def test_player_reopens_on_the_new_default_output_without_waiting_a_second(self):
        player = StreamingPlayer(self.route)
        with patch.object(audio.sd, 'OutputStream', side_effect=lambda **kw: FakeStream(
                    self.log, 'output', self.default['output'])), \
                patch('builtins.print'):
            task = asyncio.create_task(player.run())
            await until(lambda: player.ready)
            self.default['output'] = 'AirPods'
            started = asyncio.get_running_loop().time()
            self.assertTrue(await self.route.switch('测试'))
            await until(lambda: ('open', 'output', 'AirPods') in self.log)
            elapsed = asyncio.get_running_loop().time() - started
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.log[:4], [('open', 'output', '旧扬声器'), ('close', 'output', '旧扬声器'),
                                        'reinitialize', ('open', 'output', 'AirPods')])
        self.assertLess(elapsed, 0.9)  # 切换不是设备故障，不走 1 秒重试

    async def test_microphone_reopens_within_the_same_asr_session(self):
        adapter = AsrInput(NS(), NS(), route=self.route)
        with patch.object(asr_input.sd, 'RawInputStream', side_effect=lambda **kw: FakeStream(
                    self.log, 'input', self.default['input'])):
            task = asyncio.create_task(adapter.microphone())
            await until(lambda: ('open', 'input', '旧麦克风') in self.log)
            self.default['input'] = 'AirPods 麦克风'
            self.assertTrue(await self.route.switch('测试'))
            await until(lambda: ('open', 'input', 'AirPods 麦克风') in self.log)
            self.assertFalse(task.done())  # 没有抛出 SessionEnded，ASR 连接不用重建
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.log[:4], [('open', 'input', '旧麦克风'), ('close', 'input', '旧麦克风'),
                                        'reinitialize', ('open', 'input', 'AirPods 麦克风')])

    async def test_both_streams_close_before_portaudio_is_reinitialized(self):
        player = StreamingPlayer(self.route)
        adapter = AsrInput(NS(), NS(), route=self.route)
        with patch.object(audio.sd, 'OutputStream', side_effect=lambda **kw: FakeStream(
                    self.log, 'output', self.default['output'])), \
                patch.object(asr_input.sd, 'RawInputStream', side_effect=lambda **kw: FakeStream(
                    self.log, 'input', self.default['input'])), \
                patch('builtins.print'):
            tasks = [asyncio.create_task(player.run()), asyncio.create_task(adapter.microphone())]
            await until(lambda: player.ready and ('open', 'input', '旧麦克风') in self.log)
            self.assertTrue(await self.route.switch('测试'))
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        reinit = self.log.index('reinitialize')
        closes = [index for index, entry in enumerate(self.log) if entry[0] == 'close']
        self.assertEqual(len([c for c in closes if c < reinit]), 2)  # 两个流都在重新初始化之前关闭


class InputChannelTests(unittest.IsolatedAsyncioTestCase):
    """多声道麦克风（例如 XVF3800）：只取 REALTIME_INPUT_CHANNEL 指定的那个声道。"""

    async def run_microphone(self, channel, available):
        opened, received = [], []

        class Stream(FakeStream):
            def __init__(self, **kwargs):
                super().__init__([], 'input', 'xvf')
                opened.append(kwargs)
                # 交错的双声道 int16：左声道 1000（会议输出），右声道 -2000（语音识别波束）
                frames = np.array([1000, -2000] * 4, dtype='<i2')
                kwargs['callback'](frames.tobytes() if kwargs['channels'] == 2 else frames[::2].tobytes(),
                                   4, None, None)

        adapter = AsrInput(NS(), NS())

        async def process(pcm):
            received.append(np.frombuffer(pcm, dtype='<i2').tolist())
            raise asyncio.CancelledError  # 收到第一块就结束
        adapter.process = process
        with patch.dict('os.environ', {'REALTIME_INPUT_CHANNEL': str(channel)}), \
                patch.object(asr_input.sd, 'query_devices', return_value={'max_input_channels': available}), \
                patch.object(asr_input.sd, 'RawInputStream', side_effect=lambda **kw: Stream(**kw)):
            try:
                await adapter.microphone()
            except asyncio.CancelledError:
                pass
        return opened, received

    async def test_only_the_configured_channel_reaches_the_segmenter(self):
        opened, received = await self.run_microphone(2, available=2)
        self.assertEqual(opened[0]['channels'], 2)
        self.assertEqual(received, [[-2000] * 4])  # 右声道，单声道 PCM

    async def test_default_channel_keeps_mono_input_unchanged(self):
        opened, received = await self.run_microphone(1, available=1)
        self.assertEqual(opened[0]['channels'], 1)
        self.assertEqual(received, [[1000] * 4])

    async def test_a_channel_the_device_does_not_have_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, 'REALTIME_INPUT_CHANNEL=2.*1 个声道'):
            await self.run_microphone(2, available=1)

    def test_channel_setting_warns_without_a_fixed_device_or_on_the_omni_path(self):
        base = {'DASHSCOPE_API_KEY': 'k', 'QWEN_WORKSPACE_ID': 'w', 'TYPESAFE_API_KEY': 't'}
        fixed = check_settings(VOICE, {**base, 'REALTIME_INPUT_CHANNEL': '2', 'REALTIME_INPUT_DEVICE': 'reSpeaker Flex'})
        self.assertFalse(any('REALTIME_INPUT_CHANNEL' in w for w in fixed.warnings))
        floating = check_settings(VOICE, {**base, 'REALTIME_INPUT_CHANNEL': '2'})
        self.assertTrue(any('没有固定 REALTIME_INPUT_DEVICE' in w for w in floating.warnings))
        omni = check_settings(VOICE, {**base, 'REALTIME_INPUT_CHANNEL': '2', 'REALTIME_INPUT_DEVICE': 'x',
                                      'QWEN_ALPHA_INPUT': 'omni'})
        self.assertTrue(any('只对 asr 收音生效' in w for w in omni.warnings))
        self.assertEqual(check_settings(VOICE, base).values['REALTIME_INPUT_CHANNEL'], 1)


if __name__ == '__main__':
    unittest.main()
