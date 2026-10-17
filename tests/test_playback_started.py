"""首帧计时来自输出回调，使用假声卡数据，不打开设备。"""
import asyncio
import time
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase
import numpy as np
from voice.audio import StreamingPlayer


class PlaybackStartedTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.player = StreamingPlayer()
        self.player.ready = True

    def append(self, item='one', frames=240):
        self.player.append(np.ones(frames, dtype='<i2').tobytes(), item, 0)

    async def callback(self, frames=240, info=None):
        await asyncio.to_thread(self.player.callback, np.zeros((frames, 1), dtype=np.float32), frames, info, None)

    async def test_waits_for_callback_then_device_delay(self):
        self.append()
        self.player.latency = .04
        task = asyncio.create_task(self.player.wait_started('one'))
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        before = time.perf_counter()
        await self.callback()
        await asyncio.sleep(.005)
        self.assertFalse(task.done())
        started = await asyncio.wait_for(task, 1)
        self.assertGreaterEqual(started, before + .04)
        self.assertGreaterEqual(time.perf_counter(), started)

    async def test_clear_before_callback_or_before_device_time_means_not_played(self):
        for callback_first in (False, True):
            self.append('item' + str(callback_first))
            self.player.latency = .1
            task = asyncio.create_task(self.player.wait_started('item' + str(callback_first)))
            await asyncio.sleep(0)
            if callback_first:
                await self.callback()
            self.player.clear()
            self.assertIsNone(await asyncio.wait_for(task, 1))

    async def test_paused_callback_does_not_mark_playback_started(self):
        self.append()
        task = asyncio.create_task(self.player.wait_started('one'))
        await asyncio.sleep(0)
        self.player.pause()
        await self.callback()
        self.assertFalse(task.done())
        self.player.resume()
        await self.callback()
        self.assertIsInstance(await asyncio.wait_for(task, 1), float)

    async def test_each_item_has_own_dac_offset_within_same_callback(self):
        self.append('one')
        self.append('two')
        first = asyncio.create_task(self.player.wait_started('one'))
        second = asyncio.create_task(self.player.wait_started('two'))
        await asyncio.sleep(0)
        await self.callback(480, NS(outputBufferDacTime=20.02, currentTime=20.0))
        a, b = await asyncio.gather(first, second)
        self.assertGreaterEqual(b - a, .009)
        self.assertLess(b - a, .02)

    async def test_missing_or_unavailable_audio_returns_none(self):
        self.assertIsNone(await self.player.wait_started('missing'))
        self.player.ready = False
        self.append()
        self.assertIsNone(await self.player.wait_started('one'))
