"""本地 Breeze TTS + 云端兜底：请求格式、各种失败时改用云端、探测恢复。不连真实服务。"""

import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

import httpx2 as httpx

from voice.breeze_tts import BreezeBusy, BreezeTTS, FallbackTTS, with_local_tts


class FakeCloud:
    voice = 'longanlingxin_v3.1'

    def __init__(self):
        self.texts, self.styles = [], []
        self.ready = AsyncMock()
        self.close = AsyncMock()

    async def stream(self, text, *, turn_id=None, style=None):
        self.texts.append(text)
        self.styles.append(style)
        yield b'cloud'


def breeze(handler):
    return BreezeTTS('http://pc:7860/', b'RIFFref', '参考原文', seed=7,
                     client_factory=lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kw))


def audio_handler(requests, chunks=(b'\x01\x02\x03', b'\x04\x05\x06')):
    async def body():
        for chunk in chunks:
            yield chunk

    def handler(request):
        requests.append(request)
        if request.url.path == '/v1/model':
            return httpx.Response(200, json={'frame_rate': 12.5})
        return httpx.Response(200, content=body())
    return handler


class BreezeTTSTests(IsolatedAsyncioTestCase):
    async def test_clone_request_fields_without_cfg_scale_and_even_chunks(self):
        requests = []
        tts = breeze(audio_handler(requests))
        self.addAsyncCleanup(tts.close)
        chunks = [c async for c in tts.stream('客厅灯关好了')]
        self.assertEqual(chunks, [b'\x01\x02', b'\x03\x04\x05\x06'])  # 半个样本拼到下一块
        body = requests[0].content
        self.assertIn('客厅灯关好了'.encode(), body)
        self.assertIn('参考原文'.encode(), body)
        self.assertIn(b'RIFFref', body)
        self.assertIn(b'name="seed"', body)
        self.assertNotIn(b'cfg_scale', body)  # 纯克隆模式传 cfg_scale 会 500（2026-10-25 实测）
        self.assertEqual(str(requests[0].url), 'http://pc:7860/v1/audio/speech')

    async def test_busy_server_raises_busy(self):
        tts = breeze(lambda request: httpx.Response(409))
        self.addAsyncCleanup(tts.close)
        with self.assertRaises(BreezeBusy):
            async for _ in tts.stream('第二句'):
                pass

    async def test_ready_checks_the_model_warms_up_and_presynthesizes_common_phrases(self):
        requests = []
        tts = breeze(audio_handler(requests))
        self.addAsyncCleanup(tts.close)
        await tts.ready()
        self.assertEqual([r.url.path for r in requests], ['/v1/model', '/v1/audio/speech', '/v1/audio/speech'])
        self.assertIn('我查一下。'.encode(), requests[2].content)
        await tts.ready()  # 探测恢复时：预热句不走缓存（要真的请求服务端），常用短句已在缓存
        self.assertEqual([r.url.path for r in requests[3:]], ['/v1/model', '/v1/audio/speech'])
        self.assertEqual([c async for c in tts.stream('我查一下。')], [b'\x01\x02\x03\x04\x05\x06'])
        self.assertEqual(len(requests), 5)

    async def test_serious_style_uses_voice_direction_with_cfg_scale(self):
        requests = []
        tts = breeze(audio_handler(requests))
        self.addAsyncCleanup(tts.close)
        [c async for c in tts.stream('[笑] 门没锁。', style='serious')]
        body = requests[0].content
        self.assertIn('[笑] 门没锁。'.encode(), body)  # 声音事件原样交给服务端
        self.assertIn(b'name="instruction"', body)
        self.assertIn(b'name="cfg_scale"', body)

    async def test_short_sentences_are_cached_per_style_long_ones_are_not(self):
        requests = []
        tts = breeze(audio_handler(requests))
        self.addAsyncCleanup(tts.close)
        long_text = '明天早上八点有个会，记得提前十分钟出门。'
        for text, style in (('好的。', None), ('好的。', None), ('好的。', 'serious'), (long_text, None), (long_text, None)):
            chunks = [c async for c in tts.stream(text, style=style)]
            self.assertEqual(b''.join(chunks), b'\x01\x02\x03\x04\x05\x06')
        self.assertEqual(len(requests), 4)  # 第二次“好的。”直接播缓存

    async def test_cancelled_sentence_is_not_cached(self):
        requests = []
        tts = breeze(audio_handler(requests))
        self.addAsyncCleanup(tts.close)
        stream = tts.stream('好的。')
        await anext(stream)
        await stream.aclose()
        [c async for c in tts.stream('好的。')]
        self.assertEqual(len(requests), 2)


class FallbackTTSTests(IsolatedAsyncioTestCase):
    def make(self, handler, **kwargs):
        self.cloud = FakeCloud()
        self.tts = FallbackTTS(breeze(handler), self.cloud, **kwargs)
        self.addAsyncCleanup(self.tts.close)
        return self.tts

    async def collect(self, text):
        return b''.join([c async for c in self.tts.stream(text)])

    async def test_style_is_passed_to_both_backends(self):
        self.make(lambda request: httpx.Response(500))
        await anext(self.tts.stream('门没锁。', style='serious'))
        self.assertEqual(self.cloud.styles, ['serious'])

    async def test_local_first(self):
        self.make(audio_handler([]))
        self.assertEqual(await self.collect('你好'), b'\x01\x02\x03\x04\x05\x06'[:6])
        self.assertEqual(self.cloud.texts, [])
        self.assertEqual(self.tts.voice, 'breeze')

    async def test_unreachable_local_falls_back_and_stops_trying_until_the_probe_recovers(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if self.down:
                raise httpx.ConnectTimeout('PC 睡着了')
            return audio_handler([])(request)

        self.down = True
        self.make(handler, retry_after_s=0.05)
        self.assertEqual(await self.collect('第一句'), b'cloud')
        self.assertEqual(await self.collect('第二句'), b'cloud')
        self.assertEqual(self.cloud.texts, ['第一句', '第二句'])
        self.assertEqual(calls.count('/v1/audio/speech'), 1)  # 第二句没再等本地超时
        self.assertEqual(self.tts.voice, 'longanlingxin_v3.1')
        self.down = False
        await asyncio.sleep(0.2)  # 后台探测：健康检查 + 预热
        self.assertTrue(self.tts.primary_ok)
        self.assertEqual(await self.collect('第三句'), b'\x01\x02\x03\x04\x05\x06')
        self.assertEqual(self.cloud.texts, ['第一句', '第二句'])

    async def test_busy_or_error_falls_back_for_that_sentence(self):
        for status in (409, 500):
            with self.subTest(status=status):
                self.make(lambda request, s=status: httpx.Response(s))
                self.assertEqual(await self.collect('这一句'), b'cloud')
                self.assertFalse(self.tts.primary_ok)

    async def test_no_first_audio_in_time_falls_back(self):
        async def slow():
            await asyncio.sleep(1)
            yield b'\x01\x02'

        self.make(lambda request: httpx.Response(200, content=slow()), first_audio_s=0.05)
        self.assertEqual(await self.collect('慢'), b'cloud')
        self.assertFalse(self.tts.primary_ok)

    async def test_failure_mid_sentence_fails_that_sentence_then_uses_cloud(self):
        async def broken():
            yield b'\x01\x02'
            raise httpx.ReadError('断了')

        self.make(lambda request: httpx.Response(200, content=broken()))
        received = []
        with self.assertRaises(Exception):
            async for chunk in self.tts.stream('播到一半'):
                received.append(chunk)
        self.assertEqual(received, [b'\x01\x02'])
        self.assertEqual(self.cloud.texts, [])  # 不把云端的声音接在后面
        self.assertEqual(await self.collect('下一句'), b'cloud')

    async def test_ready_tolerates_a_down_local_server(self):
        def handler(request):
            raise httpx.ConnectError('关机了')

        self.make(handler, retry_after_s=10)
        await self.tts.ready()
        self.assertFalse(self.tts.primary_ok)
        self.cloud.ready.assert_awaited_once()

    async def test_probe_restarts_after_close_cancelled_it(self):
        def handler(request):
            raise httpx.ConnectError('关机了')

        self.make(handler, retry_after_s=10)
        await self.collect('第一句')
        await self.tts.close()  # 例如启动时云端预连接失败，主程序会 close
        self.assertTrue(self.tts.probe_task.done())
        await self.collect('第二句')
        self.assertFalse(self.tts.probe_task.done())


class WithLocalTTSTests(IsolatedAsyncioTestCase):
    async def test_enabled_only_with_url_audio_and_text(self):
        cloud = FakeCloud()
        with patch.dict('os.environ', {'BREEZE_TTS_URL': ''}):
            self.assertIs(with_local_tts(cloud), cloud)
        with patch.dict('os.environ', {'BREEZE_TTS_URL': 'http://pc:7860', 'BREEZE_TTS_REF_AUDIO': 'data/none.wav',
                                       'BREEZE_TTS_REF_TEXT': ''}):
            self.assertIs(with_local_tts(cloud), cloud)
        with patch.dict('os.environ', {'BREEZE_TTS_URL': 'http://pc:7860', 'BREEZE_TTS_REF_AUDIO': 'data/no-such.wav',
                                       'BREEZE_TTS_REF_TEXT': '原文'}):
            self.assertIs(with_local_tts(cloud), cloud)  # 读不到参考录音
        with patch.dict('os.environ', {'BREEZE_TTS_URL': 'http://pc:7860', 'BREEZE_TTS_REF_AUDIO': 'README.md',
                                       'BREEZE_TTS_REF_TEXT': '原文'}):
            tts = with_local_tts(cloud)
            self.assertIsInstance(tts, FallbackTTS)
            self.assertEqual(tts.primary.url, 'http://pc:7860')
            await tts.close()
