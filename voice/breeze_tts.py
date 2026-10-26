"""本地 Breeze TTS 2（家里 PC 上的 breeze_infer.api）+ 云端兜底。

BreezeTTS：克隆参考录音合成，接口和 QwenTTS 一样（ready / stream / close / voice），输出 24 kHz 16 位单声道 PCM，
Mac 播放器和卫星播放器都能直接用。纯克隆模式不能发 cfg_scale（服务端模板没有 negative prompt，会 500）。
朗读标记（voice/speech_markup.py）：声音事件原样交给服务端；语气用音色指导模式（参考录音 + 指令 + cfg_scale）。
不超过 CACHE_MAX_CHARS 字的句子完整合成后缓存，下次直接播；ready() 预先合成 PRESET_PHRASES。

FallbackTTS：每句先试本地；连不上、报错、忙（409）或 FIRST_AUDIO_S 内没出第一块音频，这一句改用云端，
并把本地标成不可用：之后直接走云端，后台每 PROBE_S 秒探测一次，恢复了自动切回。播到一半本地才出错的那句
算失败（不拼接两种声音），下一句起用云端。

2026-10-25 实测（4070 Ti SUPER、--fast-backbone-decode --fast-depth-decoder、局域网）：首包 0.23～0.28 秒，
实时率 0.52～0.54；Qwen 云端（新加坡）首包约 0.6 秒。启动后第一次克隆请求要预热（首包 1.9 秒），ready() 先合成一句。
句首带 [笑]/[叹气] 首包 0.5～0.7 秒；严肃语气首包 0.29 秒、实时率 0.77。
"""
import asyncio
import logging
import time

import httpx2 as httpx

from collections import OrderedDict

from core.paths import PROJECT_ROOT
from core.settings import setting
from voice.speech_markup import plain_speech

logger = logging.getLogger('alpha.qwen_alpha')

FIRST_AUDIO_S = 3.0  # 本地这么久没出第一块音频就改用云端
RETRY_AFTER_S = 30.0  # 本地失败后多久探测一次
WARMUP_TEXT = '好的。'  # 不进缓存：探测恢复时必须真的请求服务端
PRESET_PHRASES = ('我查一下。',)  # 搜索开始时的过渡话，每次都一样
STYLE_INSTRUCTIONS = {'serious': '语气严肃、冷静，语速稍慢，像在提醒重要的事。'}
DIRECTION_CFG_SCALE = 4
CACHE_MAX_CHARS = 12  # 按去掉标记后的字数
CACHE_SIZE = 64
CACHE_CHUNK = 9600  # 缓存按 0.2 秒一块吐出，打断时能及时停


class BreezeBusy(RuntimeError):
    """服务端一次只合成一句，上一句还没结束时返回 409。"""


class BreezeTTS:
    voice = 'breeze'

    def __init__(self, url, ref_audio, ref_text, *, seed=42, client_factory=httpx.AsyncClient):
        self.url = url.rstrip('/')
        self.ref_audio, self.ref_text, self.seed = ref_audio, ref_text, seed
        self.client_factory = client_factory
        self.client = None
        self.cache = OrderedDict()  # (文字, 语气) → 整句 PCM

    def _client(self):
        if self.client is None:
            # 连接超时要短：PC 睡眠或关机时 TCP 连不上，等久了这一句就晚了；读超时给生成留余量。
            self.client = self.client_factory(timeout=httpx.Timeout(10.0, connect=1.5))
        return self.client

    async def ready(self):
        """确认服务在、预热克隆路径（启动后第一次克隆请求慢）、合成常用短句。失败抛异常，由调用方决定是否改用云端。"""
        response = await self._client().get(self.url + '/v1/model')
        response.raise_for_status()
        async for _ in self._synthesize(WARMUP_TEXT, None):
            pass
        for text in PRESET_PHRASES:
            if (text, None) not in self.cache:
                async for _ in self.stream(text):
                    pass

    async def stream(self, text, *, turn_id=None, style=None):
        key = (text, style)
        if key in self.cache:
            self.cache.move_to_end(key)
            pcm = self.cache[key]
            for start in range(0, len(pcm), CACHE_CHUNK):
                yield pcm[start:start + CACHE_CHUNK]
            return
        keep = len(plain_speech(text)) <= CACHE_MAX_CHARS
        pcm = bytearray()
        async for chunk in self._synthesize(text, style):
            if keep:
                pcm += chunk
            yield chunk
        if keep:  # 中途取消或出错走不到这里，不会存下半句
            self.cache[key] = bytes(pcm)
            while len(self.cache) > CACHE_SIZE:
                self.cache.popitem(last=False)

    async def _synthesize(self, text, style):
        files = {'ref_audio': ('ref.wav', self.ref_audio, 'audio/wav')}
        data = {'text': text, 'ref_text': self.ref_text, 'seed': str(self.seed)}
        if style in STYLE_INSTRUCTIONS:
            data.update(instruction=STYLE_INSTRUCTIONS[style], cfg_scale=str(DIRECTION_CFG_SCALE))
        odd = b''
        async with self._client().stream('POST', self.url + '/v1/audio/speech', data=data, files=files) as response:
            if response.status_code == 409:
                raise BreezeBusy('Breeze 正在合成上一句')
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                # 网络分块不保证落在 16 位样本边界上，播放器按 int16 解析，留下半个样本拼到下一块。
                chunk = odd + chunk
                cut = len(chunk) - len(chunk) % 2
                odd = chunk[cut:]
                if cut:
                    yield chunk[:cut]

    async def close(self):
        client, self.client = self.client, None
        if client is not None:
            await client.aclose()


class FallbackTTS:
    """先用本地，不可用就用云端；接口和 QwenTTS 一样。"""

    def __init__(self, primary, fallback, *, first_audio_s=FIRST_AUDIO_S, retry_after_s=RETRY_AFTER_S):
        self.primary, self.fallback = primary, fallback
        self.first_audio_s, self.retry_after_s = first_audio_s, retry_after_s
        self.primary_ok = True
        self.probe_task = None
        self.voice = primary.voice  # 当前在用的（日志里的“TTS首包 xxx”）

    async def ready(self):
        try:
            await self.primary.ready()
            logger.info('[TTS] 本地 %s 可用，云端 %s 兜底', self.primary.voice, self.fallback.voice)
        except Exception as exc:
            self._mark_down(exc)
        await self.fallback.ready()

    async def stream(self, text, *, turn_id=None, style=None):
        if self.primary_ok:
            source = self.primary.stream(text, turn_id=turn_id, style=style)
            started = False
            try:
                async with asyncio.timeout(self.first_audio_s):
                    first = await anext(source)
                started = True
                self.voice = self.primary.voice
                yield first
                async for chunk in source:
                    yield chunk
                return
            except Exception as exc:
                self._mark_down(exc)
                if started:
                    raise  # 播了一半：这一句算失败，不和云端的声音拼在一起
            finally:
                await source.aclose()
        self._ensure_probe()  # close() 取消过探测（例如启动时云端预连接失败）的话这里补上
        self.voice = self.fallback.voice
        async for chunk in self.fallback.stream(text, turn_id=turn_id, style=style):
            yield chunk

    def _mark_down(self, exc):
        if self.primary_ok:
            logger.warning('[TTS] 本地 %s 不可用（%s），改用云端 %s；每 %.0f 秒探测一次', self.primary.voice,
                           type(exc).__name__ if not str(exc) else f'{type(exc).__name__}: {exc}',
                           self.fallback.voice, self.retry_after_s)
        self.primary_ok = False
        self._ensure_probe()

    def _ensure_probe(self):
        if not self.primary_ok and (self.probe_task is None or self.probe_task.done()):
            self.probe_task = asyncio.create_task(self._probe(), name='breeze-probe')

    async def _probe(self):
        while not self.primary_ok:
            await asyncio.sleep(self.retry_after_s)
            try:
                await self.primary.close()  # 丢掉可能半开的连接
                await self.primary.ready()
            except Exception:
                continue
            self.primary_ok = True
            logger.info('[TTS] 本地 %s 恢复，切回本地', self.primary.voice)

    async def close(self):
        if self.probe_task is not None:
            self.probe_task.cancel()
            await asyncio.gather(self.probe_task, return_exceptions=True)
        await self.primary.close()
        await self.fallback.close()


def with_local_tts(cloud):
    """BREEZE_TTS_URL 填了就本地优先、云端兜底；没填或参考录音读不到，原样返回云端 TTS。"""
    url = setting('BREEZE_TTS_URL')
    if not url or not setting('BREEZE_TTS_REF_AUDIO') or not setting('BREEZE_TTS_REF_TEXT'):
        return cloud  # 缺参考录音或原文：启动检查已经警告过
    path = PROJECT_ROOT / setting('BREEZE_TTS_REF_AUDIO')
    try:
        ref_audio = path.read_bytes()
    except OSError as exc:
        logger.error('[TTS] 读不到 BREEZE_TTS_REF_AUDIO（%s：%s），只用云端', path, type(exc).__name__)
        return cloud
    return FallbackTTS(BreezeTTS(url, ref_audio, setting('BREEZE_TTS_REF_TEXT')), cloud)
