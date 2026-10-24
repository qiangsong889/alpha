"""单段在途的 Qwen ASR 队列；上一段完成/失败前，下一段连append也不发送。"""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import base64
import json
import math
import re
import time
from uuid import uuid4

from websockets.asyncio.client import connect
from voice.segmenter import SegmentReady
from core.costs import CumulativeUsage

PREFIX = 'conversation.item.input_audio_transcription.'
MODEL = 'qwen3-asr-flash-realtime'
# 上下文增强（官方 SDK 的 corpus_text，发送时是 input_audio_transcription.corpus.text）。
# 中文识别经常把夹在中文里的“Alpha”漏掉或认成“奥特”，而播报期间只接受“Alpha……”打断。
CORPUS = ('这是和家庭语音助手的对话。语音助手的名字是英文 Alpha（也读作阿尔法）。'
          '用户常用“Alpha，”开头叫它，例如：“Alpha，我的购物清单上有什么？”“Alpha，停一下。”')


def websocket_url(workspace, region):
    regions = {'singapore': 'ap-southeast-1', 'beijing': 'cn-beijing'}
    if region not in regions or not re.fullmatch(r'[A-Za-z0-9-]+', workspace):
        raise ValueError('需要有效的 QWEN_WORKSPACE_ID 和 QWEN_REGION=singapore|beijing')
    return f'wss://{workspace}.{regions[region]}.maas.aliyuncs.com/api-ws/v1/realtime?model={MODEL}'


@dataclass(frozen=True)
class AsrEvent:
    kind: str                  # transcript / failed
    seg_id: str
    text: str
    elapsed_s: float           # 从本段出队开始，包含连接及上传。
    commit_to_result_s: float | None = None
    queue_s: float = 0


class AsrClient:
    def __init__(self, url, key, emit, *, timeout_s=5, queue_size=8, connector=connect, notify=None):
        if not key or not math.isfinite(timeout_s) or timeout_s <= 0 or queue_size <= 0:
            raise ValueError('超时和队列容量必须大于零')
        self.url, self.key, self.emit = url, key, emit
        self.timeout_s, self.connector = timeout_s, connector
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.ws = self.reader = self.incoming = None
        self.notify = notify
        self.usage_meter = CumulativeUsage()

    async def ready(self):
        """开始收音前预连接；失败时关闭半开的连接，后续段落仍可重新连接。"""
        try:
            await self._ready()
        except BaseException:
            await self.close()
            raise

    async def _notify(self, kind, seg_id, payload=None):
        if self.notify is not None:
            await self.notify(kind, seg_id, payload)

    async def submit(self, segment: SegmentReady):
        await self.queue.put((segment, time.perf_counter()))

    async def _send(self, kind, **fields):
        await self.ws.send(json.dumps({'event_id': uuid4().hex, 'type': kind, **fields}))

    async def _read(self, ws, incoming):
        # 连接自己的队列，断线后整体丢弃，旧会话的晚到结果不能喂给新段落。
        try:
            async for raw in ws:
                await incoming.put((time.perf_counter(), json.loads(raw)))
            raise ConnectionError('ASR连接已关闭')
        except Exception as exc:
            await incoming.put(exc)

    async def _wait(self, kind):
        while True:
            record = await self.incoming.get()
            if isinstance(record, Exception):
                raise record
            arrived, event = record
            if event.get('type') in ('error', PREFIX + 'failed'):
                error = event.get('error', {})
                raise RuntimeError(str(error.get('message') or error.get('type') or 'ASR服务错误'))
            if event.get('type') == kind:
                return arrived, event

    async def _ready(self):
        if self.reader is not None and self.reader.done():
            await self.close()
        if self.ws is not None:
            return
        async with asyncio.timeout(self.timeout_s):
            self.ws = await self.connector(self.url, additional_headers={'Authorization': f'Bearer {self.key}'},
                                           open_timeout=self.timeout_s, max_size=16 * 1024 * 1024)
            self.usage_meter.reset()
            self.incoming = asyncio.Queue(maxsize=256)
            self.reader = asyncio.create_task(self._read(self.ws, self.incoming), name='serial-asr-receiver')
            await self._wait('session.created')
            await self._send('session.update', session={'input_audio_format': 'pcm', 'sample_rate': 16000,
                             'input_audio_transcription': {'language': 'zh', 'corpus': {'text': CORPUS}},
                             'turn_detection': None})
            _, updated = await self._wait('session.updated')
            if updated.get('session', {}).get('turn_detection') is not None:
                raise RuntimeError('ASR没有进入手动模式')

    async def _transcribe(self, segment):
        if len(segment.pcm) < 3200 or len(segment.pcm) % 2:
            raise ValueError('段落少于100ms或不是PCM16，未发送')
        await self._ready()
        async with asyncio.timeout(self.timeout_s):
            # 开始上传就登记待核对；append之后断线也可能已产生用量。
            await self._notify('uploading', segment.seg_id)
            for start in range(0, len(segment.pcm), 3200):
                await self._send('input_audio_buffer.append', audio=base64.b64encode(
                    segment.pcm[start:start + 3200]).decode('ascii'))
            committed_at = time.perf_counter()
            deadline = asyncio.get_running_loop().time() + self.timeout_s
            await self._send('input_audio_buffer.commit')
            await self._notify('committed', segment.seg_id)
        # 5秒从commit发出开始计，不包含连接握手或本地队列等待。
        async with asyncio.timeout_at(deadline):
            arrived, event = await self._wait(PREFIX + 'completed')
            if arrived < committed_at:
                raise RuntimeError('commit之前出现最终结果，段落关联异常')
            # 即使最终文字为空，也记录服务端实际报告的用量。
            await self._notify('usage', segment.seg_id, self.usage_meter.delta(event.get('usage') or {}))
            text = event.get('transcript')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('ASR返回空转写')
            return text, arrived - committed_at

    async def run(self):
        try:
            while True:
                segment, enqueued_at = await self.queue.get()
                started = time.perf_counter()
                queue_s = started - enqueued_at
                try:
                    text, latency = await self._transcribe(segment)
                except asyncio.CancelledError:
                    await self.emit(AsrEvent('failed', segment.seg_id, '程序停止，本段未完成',
                                             time.perf_counter() - started, queue_s=queue_s))
                    raise
                except Exception as exc:
                    # 任何失败均先断开并清除接收队列；当前段不自动重试，下一段重新握手。
                    await self.close()
                    reason = ('等待ASR超时' if isinstance(exc, TimeoutError) else str(exc)).replace(self.key, '[REDACTED]')
                    await self.emit(AsrEvent('failed', segment.seg_id, reason,
                                             time.perf_counter() - started, queue_s=queue_s))
                else:
                    await self.emit(AsrEvent('transcript', segment.seg_id, text,
                                             time.perf_counter() - started, latency, queue_s))
                finally:
                    self.queue.task_done()
        finally:
            await self.close()
            while not self.queue.empty():
                segment, _ = self.queue.get_nowait()
                try:
                    await self.emit(AsrEvent('failed', segment.seg_id, '程序停止，队列中本段未发送', 0))
                finally:
                    self.queue.task_done()

    async def close(self):
        ws, reader = self.ws, self.reader
        self.ws = self.reader = self.incoming = None
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        if ws is not None:
            with suppress(Exception):
                async with asyncio.timeout(2):
                    await ws.close()
