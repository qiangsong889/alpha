"""A1：独立验证 ASR 手动模式，不启动麦克风、大脑或任何业务工具。

运行（alpha 目录下）：
    .venv/bin/python -m devtools.asr_probe first.wav second.wav
也支持 .venv/bin/python devtools/asr_probe.py first.wav second.wav。
连续提交探测：加 --back-to-back，以100ms音频块尽快上传，不等第一段结果就发送第二段。

读取项目 .env 的 DASHSCOPE_API_KEY、QWEN_WORKSPACE_ID、QWEN_REGION。
默认按真实速度上传，每块 100ms；同一连接依次 commit，最后故意空 commit。
中间结果可能先于 commit 到达；延迟为负数时，表示提前于 commit 的秒数。
协议依据：
https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-client-events
https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-server-events
"""

import argparse
import asyncio
import base64
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from uuid import uuid4
import wave

from dotenv import load_dotenv
from websockets.asyncio.client import connect

MODEL = 'qwen3-asr-flash-realtime'
PREFIX = 'conversation.item.input_audio_transcription.'
BYTES_PER_SECOND = 16000 * 2
CHUNK_BYTES = BYTES_PER_SECOND // 10


def read_wav(path: Path) -> bytes:
    """只读 WAV 数据帧，不把 RIFF 文件头作为 PCM 发送。"""
    with wave.open(str(path), 'rb') as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (16000, 1, 2, 'NONE'):
            raise ValueError(f'{path.name} 必须是 16kHz、单声道、PCM16 WAV')
        frames = audio.getnframes()
        pcm = audio.readframes(frames)
    if len(pcm) != frames * 2:
        raise ValueError(f'{path.name} 的 WAV 数据被截断')
    if len(pcm) < CHUNK_BYTES:
        raise ValueError(f'{path.name} 至少需要 100ms 音频；空 commit 由脚本最后单独测试')
    return pcm


def websocket_url(workspace: str, region: str) -> str:
    regions = {'singapore': 'ap-southeast-1', 'beijing': 'cn-beijing'}
    if region not in regions or not re.fullmatch(r'[A-Za-z0-9-]+', workspace):
        raise ValueError('需要有效的 QWEN_WORKSPACE_ID 和 QWEN_REGION=singapore|beijing')
    return f'wss://{workspace}.{regions[region]}.maas.aliyuncs.com/api-ws/v1/realtime?' + urlencode({'model': MODEL})


@dataclass
class Round:
    label: str
    commit_at: float | None
    event_id: str
    item_id: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


class Probe:
    def __init__(self, ws, *, timeout_s=30, redact=()):
        self.ws, self.timeout_s = ws, timeout_s
        self.redact = tuple(value for value in redact if value)
        self.started = time.perf_counter()
        self.current: Round | None = None
        self.items: dict[str, Round] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.back_to_back = False
        self.commits: list[Round] = []
        self.completed: list[dict] = []

    def log(self, text):
        for secret in self.redact:
            text = text.replace(secret, '[REDACTED]')
        print(text, flush=True)

    async def send(self, kind, *, event_id=None, **fields):
        await self.ws.send(json.dumps({'event_id': event_id or uuid4().hex, 'type': kind, **fields}))

    async def receive(self):
        """唯一的 recv 调用方：上传期间也持续接收，在到达时计时和打印。"""
        try:
            async for message in self.ws:
                received_at = time.perf_counter()
                event = json.loads(message)
                kind = event.get('type', '[missing type]')
                item_id = event.get('item_id') or event.get('item', {}).get('id')
                # 并发探测不能按“最新发送段落”或FIFO猜测归属；只记录原始到达次序。
                current = None if self.back_to_back else self.items.get(item_id, self.current)
                if kind == 'input_audio_buffer.committed' and current is not None:
                    current.item_id = item_id
                    if item_id:
                        self.items[item_id] = current
                stamp = (f'连续提交 连接+{received_at - self.started:.3f}s' if self.back_to_back else
                         f'{current.label} commit+{received_at - current.commit_at:.3f}s'
                         if current is not None and current.commit_at is not None else
                         f'{current.label if current else "配置"} 未commit 连接+{received_at - self.started:.3f}s')
                # 完整打印每个事件，包括 text/stash/transcript、usage 和服务端错误。
                self.log(f'[{stamp}] {kind} item_id={item_id or "-"} '
                         + json.dumps(event, ensure_ascii=False))
                if kind in ('input_audio_buffer.speech_started', 'input_audio_buffer.speech_stopped') or (
                        kind == 'input_audio_buffer.committed' and
                        (not self.commits if self.back_to_back else current is None or current.commit_at is None)):
                    raise RuntimeError('手动 commit 之前收到自动分段事件，停止探测')
                if kind == PREFIX + 'completed':
                    self.completed.append({'received_at': received_at, 'item_id': item_id,
                                           'transcript': event.get('transcript'),
                                           'commits_sent': len(self.commits)})
                if current is not None and kind in ('input_audio_buffer.committed', PREFIX + 'text', PREFIX + 'completed'):
                    # 保存接收绝对时刻；中间结果可能早于 commit，最后计算负偏移。
                    current.timings.setdefault(kind, received_at)
                await self.queue.put(event)
                if kind == 'session.finished':
                    return
            raise ConnectionError('ASR 连接已关闭')
        except Exception as exc:
            self.log(f'[接收结束] {type(exc).__name__}: {exc}')
            await self.queue.put(exc)

    async def wait(self, kind, *, item_id=None, allow_error=False):
        async with asyncio.timeout(self.timeout_s):
            while True:
                event = await self.queue.get()
                if isinstance(event, Exception):
                    raise event
                if event.get('type') == 'error' or event.get('type') == PREFIX + 'failed':
                    if allow_error:
                        return event
                    raise RuntimeError('ASR 返回错误，详见上方事件')
                if event.get('type') == kind and (item_id is None or event.get('item_id') == item_id):
                    return event

    async def commit(self, label):
        current = self.current
        if current is None or current.label != label or current.commit_at is not None:
            current = Round(label, None, uuid4().hex)
        current.commit_at = time.perf_counter()
        self.current = current
        self.commits.append(current)
        self.log(f'[{label} commit+0.000s] 发送 input_audio_buffer.commit event_id={current.event_id}')
        await self.send('input_audio_buffer.commit', event_id=current.event_id)
        return current

    async def upload(self, label, pcm, *, pace=True):
        if not self.back_to_back:
            self.current = Round(label, None, uuid4().hex)
        self.log(f'[{label}] 上传 {len(pcm) / BYTES_PER_SECOND:.2f}s PCM，100ms/块')
        upload_started = time.perf_counter()
        for offset in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[offset:offset + CHUNK_BYTES]
            await self.send('input_audio_buffer.append', audio=base64.b64encode(chunk).decode('ascii'))
            # 按采样时长模拟实时输入，防止发送本身的耗时累积为额外等待。
            remaining = upload_started + (offset + len(chunk)) / BYTES_PER_SECOND - time.perf_counter()
            if pace and remaining > 0:
                await asyncio.sleep(remaining)

    async def transcribe(self, label, pcm):
        # 每段之间只提交，不发送 session.finish，不关闭连接。
        await self.upload(label, pcm)
        current = await self.commit(label)
        committed = await self.wait('input_audio_buffer.committed')
        if not committed.get('item_id'):
            self.log(f'[{label}] committed 未提供 item_id；本探测串行发送，按本地段号关联，原字段保持 -')
        if PREFIX + 'completed' not in current.timings:
            await self.wait(PREFIX + 'completed', item_id=committed.get('item_id'))
        for name, kind in (('committed', 'input_audio_buffer.committed'),
                           ('首个中间结果', PREFIX + 'text'), ('completed', PREFIX + 'completed')):
            received_at = current.timings.get(kind)
            assert current.commit_at is not None
            self.log(f'[{label} 延迟] commit → {name}：'
                     + (f'{received_at - current.commit_at:.3f}s' if received_at is not None else '未收到'))

    async def consecutive(self, clips):
        self.log('[连续提交] 100ms音频/块、无实时等待；不等待 committed/completed，制造两个在途提交')
        for index, pcm in enumerate(clips, 1):
            await self.upload(f'第{index}段', pcm, pace=False)
            await self.commit(f'第{index}段')
        try:
            for _ in clips:
                await self.wait(PREFIX + 'completed')
        finally:
            # 即使缺少第二个结果而超时，也保留已经到达的文字；不能只汇报成功轮次。
            self.report_consecutive(len(clips))

    def report_consecutive(self, expected):
        self.log(f'[completed数量] {len(self.completed)}/{expected}')
        second_commit = self.commits[1].commit_at
        assert second_commit is not None
        overlapped = bool(self.completed and self.completed[0]['received_at'] > second_commit)
        self.log('[并发验证] 第二个commit发出时尚未收到第一个completed：' + str(overlapped))
        if not overlapped:
            self.log('[注意] 没有形成两个未完成提交，不能用本次证明并发返回顺序')
        for index, result in enumerate(self.completed, 1):
            offsets = []
            for commit in self.commits[:2]:
                assert commit.commit_at is not None
                offsets.append(f'{commit.label}commit后 {result["received_at"] - commit.commit_at:.3f}s')
            self.log(f'[completed到达顺序 #{index}] ' + ' ｜ '.join(offsets)
                     + ' ｜ ' + json.dumps(result, ensure_ascii=False))
        self.log('[顺序判断] 根据上述原始transcript区分长句/短句；未按FIFO自动标注所属段落')

    async def run(self, clips, *, back_to_back=False):
        self.back_to_back = back_to_back
        receiver = asyncio.create_task(self.receive(), name='asr-probe-receiver')
        try:
            created = await self.wait('session.created')
            self.log(f'[连接] {MODEL} session_id={created["session"]["id"]}；两段共用此连接')
            await self.send('session.update', session={
                'input_audio_format': 'pcm', 'sample_rate': 16000,
                'input_audio_transcription': {'language': 'zh'}, 'turn_detection': None,
            })
            updated = await self.wait('session.updated')
            if updated.get('session', {}).get('turn_detection') is not None:
                raise RuntimeError('服务端未确认手动模式，停止探测')
            if 'turn_detection' not in updated.get('session', {}):
                self.log('[配置] 服务端省略 turn_detection=null；继续验证手动 commit，无自动分段事件')
            if back_to_back:
                await self.consecutive(clips)
            else:
                for index, pcm in enumerate(clips, 1):
                    await self.transcribe(f'第{index}段', pcm)
            self.log('[连续提交结束] 请按原始文字核对段落边界，不能仅凭数量认定FIFO' if back_to_back else
                     '[复用成功] 两段均已收到 completed，期间没有重连')
            await self.commit('空缓冲区')
            try:
                result = await self.wait('input_audio_buffer.committed', allow_error=True)
                if result['type'] in ('error', PREFIX + 'failed'):
                    self.log('[空commit结果] ' + json.dumps(result.get('error', {}), ensure_ascii=False))
                else:
                    self.log('[空commit结果] 服务端接受了空提交，详见事件；不预设它一定报错')
            except TimeoutError:
                self.log('[空commit结果] 超时，没有收到错误或 committed')
                raise
            await self.send('session.finish')
            await self.wait('session.finished')
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('wav', nargs=2, type=Path, help='两个 16kHz 单声道 PCM16 WAV，可重复指定同一文件')
    parser.add_argument('--timeout', type=float, default=30, help='每阶段等待超时秒数，默认30')
    parser.add_argument('--back-to-back', action='store_true',
                        help='100ms音频块无间隔上传，连续commit，不等前一段结果；按到达次序打印最终文字')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout 必须是大于零的有限数字')
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    key, workspace = os.getenv('DASHSCOPE_API_KEY', '').strip(), os.getenv('QWEN_WORKSPACE_ID', '').strip()
    try:
        clips = [read_wav(path) for path in args.wav]
        if not key:
            raise ValueError('请在 alpha/.env 中配置 DASHSCOPE_API_KEY')
        url = websocket_url(workspace, os.getenv('QWEN_REGION', 'singapore'))
        async with connect(url, additional_headers={'Authorization': f'Bearer {key}', 'OpenAI-Beta': 'realtime=v1'},
                           open_timeout=15, max_size=16 * 1024 * 1024) as ws:
            await Probe(ws, timeout_s=args.timeout, redact=(key, workspace)).run(clips, back_to_back=args.back_to_back)
        return 0
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        for secret in (key, workspace):
            if secret:
                message = message.replace(secret, '[REDACTED]')
        print('[探测失败] ' + message, flush=True)
        return 1


if __name__ == '__main__':
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print('\n已退出')
