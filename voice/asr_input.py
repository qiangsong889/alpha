"""Silero 本地事件适配到现有 Receiver；不改变 TurnCollector 的提交规则。"""

import asyncio
import logging
from uuid import uuid4

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from core.costs import record_cost
from core.settings import setting
from voice.audio import audio_device
from voice.audio_devices import AudioRoute
from voice.asr_client import MODEL, PREFIX
from voice.segmenter import FRAME, RATE, SpeechStarted
from voice.omni_transport import SessionEnded

logger = logging.getLogger('alpha.qwen_alpha')


class SileroActivity:
    """给现有的 400ms 暂停确认提供 Silero 人声时长，不再运行音量门控。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.voiced_ms = 0.0


class AsrInput:
    def __init__(self, receiver, segmenter, *, region='singapore', route=None):
        self.receiver, self.segmenter = receiver, segmenter
        self.region = region
        # 系统默认设备变化时由 route 协调关闭和重开麦克风；默认不跟随。
        self.route = route if route is not None else AudioRoute(follow_input=False, follow_output=False)
        self.client = None
        self.hold_id = None

    async def local_event(self, event):
        receiver = self.receiver
        if isinstance(event, SpeechStarted):
            await receiver.handle({'type': 'input_audio_buffer.speech_started', 'item_id': event.seg_id})
            if receiver.activity is not None:
                receiver.activity.voiced_ms = (event.confirmed_s - event.speech_start_s) * 1000
            logger.info('[本地ASR] 开始 %s%s', event.seg_id, '（长句续段）' if event.continuation else '')
        else:
            await receiver.handle({'type': 'input_audio_buffer.speech_stopped', 'item_id': event.seg_id})
            if event.reason == 'max_duration' and self.hold_id is None:
                # 15秒只是上传分片。保持一个本地占位直到真正静音，不能提前向大脑提交。
                self.hold_id = 'continuation_' + uuid4().hex
                receiver.collector.started(self.hold_id)
            logger.info('[本地ASR] 结束 %s ｜ %s', event.seg_id, event.reason)
            await self.client.submit(event)

    async def notify(self, kind, seg_id, payload=None):
        if kind == 'uploading':
            record_cost('ASR', MODEL, event_id='asr-segment:' + seg_id,
                        region=self.region, reason='request_started_no_final_usage')
        elif kind == 'committed':
            # 只在 commit 已发送后通知；从登记到结果到达始终由 started 的 pending 保护。
            await self.receiver.handle({'type': 'input_audio_buffer.committed', 'item_id': seg_id})
        elif kind == 'usage':
            record_cost('ASR', MODEL, payload, event_id='asr-segment:' + seg_id, region=self.region)

    async def result(self, event):
        if event.kind == 'failed':
            logger.warning('[ASR失败] %s：%s', event.seg_id, event.text)
        else:
            logger.info('[ASR转写] %s ｜ 排队 %.3fs ｜ commit→文字 %s ｜ 语言 %s', event.seg_id,
                        event.queue_s, event.commit_to_result_s, event.language or '未知')
        # 失败也必须关闭这段的 pending；下一段不受影响。
        await self.receiver.handle({'type': PREFIX + 'completed', 'item_id': event.seg_id,
                                   'transcript': event.text if event.kind == 'transcript' else ''})

    async def process(self, pcm):
        events = await asyncio.to_thread(self.segmenter.feed, pcm)
        for event in events:
            await self.local_event(event)
        activity = self.receiver.activity
        if activity is not None and self.segmenter.last_probability >= self.segmenter.threshold:
            activity.voiced_ms += len(pcm) / (RATE * 2) * 1000
        if self.hold_id is not None:
            if self.segmenter.active is None and not self.segmenter.continuing:
                ident, self.hold_id = self.hold_id, None
                self.receiver.collector.stopped(ident)
                self.receiver.collector.transcript(ident, '')
                self.receiver.restore_if_quiet()
            else:
                # 续段自己的 stopped 不能释放仍在说话的占位。
                self.receiver.collector.started(self.hold_id)

    async def microphone(self):
        """声卡回调仅复制入队；ONNX 推理和网络发送都在异步任务里运行。"""
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=256)  # 约8秒，积压时重建本地输入，不上传陈旧音频。
        accepting = True

        def enqueue(pcm, status):
            if not accepting:
                return
            if status:
                logger.warning('[麦克风] %s', status)
            if queue.full():
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(SessionEnded('本地音频积压超过8秒，重新启动收音'))
            else:
                queue.put_nowait(pcm)

        # 多声道设备只取指定的一个声道（例如 XVF3800 的语音识别波束），其余丢弃。
        channel = setting('REALTIME_INPUT_CHANNEL')

        def callback(indata, frames, timing, status):
            pcm = bytes(indata) if channel == 1 else \
                np.frombuffer(indata, dtype='<i2').reshape(-1, channel)[:, channel - 1].tobytes()
            loop.call_soon_threadsafe(enqueue, pcm, str(status) if status else '')

        try:
            while True:
                # 切换设备进行中时先等它完成，不在重新初始化 PortAudio 前后打开流。
                await self.route.wait_ready()
                device = audio_device('REALTIME_INPUT_DEVICE')
                info = sd.query_devices(device, 'input')
                available = info['max_input_channels']
                if channel > available:
                    # 配置错误：不能悄悄改用别的声道，也不该当成断线反复重连。
                    raise ValueError(f'REALTIME_INPUT_CHANNEL={channel}，但麦克风设备只有 {available} 个声道')
                with self.route.stream(), sd.RawInputStream(
                        samplerate=RATE, channels=channel, dtype='int16', blocksize=FRAME,
                        device=device, callback=callback) as stream:
                    logger.info('[麦克风] %s ｜ 第 %d 声道（共 %d）', info.get('name', device), channel, available)
                    while not self.route.changed.is_set():
                        try:
                            chunk = await asyncio.wait_for(queue.get(), .2)
                        except TimeoutError:
                            if not stream.active:
                                raise SessionEnded('麦克风已断开')
                            continue
                        if isinstance(chunk, Exception):
                            raise chunk
                        await self.process(chunk)
                # 系统默认设备变了：上面正常关闭了流，回到开头用新设备重新打开，ASR 会话不断。
                logger.info('[麦克风] 系统默认设备变化，重新打开')
        except sd.PortAudioError as exc:
            raise SessionEnded('麦克风不可用，请检查设备连接') from exc
        finally:
            accepting = False
