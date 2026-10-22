"""Silero 本地事件适配到现有 Receiver；不改变 TurnCollector 的提交规则。"""

import asyncio
import logging
from uuid import uuid4

import sounddevice as sd  # type: ignore[import-untyped]

from core.costs import record_cost
from voice.audio import audio_device
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
    def __init__(self, receiver, segmenter, *, region='singapore'):
        self.receiver, self.segmenter = receiver, segmenter
        self.region = region
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
            logger.info('[ASR转写] %s ｜ 排队 %.3fs ｜ commit→文字 %s', event.seg_id,
                        event.queue_s, event.commit_to_result_s)
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

        def callback(indata, frames, timing, status):
            loop.call_soon_threadsafe(enqueue, bytes(indata), str(status) if status else '')

        try:
            with sd.RawInputStream(samplerate=RATE, channels=1, dtype='int16', blocksize=FRAME,
                                   device=audio_device('REALTIME_INPUT_DEVICE'), callback=callback) as stream:
                while True:
                    try:
                        chunk = await asyncio.wait_for(queue.get(), .2)
                    except TimeoutError:
                        if not stream.active:
                            raise SessionEnded('麦克风已断开')
                        continue
                    if isinstance(chunk, Exception):
                        raise chunk
                    await self.process(chunk)
        except sd.PortAudioError as exc:
            raise SessionEnded('麦克风不可用，请检查设备连接') from exc
        finally:
            accepting = False
