"""ESPHome 语音卫星（J9）：设备本地听唤醒词，唤醒后通过 API 连接收音，回答也推回设备播放。

一轮的阶段（SatelliteLink.phase）：
- listening：设备听到唤醒词发 start 请求；这里回端口 0（音频走 API 连接），设备推 16 kHz、16 位、单声道 PCM，
  直接喂进现有的 AsrInput.process（Silero 断句 → ASR → Jev → 大脑）。
- awaiting：这句话提交给大脑。设备带扬声器时发 STT_VAD_END：设备停麦、等回答，这一轮不结束；
  Alpha 播报时设备不推流，听不到自己的声音。
- speaking：第一段要播的音频到了，发 TTS_START/TTS_END/TTS_STREAM_START，之后 SatellitePlayer 按实时速度推音频。
  “我查一下”和最终回答在同一个流里。
- 最终回答播完（Receiver 调 player.end_response）发 TTS_STREAM_END + RUN_END，设备放完后回到空闲、重新听唤醒词。
  开了跟进窗口（ALPHA_SATELLITE_FOLLOWUP_SECONDS > 0）时改发 INTENT_END(continue_conversation=1) + TTS_STREAM_END、
  不发 RUN_END：设备放完自动开新的一轮（没有唤醒词），这一轮用跟进超时，不开口就结束、回去听唤醒词。
  不发 RUN_END 是因为设备放完缓冲就会开新的一轮，RUN_END 晚到一步会把新的一轮结束掉。
设备不带扬声器（旧固件）时，提交就发 RUN_END 结束这一轮，回答交给 Mac 播放器。

其他结束方式：唤醒后超过 ALPHA_SATELLITE_LISTEN_SECONDS 没有人声（或说完一直没提交）；等回答超过 90 秒；
设备自己停止。

打断：播报开始时设备重新启动唤醒词检测。一轮进行中又听到唤醒词，设备发事件
esphome.alpha_barge_in（ESPHome 的 voice_assistant.stop 在播放阶段不会停播放，也不通知服务端），这里停掉播报、
结束这一轮；设备放完缓冲（不到 0.5 秒）回到空闲后马上开新的一轮。

取消词：一句话的最后一段是取消词（DISMISS_WORDS：退下、没事了、拜拜、可以了、再见、结束、结束对话、finish、never mind）时以它为准（split_dismissal）。前面没有别的内容就直接结束这一轮、
不交给大脑；前面有请求就照常处理，回答完直接结束、不开跟进窗口。两种情况都调用设备的 dismiss 动作播退出提示音。
上一句回答在问用户时由调用方照常整句交给大脑。

API 连接跨 ASR 重连保留：run() 由外层常驻，feed() 每条 ASR 连接调用一次。
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
import time
import wave

from aioesphomeapi import (APIClient, VoiceAssistantCommandFlag, VoiceAssistantEventType as Event,
                           VoiceAssistantFeature)
import numpy as np

from voice.segmenter import RATE

logger = logging.getLogger('alpha.qwen_alpha')

PORT = 6053
# 一轮结束时补的静音：断句器要 600ms 静音才会结束一段，被设备中途停掉的话也能收尾，不留半截。
FLUSH = bytes(int(RATE * 0.7) * 2)
RESPONSE_TIMEOUT_S = 90.0
BARGE_IN_EVENT = 'esphome.alpha_barge_in'
# 比较时去掉标点和空格、转小写：never mind → nevermind
DISMISS_WORDS = frozenset({'退下', '没事了', '拜拜', '可以了', '再见', '结束', '结束对话', 'finish', 'nevermind'})
# ASR 常把“没事了”转成“没事的”“没事啦”（2026-10-25 实测）
DISMISS_ALIASES = {'没事的': '没事了', '没事啦': '没事了'}
DISMISS_SERVICE = 'dismiss'  # 设备 YAML 里 api.actions 定义的动作：播退出提示音
WAKE_RESIDUE = frozenset({'jarvis', 'heyjarvis'})  # 唤醒词尾巴偶尔会被转写出来
CLAUSE_END = re.compile(r'[；;，,。.！!？?]+\s*')  # 只按标点断句：英文取消词有空格（never mind）


def _clause(text):
    word = re.sub(r'[\W_]+', '', text).lower().removesuffix('吧')
    return DISMISS_ALIASES.get(word, word)


def split_dismissal(text):
    """最后一段是取消词就以它为准，返回（前面要处理的内容，True）；不是返回（原文，False）。

    跟进窗口会把这段时间说的话连成一句（“没有用啊。；退下。”），所以看最后一段，不要求整句只有取消词。
    前面只剩唤醒词的尾巴（“Jarvis。”）也算没有内容。
    """
    clauses = [c for c in CLAUSE_END.split(text) if c]
    if not clauses or _clause(clauses[-1]) not in DISMISS_WORDS:
        return text, False
    rest = text[:text.rfind(clauses[-1])]
    while True:  # 前面连着说了几遍取消词（“拜拜；退下”）也一起去掉
        before = [c for c in CLAUSE_END.split(rest) if c]
        if not before or _clause(before[-1]) not in DISMISS_WORDS:
            break
        rest = rest[:rest.rfind(before[-1])]
    rest = rest.rstrip('；;，,。.！!？? ')
    if all(_clause(c) in WAKE_RESIDUE for c in CLAUSE_END.split(rest) if c):
        rest = ''
    return rest, True


class _RunEnded:
    """队列里的分隔符：前面的音频属于已经结束的那一轮。"""


class SatelliteLink:
    def __init__(self, host, *, port=PORT, noise_psk=None, listen_s=8.0, followup_s=6.0, client_factory=APIClient,
                 clock=time.monotonic, dump_dir=None):
        self.host, self.port, self.noise_psk = host, port, noise_psk
        self.listen_s = listen_s
        self.followup_s = followup_s  # 回答播完后不用唤醒词接着说的窗口；0 表示关闭
        self.followup = False  # 这一轮是不是跟进
        self.followup_until = 0.0  # 在这之前设备发来的无唤醒词 start 算跟进
        self.client_factory, self.clock = client_factory, clock
        self.client = None
        self.speaker = False  # 设备声明了 SPEAKER 能力才把回答推给它
        self.queue = asyncio.Queue()
        self.phase = 'idle'
        self.gain = 1.0  # 设备在 start 请求里给的 volume_multiplier
        self.started_at = self.last_voice = self.awaiting_since = 0.0
        self.receiver = None  # feed() 设置；唤醒或设备停止时用来打断正在进行的播报
        self.tasks = set()
        self.dismiss_service = None  # 设备的 dismiss 动作；旧固件没有就只结束这一轮、不播提示音
        self.dismiss_after_answer = False  # 这一轮说了“……拜拜”：回答完直接结束、播退出提示音
        self.dump_dir = Path(dump_dir) if dump_dir else None  # 调试：每一轮的原始音频（放大前）存成 WAV
        self.recording = bytearray()

    async def run(self):
        """保持和设备的 API 连接，断线 5 秒后重连；连接期间订阅语音助手。"""
        while True:
            stopped = asyncio.Event()

            async def on_stop(expected):
                stopped.set()

            client = self.client_factory(self.host, self.port, None, client_info='alpha',
                                         noise_psk=self.noise_psk)
            try:
                await client.connect(on_stop=on_stop, login=True)
                info = await client.device_info()
                self.client = client
                self.speaker = bool(info.voice_assistant_feature_flags & VoiceAssistantFeature.SPEAKER)
                _, services = await client.list_entities_services()
                self.dismiss_service = next((s for s in services if s.name == DISMISS_SERVICE), None)
                unsubscribe = client.subscribe_voice_assistant(
                    handle_start=self.handle_start, handle_stop=self.handle_stop, handle_audio=self.handle_audio)
                client.subscribe_service_calls(self.handle_service_call)
                logger.info('[卫星] 已连接 %s（%s，ESPHome %s）｜ 播报：%s', info.name, self.host,
                            info.esphome_version, '设备扬声器' if self.speaker else 'Mac 扬声器')
                try:
                    await stopped.wait()
                finally:
                    unsubscribe()
                logger.warning('[卫星] 连接断开，5 秒后重连')
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning('[卫星] 连接 %s 失败：%s；5 秒后重试', self.host, type(exc).__name__)
            finally:
                self.client = None
                self._finish('disconnected', notify=False)
                await client.disconnect()
            await asyncio.sleep(5)

    async def handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        # 设备在回答播完后自动开的一轮没有唤醒词；诊断按钮开的也没有，但不会紧跟在回答后面。
        # 不能看 wake_word_phrase：设备自动开的跟进会沿用上一次的唤醒词名字（2026-10-25 实测）。看请求标记：
        # 唤醒词触发的一轮带 USE_WAKE_WORD，设备自动开的跟进和诊断按钮都不带；再要求紧跟在回答之后。
        self.followup = not flags & VoiceAssistantCommandFlag.USE_WAKE_WORD and self.clock() <= self.followup_until
        self.followup_until = 0.0
        self.dismiss_after_answer = False
        if self.followup:
            logger.info('[卫星] 跟进：不用唤醒词，%.0f 秒内开口就接着聊', self.followup_s)
        else:
            logger.info('[卫星] 唤醒：%s', wake_word_phrase or '（无唤醒词）')
        await self._interrupt_playback('唤醒打断当前播报')
        self._finish('restarted', notify=False)
        # 设备请求的放大倍数（ESPHome voice_assistant.volume_multiplier）。XVF3800 右声道是 ASR 波束，处理轻、
        # 播放中也认得出唤醒词，但没有自动增益，比左声道小约 25 dB，所以在设备配置里要求放大。
        multiplier = getattr(audio_settings, 'volume_multiplier', 1.0) or 1.0
        self.gain = multiplier if 0 < multiplier <= 100 else 1.0
        self.phase = 'listening'
        self.recording = bytearray()
        self.started_at = self.last_voice = self.clock()
        self._event(Event.VOICE_ASSISTANT_RUN_START)
        self._event(Event.VOICE_ASSISTANT_STT_START)
        return 0  # 0：音频走 API 连接，不另开 UDP

    async def handle_audio(self, data, data2=None):
        # 不能有 await 在 put 之前：每块音频是一个后台任务，入队顺序就是到达顺序。
        if self.phase == 'listening':
            if self.dump_dir is not None:
                self.recording += data
            if self.gain != 1.0:
                pcm = np.frombuffer(data[:len(data) - len(data) % 2], '<i2').astype(np.float32) * self.gain
                data = np.clip(pcm, -32768, 32767).astype('<i2').tobytes()
            self.queue.put_nowait(data)

    async def handle_stop(self, server_side):
        if self.phase == 'idle':
            return
        # 设备自己停的（多半是播报中再次唤醒），马上会开新的一轮：不回 RUN_END，免得结束掉新的那一轮。
        await self._interrupt_playback('设备停止，停掉播报')
        self._finish('device', notify=False)

    def handle_service_call(self, call):
        if call.is_event and call.service == BARGE_IN_EVENT:
            task = asyncio.create_task(self.barge_in(), name='satellite-barge-in')
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def barge_in(self):
        """一轮进行中又听到唤醒词：停掉这一轮，设备回到空闲后会马上开新的一轮。"""
        if self.phase == 'idle':
            return
        logger.info('[卫星] 一轮进行中再次唤醒（%s），结束这一轮', self.phase)
        await self._interrupt_playback('唤醒打断当前播报')
        if self.phase == 'listening':
            self.end_run('restarted')
        else:
            self.end_response('barge-in')

    async def _interrupt_playback(self, why):
        receiver = self.receiver
        if receiver is not None and (receiver.active or receiver.current_report is not None):
            logger.info('[卫星] %s', why)
            await receiver.interrupt()

    def end_listening(self, reason):
        """这句话提交给大脑：设备停麦。带扬声器时这一轮继续，等回答播完才结束。"""
        if self.phase != 'listening':
            return
        if not self.speaker:
            self._finish(reason)
            return
        self.queue.put_nowait(_RunEnded())
        self._dump(reason)
        self._event(Event.VOICE_ASSISTANT_STT_VAD_END)
        self.phase, self.awaiting_since = 'awaiting', self.clock()
        logger.info('[卫星] 收音结束：%s（%.1f 秒），等回答', reason, self.clock() - self.started_at)

    def start_response(self):
        """要播音频了：第一次调用时让设备进入播放状态。返回设备现在能不能播。"""
        if self.phase == 'awaiting':
            # TTS_START 要有文字、TTS_END 要有 url，设备才进入播放状态；音频本身走 API，url 不会被使用。
            self._event(Event.VOICE_ASSISTANT_TTS_START, {'text': 'Alpha'})
            self._event(Event.VOICE_ASSISTANT_TTS_END, {'url': 'api'})
            self._event(Event.VOICE_ASSISTANT_TTS_STREAM_START)
            self.phase = 'speaking'
        return self.phase == 'speaking'

    def send_audio(self, data):
        if self.client is not None and self.phase == 'speaking':
            self.client.send_voice_assistant_audio(data)

    def end_response(self, reason='done'):
        """最终回答播完：关流并结束这一轮。开了跟进窗口时让设备放完后自动开新的一轮，否则回去听唤醒词。"""
        if self.phase in ('awaiting', 'speaking') and self.dismiss_after_answer:
            if self.phase == 'speaking':
                self._event(Event.VOICE_ASSISTANT_TTS_STREAM_END)
            self._finish('回答完退下')
            self._play_exit_beep(delay=0.6)  # 等设备放完缓冲里的回答，提示音不和回答叠在一起
            return
        if self.phase == 'speaking' and reason == 'done' and self.followup_s > 0:
            self._event(Event.VOICE_ASSISTANT_INTENT_END, {'continue_conversation': '1'})
            self._event(Event.VOICE_ASSISTANT_TTS_STREAM_END)
            self._finish('done，等跟进', notify=False)
            self.followup_until = self.clock() + 5.0  # 设备放完缓冲（不到 1 秒）就会发 start
            return
        if self.phase == 'speaking':
            self._event(Event.VOICE_ASSISTANT_TTS_STREAM_END)
        if self.phase in ('awaiting', 'speaking'):
            self._finish(reason)

    def dismiss(self):
        """取消词：结束这一轮（不开跟进窗口），让设备播退出提示音。不在收音时返回 False。"""
        if self.phase != 'listening':
            return False
        self.end_run('退下')
        self._play_exit_beep()
        return True

    def _play_exit_beep(self, delay=0.0):
        if self.client is None or self.dismiss_service is None:
            return
        client, service = self.client, self.dismiss_service

        async def call():
            await asyncio.sleep(delay)
            await client.execute_service(service, {})

        task = asyncio.create_task(call(), name='satellite-dismiss')
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def end_run(self, reason):
        if self.phase != 'idle':
            self._finish(reason)

    def _finish(self, reason, *, notify=True):
        if self.phase == 'idle':
            return
        if self.phase == 'listening':
            self.queue.put_nowait(_RunEnded())
            self._dump(reason)
        self.phase = 'idle'
        if notify:
            self._event(Event.VOICE_ASSISTANT_RUN_END)
        logger.info('[卫星] 本轮结束：%s（%.1f 秒）', reason, self.clock() - self.started_at)

    def _dump(self, reason):
        if self.dump_dir is None or not self.recording:
            return
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        path = self.dump_dir / f"{time.strftime('%H%M%S')}-{reason}.wav"
        with wave.open(str(path), 'wb') as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            out.writeframes(bytes(self.recording))
        logger.info('[卫星] 调试：这一轮的原始音频存到 %s', path)
        self.recording = bytearray()

    def _event(self, kind, data=None):
        if self.client is not None:
            self.client.send_voice_assistant_event(kind, data)

    async def feed(self, adapter):
        """把设备推来的音频交给这条 ASR 连接的断句器；太久没有人声、太久没有回答都结束这一轮。"""
        self.receiver = adapter.receiver
        try:
            while True:
                # 不用 wait_for：3.11 里取消恰好和 get() 拿到数据同时发生时，wait_for 会吞掉取消，
                # 这个循环就再也停不下来（单元测试里实际卡住过）。asyncio.timeout 能区分超时和取消。
                try:
                    async with asyncio.timeout(0.5):
                        item = await self.queue.get()
                except TimeoutError:
                    item = None
                if isinstance(item, _RunEnded):
                    await adapter.process(FLUSH)
                elif item is not None:
                    await adapter.process(item)
                    if adapter.segmenter.active is not None:
                        self.last_voice = self.clock()
                now = self.clock()
                limit = self.followup_s if self.followup else self.listen_s
                if self.phase == 'listening' and now - self.last_voice >= limit:
                    self.end_run('跟进超时' if self.followup else 'timeout')
                    self._play_exit_beep()  # 超时结束也响一声（唤醒后没开口、跟进窗口结束），听得出它不再听了
                elif self.phase == 'awaiting' and now - self.awaiting_since >= RESPONSE_TIMEOUT_S:
                    self.end_run('no response')
        finally:
            self.receiver = None


class Resampler:
    """PCM16 单声道 24 kHz → 16 kHz（插 2 抽 3，加窗 sinc 低通）；跨块保留滤波器状态，块边界不爆音。"""
    TAPS = 97

    def __init__(self):
        n = np.arange(self.TAPS) - (self.TAPS - 1) / 2
        cutoff = 7200 / 48000  # 截止 7.2 kHz（新采样率的奈奎斯特是 8 kHz），按 48 kHz 中间采样率归一
        # 乘 2：插零让能量减半，补回来
        self.taps = (2 * cutoff * np.sinc(2 * cutoff * n) * np.hanning(self.TAPS) * 2).astype(np.float32)
        self.reset()

    def reset(self):
        self.history = np.zeros(self.TAPS - 1, np.float32)
        self.phase = 0
        self.odd = b''

    def __call__(self, pcm):
        pcm = self.odd + pcm
        cut = len(pcm) - len(pcm) % 2
        pcm, self.odd = pcm[:cut], pcm[cut:]
        x = np.frombuffer(pcm, '<i2').astype(np.float32)
        up = np.zeros(len(x) * 2, np.float32)
        up[::2] = x
        buffered = np.concatenate([self.history, up])
        filtered = np.convolve(buffered, self.taps, mode='valid')
        self.history = buffered[len(buffered) - (self.TAPS - 1):]
        out = filtered[self.phase::3]
        self.phase = (self.phase - len(filtered)) % 3
        return np.clip(np.round(out), -32768, 32767).astype('<i2').tobytes()


@dataclass
class _Item:
    route: str  # device：推给卫星；mac：交给 Mac 播放器；drop：设备现在不能播（不在一轮里）
    queued: int = 0  # 还没推出去的块数
    finished: bool = False  # finish() 之后不会再有音频
    cleared: bool = False
    start_at: float | None = None
    end_at: float | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)


class SatellitePlayer:
    """卫星带扬声器时把播报推给设备，否则交给 Mac 播放器；接口和 StreamingPlayer 一样。

    设备的语音助手缓冲只有约 0.5 秒（16 KB），所以按实时速度推，最多领先 LEAD 秒；
    播放开始/结束时间按推送进度估算（设备不回报播放进度）。
    """
    IN_RATE, OUT_RATE = 24000, 16000
    CHUNK = 1024  # 32 ms
    LEAD = 0.4
    MARGIN = 0.15  # 估算结束后再等一点，覆盖设备内部的缓冲和延迟

    def __init__(self, link, fallback, *, clock=time.perf_counter):
        self.link, self.fallback, self.clock = link, fallback, clock
        self.items: dict[str, _Item] = {}
        self.pending: deque[tuple[str, bytes]] = deque()
        self.resampler = Resampler()
        self.device_end = 0.0
        self.paused = False
        self.wakeup = asyncio.Event()

    @property
    def gain(self):
        return self.fallback.gain

    @gain.setter
    def gain(self, value):
        self.fallback.gain = value  # 设备音量由设备自己管，这里只影响 Mac 播放

    def _item(self, item_id):
        item = self.items.get(item_id)
        if item is None:
            if not self.link.speaker:
                route = 'mac'
            elif self.link.start_response():
                route = 'device'
            else:
                route = 'drop'
                logger.info('[卫星] 不在一轮对话里，设备无法播放，跳过这段音频')
            item = self.items[item_id] = _Item(route)
            if route == 'drop':
                item.started.set()
                item.done.set()
            while len(self.items) > 200:
                self.items.pop(next(iter(self.items)))
        return item

    def append(self, audio, item_id, content_index):
        item = self._item(item_id)
        if item.route == 'mac':
            self.fallback.append(audio, item_id, content_index)
        elif item.route == 'device' and not item.cleared:
            pcm = self.resampler(audio)
            for offset in range(0, len(pcm), self.CHUNK):
                self.pending.append((item_id, pcm[offset:offset + self.CHUNK]))
                item.queued += 1
            self.wakeup.set()

    def finish(self, item_id):
        item = self._item(item_id)
        if item.route == 'mac':
            self.fallback.finish(item_id)
        else:
            item.finished = True
            self.wakeup.set()

    async def wait_played(self, item_id):
        item = self.items.get(item_id)
        if item is None or item.route == 'mac':
            return await self.fallback.wait_played(item_id)
        await item.done.wait()
        return item.route == 'device' and not item.cleared

    async def wait_started(self, item_id):
        item = self.items.get(item_id)
        if item is None or item.route == 'mac':
            return await self.fallback.wait_started(item_id)
        await item.started.wait()
        return item.start_at

    def playback_finished(self, item_id):
        item = self.items.get(item_id)
        if item is None or item.route == 'mac':
            return self.fallback.playback_finished(item_id)
        return item.done.is_set() and item.route == 'device' and not item.cleared

    def clear(self):
        for item in self.items.values():
            if item.route == 'device' and not item.done.is_set():
                item.cleared = True
                item.started.set()
                item.done.set()
        self.pending.clear()
        self.resampler.reset()
        return self.fallback.clear()

    def pause(self):
        self.paused = True
        self.fallback.pause()

    def resume(self):
        self.paused = False
        self.fallback.resume()
        self.wakeup.set()

    def end_response(self):
        """Receiver 在一轮的最终回答播完后调用：结束设备上的这一轮。"""
        self.link.end_response()

    async def run(self):
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(self.fallback.run())
            tasks.create_task(self._pump())

    async def _pump(self):
        while True:
            now = self.clock()
            if self.pending and not self.paused and self.device_end - now < self.LEAD:
                item_id, chunk = self.pending.popleft()
                item = self.items[item_id]
                start = max(now, self.device_end)
                self.device_end = start + len(chunk) / 2 / self.OUT_RATE
                if item.start_at is None:
                    item.start_at = start
                    item.started.set()
                item.end_at = self.device_end
                item.queued -= 1
                self.link.send_audio(chunk)
                continue
            wait = 0.5
            for item in self.items.values():
                if item.route != 'device' or item.done.is_set() or not item.finished or item.queued:
                    continue
                due = (item.end_at if item.end_at is not None else now) + self.MARGIN
                if now >= due:
                    item.started.set()
                    item.done.set()
                else:
                    wait = min(wait, due - now)
            if self.pending and not self.paused:
                wait = min(wait, max(0.005, self.device_end - self.LEAD - now))
            self.wakeup.clear()
            try:
                async with asyncio.timeout(wait):
                    await self.wakeup.wait()
            except TimeoutError:
                pass
