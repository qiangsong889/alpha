"""与 API 无关的录音设备选择、播放、VAD 和语音有效性判断。"""

import asyncio
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from threading import Lock

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd  # type: ignore[import-untyped]
import webrtcvad  # type: ignore[import-untyped]

from core.settings import setting
from voice.audio_devices import AudioRoute

def audio_device(env_name: str) -> int | str | None:
    # 环境变量既可以填设备编号（如 2），也可以填设备名；没填就让音频库使用默认设备。
    value = setting(env_name)  # REALTIME_INPUT_DEVICE / REALTIME_OUTPUT_DEVICE，默认值在参数清单里
    return int(value) if value.isdecimal() else value or None


def play_audio(audio: bytes) -> None:
    # API 音频是 24kHz；播放设备可以使用自己的默认采样率。
    device = audio_device("REALTIME_OUTPUT_DEVICE")
    info = sd.query_devices(device, "output")
    output_rate = int(info["default_samplerate"])
    sd.check_output_settings(device=device, channels=1, dtype="float32", samplerate=output_rate)
    samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768
    if output_rate != 24000 and samples.size:
        # 单声道语音线性重采样；不能只改播放采样率，否则语速和音高会改变。
        count = round(samples.size * output_rate / 24000)
        samples = np.interp(
            np.arange(count) * 24000 / output_rate,
            np.arange(samples.size), samples,
        ).astype(np.float32)
    print(f"[播放设备] {info['name']} ｜ {output_rate}Hz")
    try:
        sd.play(samples, samplerate=output_rate, device=device)
        sd.wait()
    finally:
        sd.stop()


class MicrophoneUnavailable(RuntimeError):
    pass


class SpeechActivity:
    """本地 20ms VAD；支持 16kHz 和 24kHz 单声道 PCM16 输入。"""
    def __init__(self, sample_rate: int = 24000):
        if sample_rate not in (16000, 24000):
            raise ValueError("本地 VAD 输入采样率必须是 16000 或 24000")
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate // 50
        self.vad = webrtcvad.Vad(2)
        self.voiced_ms = 0
        self.last_voice_at = 0.0
        self.gap_ms = 0

    def observe(self, chunk: bytes) -> None:
        # bytes 是原始音频；frombuffer 按 PCM16 解释每个采样点，而不是把它当文字解码。
        pcm = np.frombuffer(chunk, dtype="<i2")
        if pcm.size != self.frame_samples:
            return
        if self.sample_rate == 16000:
            frame = pcm
        else:
            # WebRTC VAD 不接受 24kHz：把同一段 20ms 音频重采样成它接受的 16kHz。
            frame = np.interp(np.arange(320) * 1.5, np.arange(480), pcm).astype("<i2")
        if self.vad.is_speech(frame.tobytes(), 16000):
            # 这里累计的是检测到的人声时长，给“要不要暂停旧回答”提供依据。
            self.voiced_ms += 20
            self.gap_ms = 0
            self.last_voice_at = time.perf_counter()
        else:
            self.gap_ms += 20
            if self.gap_ms >= 100:
                # 长一些的静音将连续人声计数归零；短间隙允许一句话自然停顿。
                self.voiced_ms = 0

    def reset(self) -> None:
        self.voiced_ms = 0
        self.gap_ms = 0
        self.last_voice_at = 0.0


@dataclass
class _PlaybackWaiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[bool]
    scheduled: bool = False


@dataclass
class _PlaybackStartWaiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[float | None]
    scheduled: bool = False


@dataclass
class _ItemPlayback:
    pending_frames: int = 0
    received: bool = False
    finished: bool = False
    deadline: float | None = None
    result: bool | None = None
    waiters: list[_PlaybackWaiter] = field(default_factory=list)
    started_at: float | None = None
    cleared_at: float | None = None
    start_waiters: list[_PlaybackStartWaiter] = field(default_factory=list)


class StreamingPlayer:
    """持续打开输出流，回调从缓冲区取音频；打断时清空缓冲区。"""
    def __init__(self, route: AudioRoute | None = None):
        # 系统默认设备变化时由 route 协调关闭和重开；默认不跟随（测试、CLI 等）。
        self.route = route if route is not None else AudioRoute(follow_input=False, follow_output=False)
        # 接收模型音频的代码和 sounddevice 播放回调不在同一个线程。
        # 两边访问同一份缓冲区，所以用线程 Lock 保护，避免同时修改队列或播放位置。
        self.lock = Lock()
        # 缓冲区存“收到但还没播完”的音频；offset 是第一块音频已经播放到的位置。
        self.buffers: deque[NDArray[np.float32]] = deque()
        # 与音频块一一对应，保留每块属于哪个会话条目；新回答排队，不覆盖没播完的旧话。
        self._buffer_items: deque[tuple[str, int]] = deque()
        self.offset = 0
        self.rate = 24000
        self.ready = False
        self.item_id: str | None = None
        self.content_index = 0
        self.played_frames = 0
        self.last_submit_at = 0.0
        self.latency = 0.0
        self.gain = 1.0
        self.paused = False
        # 每个 item 独立记录收到、交给声卡和结束的状态，不能用全局 buffers 是否为空判断。
        # 按首次出现的顺序保留最近 200 个 item；更老但尚在播放/等待的记录暂不淘汰。
        # 这只是有限的晚到音频保护，语音层仍须过滤已取消回答的音频事件。
        self._playbacks: OrderedDict[str, _ItemPlayback] = OrderedDict()

    def _trim_playbacks(self) -> None:
        """调用方持有 lock。最近 200 条保留，其余只删除终态且没有等待者的记录。"""
        if len(self._playbacks) <= 200:
            return
        for item_id in list(self._playbacks)[:-200]:
            playback = self._playbacks[item_id]
            if playback.waiters or playback.start_waiters or playback.pending_frames:
                continue
            # 没有调用方等待时也能判定已过输出延迟的条目，避免它们永远积累。
            self._mark_played(playback)
            if playback.result is not None:
                del self._playbacks[item_id]

    def append(self, audio: bytes, item_id: str, content_index: int) -> None:
        # 收到模型的一小块音频时调用。这里只入队，不等待扬声器把它播放完。
        # 模型返回 int16；输出流使用 float32，数值由整数范围缩放到约 -1～1。
        samples = np.frombuffer(audio, dtype="<i2").astype(np.float32)
        samples /= 32768
        with self.lock:
            playback = self._playbacks.setdefault(item_id, _ItemPlayback())
            self._trim_playbacks()
            if playback.result is not None or playback.finished:
                # clear/finish 后晚到的块不能重新播放，也不能让旧条目重新获得完成资格。
                return
            if not self.ready:
                # 设备尚未打开或已断开时丢弃这一块，防止恢复后播放过时回答。
                playback.result = False
                self._notify_playback(playback)
                self._trim_playbacks()
                return
            if self.rate != 24000 and samples.size:
                count = round(samples.size * self.rate / 24000)
                samples = np.interp(np.arange(count) * 24000 / self.rate,
                                    np.arange(samples.size), samples).astype(np.float32)
            if not samples.size:
                return
            playback.received = True
            playback.pending_frames += samples.size
            playback.deadline = None
            if self.item_id is None:
                # 第一次入队时记录条目；后续条目的切换要等 callback 真正播放到它。
                self.item_id = item_id
                self.content_index = content_index
                self.played_frames = 0
            self.buffers.append(samples)
            self._buffer_items.append((item_id, content_index))

    def callback(self, outdata, frames, time_info, status) -> None:
        # 不是主程序手动循环调用：音频库需要下一小块声音时，在音频线程调用它。
        # outdata 是本次要交给扬声器的数组；没有数据或暂停时，保留为全零（静音）。
        outdata.fill(0)
        with self.lock:
            if self.paused:
                # 暂停只停止消费缓冲区，已经收到的音频留着，resume 后接着播。
                return
            written = 0
            while self.buffers and written < frames:
                # 一次回调可以取多块音频，也可以只取一块的一部分，直到填满本次输出。
                item_id, content_index = self._buffer_items[0]
                if (self.item_id, self.content_index) != (item_id, content_index):
                    self.item_id, self.content_index = item_id, content_index
                    self.played_frames = 0
                samples = self.buffers[0]
                count = min(frames - written, samples.size - self.offset)
                playback = self._playbacks[item_id]
                if count and playback.started_at is None:
                    # PortAudio 的 DAC 时间属于另一个时钟域；只取差值再映射到本地单调时钟。
                    delay = max(0.0, self.latency)
                    if time_info is not None:
                        try:
                            delay = max(0.0, float(time_info.outputBufferDacTime) - float(time_info.currentTime))
                        except (AttributeError, TypeError, ValueError):
                            pass
                    playback.started_at = time.perf_counter() + delay + written / self.rate
                    self._notify_started(playback)
                outdata[written:written + count, 0] = samples[self.offset:self.offset + count]
                self.offset += count
                written += count
                # 记录当前条目的播放量；同一次 callback 跨条目时也不能把前一条的帧数算进来。
                self.played_frames += count
                playback.pending_frames -= count
                if playback.pending_frames == 0:
                    # 只是交给声卡；完成通知要等到最后一次提交的输出延迟过去。
                    playback.deadline = time.perf_counter() + max(0.0, self.latency)
                    self._notify_playback(playback)
                if self.offset == samples.size:
                    self.buffers.popleft()
                    self._buffer_items.popleft()
                    self.offset = 0
            outdata *= self.gain
            # gain=0.2 是降低音量；与 pause 不同，降低音量时仍然继续消费音频。
            if written:
                self.last_submit_at = time.perf_counter()
            self._trim_playbacks()

    def finish(self, item_id: str) -> None:
        """全部音频收到；即使缓冲区已被取空，也需要这一步才允许报告完成。"""
        with self.lock:
            playback = self._playbacks.setdefault(item_id, _ItemPlayback())
            playback.finished = True
            if not playback.received and playback.result is None:
                playback.result = False
            self._notify_playback(playback)
            self._trim_playbacks()

    def playback_finished(self, item_id: str) -> bool:
        """同步快照：含声卡延迟，供打断瞬间记录已完整播完的短句。"""
        with self.lock:
            playback = self._playbacks.get(item_id)
            if playback is None:
                return False
            self._mark_played(playback)
            return playback.result is True

    async def wait_played(self, item_id: str) -> bool:
        """等待该 item 播完；无音频、丢弃或被 clear 打断都返回 False。

        语音接线时先在 response.audio.done / response.done 调用 finish(item_id)，
        再调用本方法；一块音频也没收到时会立即返回 False，不会等待音频开始。
        已收到音频时也支持提前等待，直到 finish 和实际播放完成两个条件都满足。
        """
        loop = asyncio.get_running_loop()
        with self.lock:
            playback = self._playbacks.get(item_id)
            if playback is None or not playback.received:
                return False
            self._mark_played(playback)
            if playback.result is not None:
                self._trim_playbacks()
                return playback.result
            waiter = _PlaybackWaiter(loop, loop.create_future())
            playback.waiters.append(waiter)
            self._notify_playback(playback)
            self._trim_playbacks()
        try:
            return await waiter.future
        finally:
            with self.lock:
                playback.waiters.remove(waiter)
                self._trim_playbacks()

    @staticmethod
    def _mark_played(playback: _ItemPlayback) -> None:
        # 调用方持有 lock；结果一旦确定就不再改变，clear 不会推翻已经播放完成的条目。
        if (playback.result is None and playback.received and playback.finished
                and playback.pending_frames == 0 and playback.deadline is not None
                and time.perf_counter() >= playback.deadline):
            playback.result = True

    def _notify_playback(self, playback: _ItemPlayback) -> None:
        # 调用方持有 lock。这里可能在声卡线程，绝不能直接设置 asyncio.Future。
        self._notify_started(playback)
        ready = playback.finished and playback.pending_frames == 0 and playback.deadline is not None
        for waiter in playback.waiters:
            if playback.result is not None or (ready and not waiter.scheduled):
                waiter.scheduled = True
                try:
                    waiter.loop.call_soon_threadsafe(self._deliver_playback, playback, waiter)
                except RuntimeError:
                    if not waiter.loop.is_closed():
                        raise

    async def wait_started(self, item_id: str) -> float | None:
        """首帧交给声卡后，等待到预计 DAC 播放时刻；无音频或播放前被清空返回 None。

        在 append 之后调用。返回本进程 perf_counter 时间，不是首个网络音频包的时间；
        这是根据设备时间/延迟的估计，不能测量扬声器的物理声波。
        """
        loop = asyncio.get_running_loop()
        with self.lock:
            playback = self._playbacks.get(item_id)
            if playback is None or not playback.received:
                return None
            waiter = _PlaybackStartWaiter(loop, loop.create_future())
            playback.start_waiters.append(waiter)
            self._notify_started(playback)
        try:
            return await waiter.future
        finally:
            with self.lock:
                playback.start_waiters.remove(waiter)
                self._trim_playbacks()

    def _notify_started(self, playback: _ItemPlayback) -> None:
        for waiter in playback.start_waiters:
            if not waiter.scheduled and (playback.started_at is not None or playback.result is False):
                waiter.scheduled = True
                try:
                    waiter.loop.call_soon_threadsafe(self._deliver_started, playback, waiter)
                except RuntimeError:
                    if not waiter.loop.is_closed():
                        raise

    def _deliver_started(self, playback: _ItemPlayback, waiter: _PlaybackStartWaiter) -> None:
        with self.lock:
            if waiter.future.done():
                return
            started = playback.started_at
            if started is None or (playback.cleared_at is not None and playback.cleared_at < started):
                waiter.future.set_result(None)
            elif started <= time.perf_counter():
                waiter.future.set_result(started)
            else:
                waiter.loop.call_later(started - time.perf_counter(), self._deliver_started, playback, waiter)

    def _deliver_playback(self, playback: _ItemPlayback, waiter: _PlaybackWaiter) -> None:
        # 只在对应的事件循环运行；call_later 等输出延迟，不阻塞声卡回调线程。
        with self.lock:
            if waiter.future.done():
                return
            self._mark_played(playback)
            result = playback.result
            deadline = playback.deadline
        if result is not None:
            waiter.future.set_result(result)
        elif deadline is not None:
            waiter.loop.call_later(max(0.0, deadline - time.perf_counter()),
                                   self._deliver_playback, playback, waiter)

    def duck(self, enabled: bool) -> None:
        with self.lock:
            self.gain = 0.2 if enabled else 1.0

    def pause(self) -> None:
        with self.lock:
            self.paused = True

    def resume(self) -> None:
        with self.lock:
            self.paused = False
            self.gain = 1.0

    def clear(self) -> tuple[str, int, int] | None:
        # 真正打断时丢弃未播内容，并返回“哪个条目、哪段内容、听到了多少毫秒”。
        # 语音层用这个结果请求截断服务端历史，避免模型以为用户已经听完。
        with self.lock:
            for playback in self._playbacks.values():
                if playback.result is None:
                    playback.result = False
                    playback.cleared_at = time.perf_counter()
                playback.pending_frames = 0
                self._notify_playback(playback)
            # 扣除估计仍在设备缓冲区的尾部，得到保守的已听时长。
            pending_frames = min(self.played_frames, max(0, round(
                (self.latency - (time.perf_counter() - self.last_submit_at)) * self.rate
            )))
            interrupted = bool(self.buffers) or pending_frames > 0
            played_ms = max(0, (self.played_frames - pending_frames) * 1000 // self.rate)
            item = (self.item_id, self.content_index, played_ms) if self.item_id and interrupted else None
            if self._buffer_items and not pending_frames:
                next_item = self._buffer_items[0]
                if next_item != (self.item_id, self.content_index):
                    # 上一句已播完、下一句还没开始：下一条的已听时长应为 0。
                    item = (*next_item, 0)
            self.buffers.clear()
            self._buffer_items.clear()
            self.offset = 0
            self.item_id = None
            self.played_frames = 0
            self.gain = 1.0
            self.paused = False
            self._trim_playbacks()
            return item

    async def run(self) -> None:
        # 这是播放器的长期任务：打开设备 → 维持输出流 → 设备出错后等待并重试。
        # 实际音频搬运由 callback 完成，这里不逐块播放模型返回的数据。
        while True:
            switched = False
            try:
                # 切换设备进行中时先等它完成，不在重新初始化 PortAudio 前后打开流。
                await self.route.wait_ready()
                with self.route.stream():
                    device = audio_device("REALTIME_OUTPUT_DEVICE")
                    info = sd.query_devices(device, "output")
                    self.rate = int(info["default_samplerate"])
                    # 使用播放设备自己的采样率；append 会把模型的 24kHz 音频转换过来。
                    stream = sd.OutputStream(
                        samplerate=self.rate, channels=1, dtype="float32",
                        device=device, blocksize=max(1, self.rate // 100),
                        latency="low", callback=self.callback,
                    )
                    with stream:
                        # 普通 with 负责同步打开/关闭本地设备；内部仍然可以使用 await。
                        self.latency = float(stream.latency)
                        self.ready = True
                        print(f"[流式播放] {info['name']} ｜ {self.rate}Hz")
                        while stream.active:
                            if self.route.changed.is_set():
                                # 系统默认设备变了：正常关闭，随后用新设备重新打开。
                                switched = True
                                break
                            # 只暂停当前 run 任务 0.1 秒，让录音、收网络消息等任务继续执行。
                            # 扬声器由音频线程持续处理，不是每隔 0.1 秒才播放一次。
                            await asyncio.sleep(0.1)
                        else:
                            raise sd.PortAudioError("输出流已停止")
            except (sd.PortAudioError, ValueError) as exc:
                print(f"[播放设备不可用] {exc}。1 秒后重试。")
            finally:
                # 正常退出、切换设备、设备报错或任务被取消都会经过这里，清掉过时音频。
                self.ready = False
                self.clear()
            if not switched:
                await asyncio.sleep(1)


def is_meaningful(text: str) -> bool:
    """过滤空白、标点、咳嗽及纯语气声，保留简短的有效回答。"""
    # 这是简单的文本规则，不是另一次模型调用；“好”“是”等简短回答仍可通过。
    normalized = "".join(char for char in text.lower() if char.isalnum())
    if not normalized:
        return False
    if normalized in {"cough", "coughing"}:
        return False
    return not all(char in "咳嗯呃啊唔哼" for char in normalized)


@dataclass
class SpeechCandidate:
    # 一次疑似用户发声的状态：音频提交、转写完成、本地 VAD 确认可能先后到达。
    # paused=暂时停播等待判断；confirmed=已确认打断；responded=已经请求模型回答。
    item_id: str
    audio_start_ms: int
    stopped: bool = False
    committed: bool = False
    confirmed: bool = False
    responded: bool = False
    meaningful: bool | None = None
    paused: bool = False


