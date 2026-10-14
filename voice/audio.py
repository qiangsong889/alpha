"""与 API 无关的录音设备选择、播放、VAD 和语音有效性判断。"""

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
import webrtcvad  # type: ignore[import-untyped]

def audio_device(env_name: str) -> int | str | None:
    # 环境变量既可以填设备编号（如 2），也可以填设备名；没填就让音频库使用默认设备。
    value = os.environ.get(env_name, "").strip()
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


class StreamingPlayer:
    """持续打开输出流，回调从缓冲区取音频；打断时清空缓冲区。"""
    def __init__(self):
        # 接收模型音频的代码和 sounddevice 播放回调不在同一个线程。
        # 两边访问同一份缓冲区，所以用线程 Lock 保护，避免同时修改队列或播放位置。
        self.lock = Lock()
        # 缓冲区存“收到但还没播完”的音频；offset 是第一块音频已经播放到的位置。
        self.buffers = deque()
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

    def append(self, audio: bytes, item_id: str, content_index: int) -> None:
        # 收到模型的一小块音频时调用。这里只入队，不等待扬声器把它播放完。
        # 模型返回 int16；输出流使用 float32，数值由整数范围缩放到约 -1～1。
        samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768
        with self.lock:
            if not self.ready:
                # 设备尚未打开或已断开时丢弃这一块，防止恢复后播放过时回答。
                return
            if self.rate != 24000 and samples.size:
                count = round(samples.size * self.rate / 24000)
                samples = np.interp(np.arange(count) * 24000 / self.rate,
                                    np.arange(samples.size), samples).astype(np.float32)
            if self.item_id != item_id:
                # item_id 标识这段回答音频；新条目开始时重新统计已播放时长。
                self.buffers.clear()
                self.offset = 0
                self.item_id = item_id
                self.content_index = content_index
                self.played_frames = 0
            self.buffers.append(samples)

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
                samples = self.buffers[0]
                count = min(frames - written, samples.size - self.offset)
                outdata[written:written + count, 0] = samples[self.offset:self.offset + count]
                self.offset += count
                written += count
                if self.offset == samples.size:
                    self.buffers.popleft()
                    self.offset = 0
            outdata *= self.gain
            # gain=0.2 是降低音量；与 pause 不同，降低音量时仍然继续消费音频。
            self.played_frames += written
            if written:
                self.last_submit_at = time.perf_counter()

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
            # 扣除估计仍在设备缓冲区的尾部，得到保守的已听时长。
            pending_frames = min(self.played_frames, max(0, round(
                (self.latency - (time.perf_counter() - self.last_submit_at)) * self.rate
            )))
            interrupted = bool(self.buffers) or pending_frames > 0
            played_ms = max(0, (self.played_frames - pending_frames) * 1000 // self.rate)
            item = (self.item_id, self.content_index, played_ms) if self.item_id and interrupted else None
            self.buffers.clear()
            self.offset = 0
            self.item_id = None
            self.played_frames = 0
            self.gain = 1.0
            self.paused = False
            return item

    async def run(self) -> None:
        # 这是播放器的长期任务：打开设备 → 维持输出流 → 设备出错后等待并重试。
        # 实际音频搬运由 callback 完成，这里不逐块播放模型返回的数据。
        while True:
            stream = None
            try:
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
                        # 只暂停当前 run 任务 0.1 秒，让录音、收网络消息等任务继续执行。
                        # 扬声器由音频线程持续处理，不是每隔 0.1 秒才播放一次。
                        await asyncio.sleep(0.1)
                    raise sd.PortAudioError("输出流已停止")
            except (sd.PortAudioError, ValueError) as exc:
                print(f"[播放设备不可用] {exc}。1 秒后重试。")
            finally:
                # 正常退出、设备报错或任务被取消都会经过这里，清掉过时音频。
                self.ready = False
                self.clear()
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


