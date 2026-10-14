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
        pcm = np.frombuffer(chunk, dtype="<i2")
        if pcm.size != self.frame_samples:
            return
        if self.sample_rate == 16000:
            frame = pcm
        else:
            frame = np.interp(np.arange(320) * 1.5, np.arange(480), pcm).astype("<i2")
        if self.vad.is_speech(frame.tobytes(), 16000):
            self.voiced_ms += 20
            self.gap_ms = 0
            self.last_voice_at = time.perf_counter()
        else:
            self.gap_ms += 20
            if self.gap_ms >= 100:
                self.voiced_ms = 0

    def reset(self) -> None:
        self.voiced_ms = 0
        self.gap_ms = 0
        self.last_voice_at = 0.0


class StreamingPlayer:
    """持续打开输出流，回调从缓冲区取音频；打断时清空缓冲区。"""
    def __init__(self):
        self.lock = Lock()
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
        samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768
        with self.lock:
            if not self.ready:
                return
            if self.rate != 24000 and samples.size:
                count = round(samples.size * self.rate / 24000)
                samples = np.interp(np.arange(count) * 24000 / self.rate,
                                    np.arange(samples.size), samples).astype(np.float32)
            if self.item_id != item_id:
                self.buffers.clear()
                self.offset = 0
                self.item_id = item_id
                self.content_index = content_index
                self.played_frames = 0
            self.buffers.append(samples)

    def callback(self, outdata, frames, time_info, status) -> None:
        outdata.fill(0)
        with self.lock:
            if self.paused:
                return
            written = 0
            while self.buffers and written < frames:
                samples = self.buffers[0]
                count = min(frames - written, samples.size - self.offset)
                outdata[written:written + count, 0] = samples[self.offset:self.offset + count]
                self.offset += count
                written += count
                if self.offset == samples.size:
                    self.buffers.popleft()
                    self.offset = 0
            outdata *= self.gain
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
        while True:
            stream = None
            try:
                device = audio_device("REALTIME_OUTPUT_DEVICE")
                info = sd.query_devices(device, "output")
                self.rate = int(info["default_samplerate"])
                stream = sd.OutputStream(
                    samplerate=self.rate, channels=1, dtype="float32",
                    device=device, blocksize=max(1, self.rate // 100),
                    latency="low", callback=self.callback,
                )
                with stream:
                    self.latency = float(stream.latency)
                    self.ready = True
                    print(f"[流式播放] {info['name']} ｜ {self.rate}Hz")
                    while stream.active:
                        await asyncio.sleep(0.1)
                    raise sd.PortAudioError("输出流已停止")
            except (sd.PortAudioError, ValueError) as exc:
                print(f"[播放设备不可用] {exc}。1 秒后重试。")
            finally:
                self.ready = False
                self.clear()
            await asyncio.sleep(1)


def is_meaningful(text: str) -> bool:
    """过滤空白、标点、咳嗽及纯语气声，保留简短的有效回答。"""
    normalized = "".join(char for char in text.lower() if char.isalnum())
    if not normalized:
        return False
    if normalized in {"cough", "coughing"}:
        return False
    return not all(char in "咳嗯呃啊唔哼" for char in normalized)


@dataclass
class SpeechCandidate:
    item_id: str
    audio_start_ms: int
    stopped: bool = False
    committed: bool = False
    confirmed: bool = False
    responded: bool = False
    meaningful: bool | None = None
    paused: bool = False


