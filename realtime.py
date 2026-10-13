"""第 4/5 步：流式语音播放、打断和首段音频延迟测量。

运行：alpha/.venv/bin/python alpha/realtime.py
.env：OPENAI_API_KEY；可选 OPENAI_REALTIME_MODEL、OPENAI_REALTIME_VOICE。
收到 speech_started 时清空播放缓冲区，并截断未听完的回答。
"""

import asyncio
import base64
import os
from pathlib import Path
import time
from collections import deque
from threading import Lock
from dataclasses import dataclass
import numpy as np
import sounddevice as sd
import webrtcvad

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.realtime.session_update_event_param import Session
from openai.resources.realtime.realtime import AsyncRealtimeConnection


def session_config(voice: str) -> Session:
    return {
        "type": "realtime",
        "instructions": (
            "你是一个中文语音助手。无论用户说中文还是英文，始终只用普通话回答，"
            "用一两句简短口语回答，不使用 Markdown、列表或表情符号。"
            "用户打断你时，停止原来的回答，听取并回应用户的新问题。"
        ),
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "gpt-4o-mini-transcribe", "language": "zh"},
                "turn_detection": {
                    "type": "semantic_vad",
                    "eagerness": "auto",
                    "create_response": False,
                    "interrupt_response": False,
                },
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": voice,
            },
        },
    }


async def text_to_audio(connection: AsyncRealtimeConnection, text: str) -> bytes:
    print(f"你：{text}")
    started = time.perf_counter()
    await connection.conversation.item.create(item={
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    })
    await connection.response.create()
    chunks: list[bytes] = []
    transcript: list[str] = []
    async with asyncio.timeout(60):
        async for event in connection:
            if event.type == "error":
                raise RuntimeError(f"{event.error.code}: {event.error.message}")
            if event.type == "response.output_audio.delta":
                chunk = base64.b64decode(event.delta)
                if chunk:
                    if not chunks:
                        print(f"[延迟] 文字发送到首个音频包：{time.perf_counter() - started:.2f}s")
                    chunks.append(chunk)
            elif event.type == "response.output_audio_transcript.delta":
                transcript.append(event.delta)
            elif event.type == "response.done":
                usage = event.response.usage
                print(f"[usage] {usage.model_dump_json(exclude_none=True) if usage is not None else '未返回'}")
                if event.response.status != "completed":
                    raise RuntimeError(f"回答未完成：{event.response.status}，{event.response.status_details}")
                print(f"模型：{''.join(transcript) or '（未收到文字转录）'}")
                print(f"[耗时] 完整回答接收：{time.perf_counter() - started:.2f}s")
                if not chunks:
                    raise RuntimeError("回答完成但没有收到音频")
                return b"".join(chunks)
    raise RuntimeError("连接关闭，未收到完整回答")


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
    """本地 20ms VAD，避免 semantic VAD 的句尾等待被算成人声。"""
    def __init__(self):
        self.vad = webrtcvad.Vad(2)
        self.voiced_ms = 0
        self.last_voice_at = 0.0
        self.gap_ms = 0

    def observe(self, chunk: bytes) -> None:
        pcm = np.frombuffer(chunk, dtype="<i2")
        # 麦克风固定每块 20ms/24kHz；WebRTC VAD 使用 16kHz/320 samples。
        if pcm.size != 480:
            return
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


async def send_microphone(connection: AsyncRealtimeConnection, activity: SpeechActivity) -> None:
    while True:
        try:
            await send_microphone_once(connection, activity)
        except (sd.PortAudioError, MicrophoneUnavailable) as exc:
            print(f"[麦克风断开] {exc}。会话保留，1 秒后尝试重新打开。")
            # 丢弃断开前未完成的输入，避免与恢复后的句子拼接。
            await connection.input_audio_buffer.clear()
            activity.reset()
            await asyncio.sleep(1)


async def send_microphone_once(connection: AsyncRealtimeConnection, activity: SpeechActivity) -> None:
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[bytes | Exception] = asyncio.Queue(maxsize=100)
    accepting = True

    def enqueue(chunk: bytes, status: str) -> None:
        nonlocal accepting
        if not accepting:
            return
        if status:
            print(f"[麦克风] {status}")
        activity.observe(chunk)
        if audio_queue.full():
            accepting = False
            audio_queue.get_nowait()
            audio_queue.put_nowait(RuntimeError("音频发送积压超过 2 秒，请检查网络"))
        else:
            audio_queue.put_nowait(chunk)

    def callback(indata, frames, time_info, status):
        # sounddevice 回调在音频线程执行，不直接操作 asyncio.Queue。
        chunk = indata.astype("<i2", copy=False).tobytes()
        loop.call_soon_threadsafe(enqueue, chunk, str(status) if status else "")

    try:
        try:
            stream = sd.InputStream(
                samplerate=24000, channels=1, dtype="int16",
                blocksize=480, callback=callback,  # 每块 20 ms
                device=audio_device("REALTIME_INPUT_DEVICE"),
            )
        except ValueError as exc:
            raise MicrophoneUnavailable(str(exc)) from exc
        with stream:
            print("麦克风已开启，请用中文说话。Ctrl+C 退出，建议戴耳机。")
            while True:
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=1)
                except TimeoutError as exc:
                    raise MicrophoneUnavailable("1 秒内没有收到麦克风音频") from exc
                if not stream.active:
                    raise MicrophoneUnavailable("麦克风流已停止")
                if isinstance(chunk, Exception):
                    raise chunk
                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(chunk).decode("ascii")
                )
    finally:
        accepting = False


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


async def receive_microphone_answers(
    connection: AsyncRealtimeConnection, player: StreamingPlayer, activity: SpeechActivity,
) -> None:
    threshold_ms = int(os.environ.get("INTERRUPT_MS", "400"))
    if not 300 <= threshold_ms <= 500:
        raise ValueError("INTERRUPT_MS 必须在 300 到 500 之间")
    transcript: list[str] = []
    speech_stopped_at: float | None = None
    first_audio = True
    response_id: str | None = None
    interrupted_responses: set[str] = set()
    candidate: SpeechCandidate | None = None
    timer: asyncio.Task | None = None
    response_pending = False
    deleted_items: set[str] = set()
    receive_started = time.perf_counter()

    def log(message: str) -> None:
        print(f"[{time.perf_counter() - receive_started:.3f}] {message}", flush=True)

    async def delete_noise(item_id: str) -> None:
        if item_id in deleted_items:
            return
        await connection.conversation.item.delete(item_id=item_id)
        deleted_items.add(item_id)
        log(f"客户端发送 conversation.item.delete（无意义条目 {item_id}）")

    async def maybe_respond() -> None:
        nonlocal response_pending
        if (candidate is not None and candidate.committed and candidate.meaningful is True
                and not candidate.responded and response_id is None and not response_pending):
            candidate.responded = True
            response_pending = True
            log(f"客户端发送 response.create（用户条目 {candidate.item_id}）")
            await connection.response.create()

    async def confirm(target: SpeechCandidate, reason: str) -> None:
        if candidate is not target or target.confirmed:
            return
        target.confirmed = True
        item = player.clear()
        log(f"确认结果：是打断 → 清空播放缓冲区（{reason}）")
        if response_id is not None:
            interrupted_responses.add(response_id)
            log(f"客户端发送 response.cancel（{response_id}）")
            await connection.response.cancel(response_id=response_id)
        if item is not None:
            item_id, content_index, played_ms = item
            await connection.conversation.item.truncate(
                item_id=item_id, content_index=content_index, audio_end_ms=played_ms,
            )
        await maybe_respond()

    def pause_candidate(target: SpeechCandidate) -> None:
        if (candidate is not target or target.paused or target.confirmed
                or target.meaningful is not None):
            return
        target.paused = True
        player.pause()
        log(f"[暂停] 本地 VAD 人声 {activity.voiced_ms}ms，保留缓冲区，等待转录")

    async def confirm_after_delay(target: SpeechCandidate) -> None:
        while candidate is target and not target.stopped and not target.confirmed:
            if activity.voiced_ms >= threshold_ms:
                pause_candidate(target)
                return
            # 服务端尚未报告 speech_stopped，也可以恢复短噪音后的音量。
            recent_voice = time.perf_counter() - activity.last_voice_at < 0.15
            if player.gain != (0.2 if recent_voice else 1.0):
                log("继续确认 → 降低音量" if recent_voice else
                    "确认结果：暂未确认打断 → 恢复音量，仍等待文字")
            player.duck(recent_voice)
            await asyncio.sleep(0.02)

    try:
        async with asyncio.TaskGroup() as timers:
            async for event in connection:
                if event.type == "error":
                    # 取消请求可能与 response.done 交错，此错误无需结束会话。
                    if event.error.code == "response_cancel_not_active":
                        log(f"error：{event.error.code}（回答已结束，忽略取消竞态）")
                        continue
                    raise RuntimeError(f"{event.error.code}: {event.error.message}")
                if event.type == "input_audio_buffer.speech_started":
                    log(f"speech_started（item_id={event.item_id}）")
                    if timer is not None:
                        timer.cancel()
                    candidate = SpeechCandidate(event.item_id, event.audio_start_ms)
                    player.duck(True)
                    log(f"降低音量到 20%，开始确认（阈值 {threshold_ms}ms）")
                    timer = timers.create_task(confirm_after_delay(candidate))
                    speech_stopped_at = None
                elif event.type == "input_audio_buffer.speech_stopped":
                    speech_stopped_at = time.perf_counter()
                    duration = (
                        f"持续了 {(event.audio_end_ms - candidate.audio_start_ms) / 1000:.2f}s"
                        if candidate is not None and candidate.item_id == event.item_id
                        else "未匹配到开始事件"
                    )
                    log(f"speech_stopped（{duration}，item_id={event.item_id}）")
                    if candidate is not None and candidate.item_id == event.item_id:
                        candidate.stopped = True
                        if timer is not None:
                            timer.cancel()
                        if activity.voiced_ms >= threshold_ms:
                            pause_candidate(candidate)
                        elif not candidate.confirmed and not candidate.paused:
                            player.duck(False)
                            log("确认结果：暂不是打断 → 恢复音量（后续非空文字仍可确认）")
                elif event.type == "input_audio_buffer.committed":
                    log(f"input_audio_buffer.committed（item_id={event.item_id}）")
                    if candidate is not None and candidate.item_id == event.item_id:
                        candidate.committed = True
                        await maybe_respond()
                elif event.type == "conversation.item.input_audio_transcription.completed":
                    text = event.transcript.strip()
                    log(f"识别出的文字：{text!r}（input_audio_transcription.completed，item_id={event.item_id}）")
                    if candidate is not None and candidate.item_id == event.item_id:
                        candidate.meaningful = is_meaningful(text)
                        if timer is not None:
                            timer.cancel()
                        if candidate.meaningful is False:
                            log(f'[忽略] 识别出的文字："{text}"，不回应')
                            player.resume()
                            log("[恢复] 只是噪音，继续播放")
                            await delete_noise(event.item_id)
                        else:
                            await confirm(candidate, "识别到有意义的文字")
                        await maybe_respond()
                    elif not is_meaningful(text):
                        # 迟到的噪音转录也应删除，但不能改变当前新一轮的播放状态。
                        await delete_noise(event.item_id)
                elif event.type == "conversation.item.deleted":
                    log(f"conversation.item.deleted（item_id={event.item_id}）")
                elif event.type == "conversation.item.input_audio_transcription.failed":
                    log(f"input_audio_transcription.failed：{event.error.message}")
                    if candidate is not None and candidate.item_id == event.item_id:
                        candidate.meaningful = False
                        player.resume()
                        log("[恢复] 转录失败，保留旧回答继续播放，不生成新回答")
                elif event.type == "response.created":
                    source = "客户端已请求创建" if response_pending else "未记录到客户端创建请求"
                    log(f"response.created（{event.response.id}；{source}）")
                    response_id = event.response.id
                    response_pending = False
                    first_audio = True
                    transcript = []
                    # 新回答建立时，尚在说话的已确认用户需要取消它。
                    if candidate is not None and candidate.confirmed and not candidate.stopped:
                        interrupted_responses.add(response_id)
                        await connection.response.cancel(response_id=response_id)
                elif event.type == "response.output_audio.delta":
                    if event.response_id in interrupted_responses:
                        continue
                    chunk = base64.b64decode(event.delta)
                    if chunk:
                        if first_audio:
                            if speech_stopped_at is not None:
                                log(f"speech_stopped 到首段音频：{time.perf_counter() - speech_stopped_at:.2f}s")
                            first_audio = False
                        player.append(chunk, event.item_id, event.content_index)
                elif event.type == "response.output_audio_transcript.delta":
                    if event.response_id not in interrupted_responses:
                        transcript.append(event.delta)
                elif event.type == "response.done":
                    log(f"response.done（{event.response.id}，{event.response.status}）")
                    usage = event.response.usage
                    log(f"[usage] {usage.model_dump_json(exclude_none=True) if usage is not None else '未返回'}")
                    if event.response.id not in interrupted_responses:
                        if event.response.status == "completed":
                            log(f"模型：{''.join(transcript) or '（未收到文字转录）'}")
                        else:
                            log(f"回答状态：{event.response.status}")
                    if response_id == event.response.id:
                        response_id = None
                    interrupted_responses.discard(event.response.id)
                    await maybe_respond()
            raise RuntimeError("Realtime 连接已关闭")
    finally:
        if timer is not None:
            timer.cancel()
        player.resume()


async def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise SystemExit("请在 alpha/.env 中设置 OPENAI_API_KEY")
    model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
    voice = os.environ.get("OPENAI_REALTIME_VOICE", "marin")

    async with AsyncOpenAI() as client:
        started = time.perf_counter()
        async with client.realtime.connect(model=model) as connection:
            print(f"[连接] {model}，WebSocket 建立耗时 {time.perf_counter() - started:.2f}s")
            configured_at = time.perf_counter()
            await connection.session.update(session=session_config(voice))
            # update() 只发送配置；session.updated 才表示服务端已确认。
            async with asyncio.timeout(30):
                async for event in connection:
                    if event.type == "error":
                        raise RuntimeError(f"{event.error.code}: {event.error.message}")
                    if event.type == "session.created":
                        print(f"[会话] {event.session.id}")
                    elif event.type == "session.updated":
                        print(f"[配置] 服务端已确认，耗时 {time.perf_counter() - configured_at:.2f}s")
                        print(f"中文指令 ｜ PCM16 单声道 24kHz ｜ voice={voice} ｜ semantic_vad")
                        break
                else:
                    raise RuntimeError("连接关闭，未收到 session.updated")
            player = StreamingPlayer()
            activity = SpeechActivity()
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(send_microphone(connection, activity))
                tasks.create_task(receive_microphone_answers(connection, player, activity))
                tasks.create_task(player.run())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已退出")
