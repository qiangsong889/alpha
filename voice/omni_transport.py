"""Omni 后备收音/播报的传输与协议辅助；不包含业务工具执行入口。"""
import asyncio
import base64
import json
import time
from decimal import Decimal
from urllib.parse import urlencode
from uuid import uuid4

import sounddevice as sd  # type: ignore[import-untyped]
from websockets.exceptions import ConnectionClosed

from core.settings import setting
from voice.audio import SpeechActivity, audio_device

MODEL = "qwen3.8-omni-flash-realtime"
INPUT_RATE = 16000
AUDIO_SEND_TIMEOUT_S = 2.0

def websocket_url(workspace: str, region: str) -> str:
    regions = {"singapore": "ap-southeast-1", "beijing": "cn-beijing"}
    if region not in regions:
        raise ValueError("QWEN_REGION 必须是 singapore 或 beijing")
    if not workspace or not all(char.isalnum() or char == "-" for char in workspace):
        raise ValueError("QWEN_WORKSPACE_ID 必须是实际工作空间 ID")
    return (
        f"wss://{workspace}.{regions[region]}.maas.aliyuncs.com/api-ws/v1/realtime?"
        + urlencode({"model": MODEL})
    )

def turn_detection_config() -> dict:
    vad = setting("QWEN_VAD")
    if vad not in {"server_vad", "semantic_vad"}:
        raise ValueError("QWEN_VAD 必须是 server_vad 或 semantic_vad")
    try:
        silence_ms = setting("QWEN_SILENCE_MS")
    except ValueError as exc:
        raise ValueError("QWEN_SILENCE_MS 必须是 200 到 6000 的整数") from exc
    if not 200 <= silence_ms <= 6000:
        raise ValueError("QWEN_SILENCE_MS 必须是 200 到 6000 的整数")
    config = {"type": vad, "create_response": False, "interrupt_response": False}
    if vad == "server_vad":
        config["silence_duration_ms"] = silence_ms
    return config

def session_config(voice: str) -> dict:
    """仅配置语音输入/输出；Qwen不接收任何业务或控制工具。"""
    return {
        "modalities": ["text", "audio"], "voice": voice, "instructions": "", "tools": [],
        "audio": {
            "input": {"format": {
                "type": "pcm", "sample_rate": 16000,
                "sample_format": "s16le", "channels": 1,
                "packing": "interleaved", "channel_layout": "mono",
            }},
            "output": {"format": {"type": "pcm", "sample_rate": 24000}},
        },
        "input_audio_transcription": {"model": "qwen3-asr-flash-realtime"},
        "turn_detection": turn_detection_config(),
    }


async def send_event(ws, event_type: str, **payload) -> str:
    # Qwen 的 WebSocket 消息用 JSON 表达；event_id 用于追踪一次客户端发送。
    # await ws.send 只等消息发送，不等模型完整回答；回答由接收任务另行处理。
    event_id = "event_" + uuid4().hex
    await ws.send(json.dumps({"event_id": event_id, "type": event_type, **payload}, ensure_ascii=False))
    return event_id

async def wait_for_event(ws, expected: str) -> dict:
    # 启动阶段先等服务端确认配置，再开始传音频；30 秒超时会向外抛异常。
    async with asyncio.timeout(30):
        while True:
            event = json.loads(await ws.recv())
            if event["type"] == "error":
                raise_api_error(event.get("error", {}))
            if event["type"] == expected:
                return event

async def send_microphone(ws, activity: SpeechActivity, *, on_sent=None, upload_gate=None) -> None:
    loop = asyncio.get_running_loop()
    while True:
        # 音频线程产生数据，异步任务取数据并发送；队列把两者的速度差暂时缓冲起来。
        # 每块 20ms，最多存 100 块，即 2 秒；不是无限积压、越说延迟越大。
        if upload_gate is not None:
            upload_gate.reset()
        queue: asyncio.Queue[bytes | Exception] = asyncio.Queue(maxsize=100)
        accepting = True
        received = sent = 0
        max_dispatch_delay = 0.0

        def enqueue(chunk: bytes, status: str, captured_at: float) -> None:
            # 这个函数由事件循环执行，所以这里可以安全操作 asyncio.Queue。
            nonlocal accepting, received, max_dispatch_delay
            if not accepting:
                return
            received += 1
            max_dispatch_delay = max(max_dispatch_delay, time.perf_counter() - captured_at)
            if status:
                print(f"[麦克风] {status}")
            activity.observe(chunk)
            if queue.full():
                accepting = False
                # 不继续把两秒旧音频送到服务器，也不把拥塞当作程序致命错误。
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(SessionEnded(
                    "音频发送积压超过 2 秒，丢弃旧音频并重建连接；"
                    f"收到 {received} 块、已发送 {sent} 块、最大回调调度延迟 {max_dispatch_delay:.2f}s"))
            else:
                queue.put_nowait(chunk)

        def callback(indata, frames, time_info, status):
            # sounddevice 在音频线程调用 callback，不能直接操作异步队列。
            # call_soon_threadsafe 把 enqueue 安排到事件循环；不在音频线程做网络发送。
            loop.call_soon_threadsafe(
                enqueue, indata.astype("<i2", copy=False).tobytes(),
                str(status) if status else "",
                time.perf_counter(),
            )

        try:
            with sd.InputStream(
                samplerate=INPUT_RATE, channels=1, dtype="int16", blocksize=320,
                device=audio_device("REALTIME_INPUT_DEVICE"), callback=callback,
            ) as stream:
                print("麦克风已开启。直接说中文，模型自动判断说完；Ctrl+C 退出。建议戴耳机。")
                while True:
                    try:
                        # 没有音频时只挂起当前发送任务，接收模型消息和播放仍能继续。
                        async with asyncio.timeout(1):
                            chunk = await queue.get()
                    except TimeoutError as exc:
                        raise sd.PortAudioError("麦克风 1 秒未返回音频") from exc
                    if not stream.active:
                        raise sd.PortAudioError("麦克风流已停止")
                    if isinstance(chunk, Exception):
                        raise chunk
                    if upload_gate is not None:
                        outgoing = upload_gate.feed(chunk)
                        if outgoing is None:
                            continue  # 持续消费本地录音；不让静音积压进网络队列。
                        chunk = outgoing
                    # PCM 是二进制，base64 将它变成能放进 JSON 的文本；没有先转成文字。
                    try:
                        async with asyncio.timeout(AUDIO_SEND_TIMEOUT_S):
                            await send_event(ws, "input_audio_buffer.append",
                                             audio=base64.b64encode(chunk).decode("ascii"))
                        sent += 1
                        if on_sent is not None:
                            on_sent(chunk)
                    except TimeoutError as exc:
                        raise SessionEnded(
                            f"音频发送超时（上限 {AUDIO_SEND_TIMEOUT_S:.1f}s，队列 {queue.qsize()}/100），"
                            "丢弃旧音频并重建连接") from exc
        except (sd.PortAudioError, ValueError) as exc:
            print(f"[麦克风不可用] {exc}。1 秒后重试。")
            await send_event(ws, "input_audio_buffer.clear")
            activity.reset()
        finally:
            accepting = False
            if upload_gate is not None:
                upload_gate.report()
        await asyncio.sleep(1)

class SessionEnded(RuntimeError):
    """可重新建立连接的会话结束或过期。"""

class QwenProtocolError(RuntimeError):
    """配置、凭据或接口错误，不能靠无限重连解决。"""

def raise_api_error(error: dict) -> None:
    # 会话到期可以新建连接恢复；参数或凭据错误需要修正，不能靠无限重连解决。
    code = str(error.get("code", ""))
    message = str(error.get("message", ""))
    lowered = (code + " " + message).lower()
    if "turn queue overflow" in lowered:
        # 服务端轮次队列已经满了，旧连接不能继续可靠使用；重连但绝不重放写入工具。
        raise SessionEnded(message or code)
    if (code in {"session_expired", "session_timeout", "session_duration_exceeded", "session_closed"}
            or ("session" in lowered and any(word in lowered for word in
                                             ("expired", "maximum duration", "timed out")))):
        raise SessionEnded(f"{code}: {message}")
    raise QwenProtocolError(f"{code}: {message}")

class CostTracker:
    # USD / 1M tokens；Qwen3.8 同时计费音频输出和对应文字。
    # 2026-10-13：https://www.alibabacloud.com/help/en/model-studio/qwen3-8-omni-flash-realtime
    PRICES = {
        "singapore": {"input_text": "0.23", "input_audio": "0.93", "output_text": "0.70", "output_audio": "1.87"},
        "beijing": {"input_text": "0.212", "input_audio": "0.848", "output_text": "0.636", "output_audio": "1.696"},
    }

    def __init__(self, region: str):
        self.prices = {name: Decimal(value) for name, value in self.PRICES[region].items()}
        self.total = Decimal("0")
        self.seen: set[tuple[str, str]] = set()
        self.unpriced = 0

    def record(self, session_id: str, response: dict) -> Decimal | None:
        # 同一回答的 done 事件即使处理两次，也只能计费一次；跨重连仍使用同一计价器。
        key = (session_id, response["id"])
        if key in self.seen:
            return None
        self.seen.add(key)
        # 使用服务端返回的 token 明细估算标准价；缺明细时标记“未计价”，不当作免费。
        usage = response.get("usage")
        if not isinstance(usage, dict):
            self.unpriced += 1
            return None
        inputs, outputs = usage.get("input_tokens_details"), usage.get("output_tokens_details")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            self.unpriced += 1
            return None
        values = {
            "input_text": inputs.get("text_tokens"), "input_audio": inputs.get("audio_tokens"),
            "output_text": outputs.get("text_tokens"), "output_audio": outputs.get("audio_tokens"),
        }
        if any(type(value) is not int or value < 0 for value in values.values()):
            self.unpriced += 1
            return None
        counts = {name: int(value) for name, value in values.items() if isinstance(value, int)}
        cost = sum((Decimal(counts[name]) * rate for name, rate in self.prices.items()), Decimal("0")) / 1_000_000
        self.total += cost
        return cost

def verify_session(session: dict) -> None:
    # 检查的是服务端确认后的配置，避免只看本地“发出了什么”就以为配置已经生效。
    detection = session.get("turn_detection") or {}
    expected_detection = turn_detection_config()
    if (detection.get("type") != expected_detection["type"]
            or (expected_detection["type"] == "server_vad"
                and detection.get("silence_duration_ms") != expected_detection["silence_duration_ms"])):
        raise QwenProtocolError("服务端未确认请求的 VAD 配置：" + json.dumps(detection, ensure_ascii=False))
    if detection.get("create_response") is not False or detection.get("interrupt_response") is not False:
        raise QwenProtocolError(
            "服务端未确认 create_response=False / interrupt_response=False。"
            "不能保证暂停后噪音恢复，请检查模型/接口支持。实际 turn_detection="
            + json.dumps(detection, ensure_ascii=False)
        )


def reconnectable(exc: BaseException) -> bool:
    # TaskGroup 可能把多个异常包成 ExceptionGroup；逐层检查，不能把真正的配置错误忽略。
    if isinstance(exc, BaseExceptionGroup):
        return all(reconnectable(error) for error in exc.exceptions)
    return isinstance(exc, (SessionEnded, ConnectionClosed, OSError, TimeoutError))

async def rotate_session(seconds: float) -> None:
    # 定时任务只负责到点通知；抛出的 SessionEnded 交给外层重连循环处理。
    await asyncio.sleep(seconds)
    raise SessionEnded("达到配置的会话轮换时间")

