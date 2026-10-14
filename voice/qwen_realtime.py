"""Qwen3.8 Omni 语音 MVP：16kHz 麦克风 → 云端 → 24kHz 流式播放。

运行：alpha/.venv/bin/python alpha/realtime_qwen.py
.env：DASHSCOPE_API_KEY、QWEN_WORKSPACE_ID。
可选：QWEN_REGION=singapore/beijing、QWEN_VOICE=Tina。
带噪音确认、暂停恢复、费用统计和自动重连；目前不连接 HA 工具。
"""

import asyncio
import base64
import json
import os
import time
from urllib.parse import urlencode
from uuid import uuid4
from decimal import Decimal
from websockets.exceptions import ConnectionClosed

import sounddevice as sd  # type: ignore[import-untyped]
from dotenv import load_dotenv

from core.paths import ENV_FILE
from websockets.asyncio.client import connect

from voice.audio import (
    StreamingPlayer, audio_device, SpeechActivity, SpeechCandidate, is_meaningful,
)


MODEL = "qwen3.8-omni-flash-realtime"
INPUT_RATE = 16000


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


def session_config(voice: str) -> dict:
    return {
        "modalities": ["text", "audio"],
        "voice": voice,
        "instructions": (
            "你是一个中文语音助手。无论用户说哪种语言，都使用普通话回答。"
            "回答简短自然，用一两句口语表达，不使用 Markdown、列表和表情符号。"
            "本次只进行语音聊天，没有设备操作工具，不能声称已控制设备。"
        ),
        "audio": {
            "input": {"format": {
                "type": "pcm", "sample_rate": 16000,
                "sample_format": "s16le", "channels": 1,
                "packing": "interleaved", "channel_layout": "mono",
            }},
            "output": {"format": {"type": "pcm", "sample_rate": 24000}},
        },
        "input_audio_transcription": {"model": "qwen3-asr-flash-realtime"},
        "turn_detection": {
            "type": "semantic_vad", "create_response": False, "interrupt_response": False,
        },
    }


async def send_event(ws, event_type: str, **payload) -> str:
    event_id = "event_" + uuid4().hex
    await ws.send(json.dumps({"event_id": event_id, "type": event_type, **payload}, ensure_ascii=False))
    return event_id


async def wait_for_event(ws, expected: str) -> dict:
    async with asyncio.timeout(30):
        while True:
            event = json.loads(await ws.recv())
            if event["type"] == "error":
                raise_api_error(event.get("error", {}))
            if event["type"] == expected:
                return event


async def send_microphone(ws, activity: SpeechActivity) -> None:
    loop = asyncio.get_running_loop()
    while True:
        queue: asyncio.Queue[bytes | Exception] = asyncio.Queue(maxsize=100)
        accepting = True

        def enqueue(chunk: bytes, status: str) -> None:
            nonlocal accepting
            if not accepting:
                return
            if status:
                print(f"[麦克风] {status}")
            activity.observe(chunk)
            if queue.full():
                accepting = False
                queue.get_nowait()
                queue.put_nowait(RuntimeError("音频发送积压超过 2 秒，请检查网络"))
            else:
                queue.put_nowait(chunk)

        def callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(
                enqueue, indata.astype("<i2", copy=False).tobytes(),
                str(status) if status else "",
            )

        try:
            with sd.InputStream(
                samplerate=INPUT_RATE, channels=1, dtype="int16", blocksize=320,
                device=audio_device("REALTIME_INPUT_DEVICE"), callback=callback,
            ) as stream:
                print("麦克风已开启。直接说中文，模型自动判断说完；Ctrl+C 退出。建议戴耳机。")
                while True:
                    try:
                        chunk = await asyncio.wait_for(queue.get(), timeout=1)
                    except TimeoutError as exc:
                        raise sd.PortAudioError("麦克风 1 秒未返回音频") from exc
                    if not stream.active:
                        raise sd.PortAudioError("麦克风流已停止")
                    if isinstance(chunk, Exception):
                        raise chunk
                    await send_event(ws, "input_audio_buffer.append",
                                     audio=base64.b64encode(chunk).decode("ascii"))
        except (sd.PortAudioError, ValueError) as exc:
            print(f"[麦克风不可用] {exc}。1 秒后重试。")
            await send_event(ws, "input_audio_buffer.clear")
            activity.reset()
        finally:
            accepting = False
        await asyncio.sleep(1)


class SessionEnded(RuntimeError):
    """可重新建立连接的会话结束或过期。"""


class QwenProtocolError(RuntimeError):
    """配置、凭据或接口错误，不能靠无限重连解决。"""


def raise_api_error(error: dict) -> None:
    code = str(error.get("code", ""))
    message = str(error.get("message", ""))
    lowered = (code + " " + message).lower()
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
        key = (session_id, response["id"])
        if key in self.seen:
            return None
        self.seen.add(key)
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


class QwenReceiver:
    def __init__(self, ws, player: StreamingPlayer, activity: SpeechActivity,
                 costs: CostTracker, session_id: str):
        self.ws, self.player, self.activity = ws, player, activity
        self.costs, self.session_id = costs, session_id
        self.started = time.perf_counter()
        self.threshold_ms = int(os.environ.get("INTERRUPT_MS", "400"))
        if not 300 <= self.threshold_ms <= 500:
            raise ValueError("INTERRUPT_MS 必须在 300 到 500 之间")
        self.candidate: SpeechCandidate | None = None
        self.timer: asyncio.Task | None = None
        self.stopped_at: float | None = None
        self.current_response: str | None = None
        self.response_pending = False
        self.requested_candidate: SpeechCandidate | None = None
        self.interrupted: set[str] = set()
        self.deleted: set[str] = set()
        self.pending_commands: dict[str, str] = {}
        self.first_audio = True

    def log(self, text: str) -> None:
        print(f"[{time.perf_counter() - self.started:.3f}] {text}", flush=True)

    async def command(self, kind: str, **payload) -> None:
        event_id = await send_event(self.ws, kind, **payload)
        self.pending_commands[event_id] = kind
        # 仅保留近期事件的关联信息；服务器不会为每个成功命令回传 event_id。
        if len(self.pending_commands) > 256:
            self.pending_commands.pop(next(iter(self.pending_commands)))

    async def maybe_respond(self) -> None:
        target = self.candidate
        if (target is not None and target.committed and target.meaningful is True
                and not target.responded and self.current_response is None and not self.response_pending):
            target.responded = True
            self.response_pending = True
            self.requested_candidate = target
            self.log("客户端发送 response.create（已提交且文字有意义）")
            await self.command("response.create")

    def pause_candidate(self, target: SpeechCandidate) -> None:
        if self.candidate is not target or target.paused or target.meaningful is not None:
            return
        target.paused = True
        self.player.pause()
        self.log(f"[暂停] 本地 VAD 人声 {self.activity.voiced_ms}ms，保留缓冲区，等待转录")

    async def observe_candidate(self, target: SpeechCandidate) -> None:
        while self.candidate is target and not target.stopped and target.meaningful is None:
            if self.activity.voiced_ms >= self.threshold_ms:
                self.pause_candidate(target)
                return
            recent = time.perf_counter() - self.activity.last_voice_at < .15
            self.player.duck(recent)
            await asyncio.sleep(.02)

    async def confirm(self, target: SpeechCandidate) -> None:
        if self.candidate is not target or target.confirmed:
            return
        target.confirmed = True
        item = self.player.clear()
        self.log("[确认] 有意义的输入 → 清空缓冲区、取消并截断旧回答")
        if self.current_response:
            self.interrupted.add(self.current_response)
            # Qwen 文档的 response.cancel 只要求 type，不传 OpenAI 的 response_id。
            await self.command("response.cancel")
        if item is not None:
            item_id, index, played_ms = item
            # Qwen 当前文档未列出 truncate；拒绝时明确报告，不能假定与 OpenAI 兼容。
            await self.command("conversation.item.truncate", item_id=item_id,
                               content_index=index, audio_end_ms=played_ms)
        await self.maybe_respond()

    async def delete_noise(self, item_id: str) -> None:
        if item_id not in self.deleted:
            self.deleted.add(item_id)
            await self.command("conversation.item.delete", item_id=item_id)
            self.log(f"[删除噪音] item_id={item_id}")

    async def handle(self, event: dict, timers: asyncio.TaskGroup) -> None:
        kind = event["type"]
        if kind == "error":
            error = event.get("error", {})
            operation = self.pending_commands.pop(error.get("event_id", ""), "")
            code = str(error.get("code", ""))
            if operation == "response.cancel" and code in {"response_cancel_not_active", "response_not_active"}:
                self.log("[取消] 回答已经结束，继续处理")
                self.current_response = None
                await self.maybe_respond()
                return
            if operation in {"conversation.item.truncate", "conversation.item.delete"}:
                raise QwenProtocolError(
                    f"Qwen 拒绝 {operation}，不能保证服务端历史与本地对话一致："
                    + json.dumps(error, ensure_ascii=False)
                )
            raise_api_error(error)
        if kind == "input_audio_buffer.speech_started":
            if self.timer:
                self.timer.cancel()
            self.candidate = SpeechCandidate(event["item_id"], event["audio_start_ms"])
            self.player.duck(True)
            self.stopped_at = None
            self.log("speech_started → 降低音量到 20%，开始确认")
            self.timer = timers.create_task(self.observe_candidate(self.candidate))
        elif kind == "input_audio_buffer.speech_stopped":
            self.stopped_at = time.perf_counter()
            target = self.candidate
            if target is not None and target.item_id == event["item_id"]:
                target.stopped = True
                if self.timer:
                    self.timer.cancel()
                self.log(f"speech_stopped（持续 {(event['audio_end_ms'] - target.audio_start_ms) / 1000:.2f}s）")
                if self.activity.voiced_ms >= self.threshold_ms:
                    self.pause_candidate(target)
                elif not target.paused and not target.confirmed:
                    self.player.duck(False)
                    self.log("[恢复音量] 短促声响，仍等待文字")
        elif kind == "input_audio_buffer.committed":
            if self.candidate is not None and self.candidate.item_id == event["item_id"]:
                self.candidate.committed = True
                await self.maybe_respond()
        elif kind == "conversation.item.input_audio_transcription.completed":
            text = event.get("transcript", "").strip()
            meaningful = is_meaningful(text)
            self.log(f"你：{text!r} ｜ meaningful={meaningful}")
            target = self.candidate
            if target is not None and target.item_id == event["item_id"]:
                target.meaningful = meaningful
                if self.timer:
                    self.timer.cancel()
                if meaningful:
                    await self.confirm(target)
                else:
                    self.player.resume()
                    self.log("[恢复] 只是噪音，继续播放")
                    await self.delete_noise(target.item_id)
                await self.maybe_respond()
            elif not meaningful:
                await self.delete_noise(event["item_id"])
        elif kind == "conversation.item.input_audio_transcription.failed":
            self.log(f"[转录失败] {event.get('error')}")
            if self.candidate is not None and self.candidate.item_id == event["item_id"]:
                self.candidate.meaningful = False
                if self.timer:
                    self.timer.cancel()
                self.player.resume()
        elif kind == "conversation.item.deleted":
            self.log(f"conversation.item.deleted（{event['item_id']}）")
        elif kind == "response.created":
            self.current_response = event["response"]["id"]
            self.first_audio = True
            self.log(f"response.created（客户端请求={self.response_pending}）")
            if not self.response_pending:
                raise QwenProtocolError("服务端自动创建了回答，create_response=False 未生效")
            self.response_pending = False
            # 旧的 response.create 与新的用户输入交错。
            if (self.candidate is not None and self.candidate is not self.requested_candidate
                    and self.candidate.confirmed):
                self.interrupted.add(self.current_response)
                await self.command("response.cancel")
        elif kind == "response.audio.delta":
            if event.get("response_id") in self.interrupted:
                return
            chunk = base64.b64decode(event["delta"])
            if chunk:
                if self.first_audio and self.stopped_at is not None:
                    self.log(f"[延迟] speech_stopped 到首段音频：{time.perf_counter() - self.stopped_at:.2f}s")
                self.first_audio = False
                self.player.append(chunk, event["item_id"], event.get("content_index", 0))
        elif kind == "response.audio_transcript.done":
            if event.get("response_id") not in self.interrupted:
                self.log(f"模型：{event.get('transcript', '')}")
        elif kind == "response.done":
            response = event["response"]
            self.log(f"response.done：{response.get('status')}")
            self.log("[usage] " + json.dumps(response.get("usage"), ensure_ascii=False))
            duplicate = (self.session_id, response["id"]) in self.costs.seen
            cost = self.costs.record(self.session_id, response)
            if not duplicate:
                if cost is None:
                    self.log(f"[费用] usage 明细不足，未计价；已知累计 ${self.costs.total:.8f}，未计价 {self.costs.unpriced} 轮")
                else:
                    self.log(f"[费用] 本轮 ${cost:.8f} ｜ 累计 ${self.costs.total:.8f} USD（标准价估算）")
            if self.current_response == response["id"]:
                self.current_response = None
            self.interrupted.discard(response["id"])
            await self.maybe_respond()

    async def run(self) -> None:
        try:
            async with asyncio.TaskGroup() as timers:
                async for message in self.ws:
                    await self.handle(json.loads(message), timers)
                raise SessionEnded("Qwen WebSocket 连接已关闭")
        finally:
            if self.timer:
                self.timer.cancel()
            self.player.resume()


def verify_session(session: dict) -> None:
    detection = session.get("turn_detection") or {}
    if detection.get("create_response") is not False or detection.get("interrupt_response") is not False:
        raise QwenProtocolError(
            "服务端未确认 create_response=False / interrupt_response=False。"
            "不能保证暂停后噪音恢复，请检查模型/接口支持。实际 turn_detection="
            + json.dumps(detection, ensure_ascii=False)
        )


def reconnectable(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return all(reconnectable(error) for error in exc.exceptions)
    return isinstance(exc, (SessionEnded, ConnectionClosed, OSError, TimeoutError))


async def rotate_session(seconds: float) -> None:
    await asyncio.sleep(seconds)
    raise SessionEnded("达到配置的会话轮换时间")


async def run_session(ws, player: StreamingPlayer, costs: CostTracker,
                      voice: str, region: str, session_seconds: float) -> None:
    created = await wait_for_event(ws, "session.created")
    session_id = created["session"]["id"]
    print(f"[会话] {session_id} ｜ {MODEL} ｜ {region}")
    await send_event(ws, "session.update", session=session_config(voice))
    updated = await wait_for_event(ws, "session.updated")
    verify_session(updated["session"])
    print(f"[配置] 中文 ｜ {voice} ｜ 16kHz 输入/24kHz 输出 ｜ semantic_vad ｜ 客户端控制回答与打断")
    activity = SpeechActivity(sample_rate=INPUT_RATE)
    receiver = QwenReceiver(ws, player, activity, costs, session_id)
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(send_microphone(ws, activity))
        tasks.create_task(receiver.run())
        tasks.create_task(rotate_session(session_seconds))


async def reconnect_loop(url: str, key: str, player: StreamingPlayer,
                         costs: CostTracker, voice: str, region: str,
                         session_seconds: float = 1800) -> None:
    if session_seconds <= 0:
        raise ValueError("QWEN_SESSION_SECONDS 必须大于 0")
    while True:
        try:
            started = time.perf_counter()
            async with connect(url, additional_headers={"Authorization": f"Bearer {key}"},
                               open_timeout=30, max_size=16 * 1024 * 1024) as ws:
                print(f"[连接] WebSocket {time.perf_counter() - started:.2f}s")
                await run_session(ws, player, costs, voice, region, session_seconds)
        except Exception as exc:
            if not reconnectable(exc):
                raise
            print("[重连] 会话已结束，重新连接")
            print(f"[重连原因] {exc}；新会话的对话历史将重新开始，累计费用保留。")
        finally:
            player.clear()
        await asyncio.sleep(1)


async def main() -> None:
    load_dotenv(ENV_FILE)
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    workspace = os.environ.get("QWEN_WORKSPACE_ID", "").strip()
    if not key or not workspace:
        raise SystemExit("请在 alpha/.env 中设置 DASHSCOPE_API_KEY 和 QWEN_WORKSPACE_ID")
    region = os.environ.get("QWEN_REGION", "singapore")
    voice = os.environ.get("QWEN_VOICE", "Tina")
    url = websocket_url(workspace, region)
    costs = CostTracker(region)
    player = StreamingPlayer()
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(player.run())
        tasks.create_task(reconnect_loop(
            url, key, player, costs, voice, region,
            float(os.environ.get("QWEN_SESSION_SECONDS", "1800")),
        ))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已退出")
