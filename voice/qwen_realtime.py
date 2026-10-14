"""Qwen3.8 Omni 语音 MVP：16kHz 麦克风 → 云端 → 24kHz 流式播放。

运行：cd alpha && .venv/bin/python -m voice.qwen_realtime
.env：DASHSCOPE_API_KEY、QWEN_WORKSPACE_ID。
可选：QWEN_REGION=singapore/beijing、QWEN_VOICE=Tina。
带噪音确认、暂停恢复、费用统计和自动重连；目前不连接 HA 工具。
"""

import asyncio
import base64
import json
import os
import time
from contextlib import AsyncExitStack
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from uuid import uuid4
from decimal import Decimal
from websockets.exceptions import ConnectionClosed

import sounddevice as sd  # type: ignore[import-untyped]
from dotenv import load_dotenv

from anthropic.types import ToolParam
from mcp import StdioServerParameters
from mcp.types import TextContent
from core.paths import ENV_FILE, PROJECT_ROOT, VENV_PYTHON
from integrations.mcp_client import load_personal_catalog, format_error
from integrations.personal_worker import PersonalWorker, stop_worker
from websockets.asyncio.client import connect

from voice.audio import (
    StreamingPlayer, audio_device, SpeechActivity, SpeechCandidate, is_meaningful,
)


MODEL = "qwen3.8-omni-flash-realtime"
INPUT_RATE = 16000
PERSONAL_TOOLS = {"list_items", "add_items", "complete_items", "create_reminder"}


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


def session_config(voice: str, tools: list[ToolParam] | None = None) -> dict:
    # 这里只组装配置字典，不建立连接、不录音，也不调用工具；session.update 才发送它。
    # 工具描述让模型决定“想调用什么”，真正执行仍要经过本地白名单和 worker。
    return {
        "modalities": ["text", "audio"],
        "voice": voice,
        "instructions": (
            "你是一个中文语音助手。无论用户说哪种语言，都使用普通话回答。"
            "回答简短自然，用一两句口语表达，不使用 Markdown、列表和表情符号。"
            "你可以用工具管理购物清单、待办和提醒。"
            '“X 买了”表示把清单上的 X 标记为完成。不要声称做了工具没有执行成功的事。'
            "没有家居设备操作工具，不能声称已控制设备。"
            + ("" if tools else "个人服务暂时不可用，不能声称已完成清单或提醒操作。")
        ),
        "tools": [
            {"type": "function", "function": {
                "name": tool["name"], "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            }} for tool in tools or [] if tool["name"] in PERSONAL_TOOLS
        ],
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


async def send_microphone(ws, activity: SpeechActivity) -> None:
    loop = asyncio.get_running_loop()
    while True:
        # 音频线程产生数据，异步任务取数据并发送；队列把两者的速度差暂时缓冲起来。
        # 每块 20ms，最多存 100 块，即 2 秒；不是无限积压、越说延迟越大。
        queue: asyncio.Queue[bytes | Exception] = asyncio.Queue(maxsize=100)
        accepting = True

        def enqueue(chunk: bytes, status: str) -> None:
            # 这个函数由事件循环执行，所以这里可以安全操作 asyncio.Queue。
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
            # sounddevice 在音频线程调用 callback，不能直接操作异步队列。
            # call_soon_threadsafe 把 enqueue 安排到事件循环；不在音频线程做网络发送。
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
                        # 没有音频时只挂起当前发送任务，接收模型消息和播放仍能继续。
                        chunk = await asyncio.wait_for(queue.get(), timeout=1)
                    except TimeoutError as exc:
                        raise sd.PortAudioError("麦克风 1 秒未返回音频") from exc
                    if not stream.active:
                        raise sd.PortAudioError("麦克风流已停止")
                    if isinstance(chunk, Exception):
                        raise chunk
                    # PCM 是二进制，base64 将它变成能放进 JSON 的文本；没有先转成文字。
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
    # 会话到期可以新建连接恢复；参数或凭据错误需要修正，不能靠无限重连解决。
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


class QwenReceiver:
    def __init__(self, ws, player: StreamingPlayer, activity: SpeechActivity,
                 costs: CostTracker, session_id: str, worker: PersonalWorker | None = None,
                 tools: list[ToolParam] | None = None):
        self.ws, self.player, self.activity = ws, player, activity
        self.costs, self.session_id = costs, session_id
        self.started = time.perf_counter()
        self.threshold_ms = int(os.environ.get("INTERRUPT_MS", "400"))
        if not 300 <= self.threshold_ms <= 500:
            raise ValueError("INTERRUPT_MS 必须在 300 到 500 之间")
        # candidate 记录用户这次发声；current_response 记录模型正在生成的回答。
        # 两者会重叠：模型说话时用户可能开口，也可能只是咳嗽。
        self.candidate: SpeechCandidate | None = None
        self.timer: asyncio.Task | None = None
        self.stopped_at: float | None = None
        self.current_response: str | None = None
        # 已发送 response.create、尚未收到 response.created 时为 True，防止重复请求。
        self.response_pending = False
        self.requested_candidate: SpeechCandidate | None = None
        self.last_meaningful_candidate: SpeechCandidate | None = None
        self.interrupted: set[str] = set()
        self.deleted: set[str] = set()
        self.pending_commands: dict[str, str] = {}
        self.first_audio = True
        self.worker = worker
        self.tool_names = {tool["name"] for tool in tools or []} & PERSONAL_TOOLS
        self.timezone = ZoneInfo(os.environ.get("REMINDER_TIMEZONE", "America/Los_Angeles"))
        # call_id 去重，response_id 按一轮回答分组：一轮可能同时要求多个工具。
        # pending 归零且该轮 response.done 后，才继续请求模型组织回答。
        self.tool_calls: set[str] = set()
        self.tool_pending: dict[str, int] = {}
        self.tool_candidates: dict[str, SpeechCandidate | None] = {}
        self.finished_responses: dict[str, str] = {}
        self.tool_continued: set[str] = set()

    def log(self, text: str) -> None:
        print(f"[{time.perf_counter() - self.started:.3f}] {text}", flush=True)

    async def command(self, kind: str, **payload) -> None:
        event_id = await send_event(self.ws, kind, **payload)
        self.pending_commands[event_id] = kind
        # 仅保留近期事件的关联信息；服务器不会为每个成功命令回传 event_id。
        if len(self.pending_commands) > 256:
            self.pending_commands.pop(next(iter(self.pending_commands)))

    async def maybe_respond(self) -> None:
        # 所有“现在能不能请模型回答”的判断集中在这里；单个事件不直接无条件触发回答。
        if self.current_response is not None or self.response_pending:
            return
        target = self.candidate
        if (target is not None and target.committed and target.meaningful is True
                and not target.responded):
            target.responded = True
            self.log("客户端发送 response.create（已提交且文字有意义）")
            await self.request_response(target)
            return
        # 工具已完成时继续旧问题；若用户已经提出新的有效问题，就不播过时的后续回答。
        for response_id, pending in self.tool_pending.items():
            if (pending == 0 and self.finished_responses.get(response_id) == "completed"
                    and response_id not in self.tool_continued
                    and (self.tool_candidates[response_id] is self.candidate
                         or (self.candidate is not None and self.candidate.meaningful is False
                             and self.tool_candidates[response_id] is self.last_meaningful_candidate))):
                self.tool_continued.add(response_id)
                await self.request_response(self.tool_candidates[response_id])
                return

    async def request_response(self, target: SpeechCandidate | None) -> None:
        # 先占住“已请求”状态，再执行 await；其他异步任务此时也可能检查能否回答。
        self.response_pending = True
        self.requested_candidate = target
        now = datetime.now(self.timezone)
        weekday = "一二三四五六日"[now.weekday()]
        text = f"[当前时间：{now:%Y-%m-%d} 星期{weekday} {now:%H:%M}，{self.timezone.key}]"
        # 时间是会话消息，不改动静态 instructions；首次推理也能正确理解“明天”。
        await self.command("conversation.item.create", item={
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": text}],
        })
        await self.command("response.create")

    def schedule_tool(self, event: dict, timers: asyncio.TaskGroup) -> None:
        # 这里只登记调用并安排任务，马上返回事件循环，不等 Google API 完成。
        call_id, response_id = event["call_id"], event["response_id"]
        if call_id in self.tool_calls or response_id in self.interrupted:
            return
        self.tool_calls.add(call_id)
        self.tool_candidates.setdefault(response_id, self.requested_candidate)
        self.tool_pending[response_id] = self.tool_pending.get(response_id, 0) + 1
        timers.create_task(self.execute_tool(event), name=f"qwen-tool-{event['name']}")

    async def execute_tool(self, event: dict) -> None:
        # 模型给的是函数名和 JSON 参数，并没有直接执行 Python 函数的权限。
        # 校验后经 PersonalWorker 发到个人 MCP server，等待属于这次调用的结果。
        name, call_id, response_id = event["name"], event["call_id"], event["response_id"]
        arguments = event.get("arguments", "")
        try:
            if name not in self.tool_names or name not in PERSONAL_TOOLS:
                raise ValueError(f"工具不在白名单或未加载：{name}")
            arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")
            if self.worker is None:
                raise RuntimeError("个人服务暂时不可用")
            result = await self.worker.call(name, arguments)
            text = "\n".join(item.text for item in result.content if isinstance(item, TextContent))
            if not text and result.structured_content is not None:
                text = json.dumps(result.structured_content, ensure_ascii=False)
            output = {"is_error": bool(result.is_error), "text": text or "工具未返回文本"}
        except Exception as exc:
            output = {"is_error": True, "text": (
                f"工具执行失败：{format_error(exc)}。操作可能已生效，写入操作不要盲目重试。"
            )}
        preview = str(output["text"])[:100]
        self.log(f"[工具] {name} {json.dumps(arguments, ensure_ascii=False)} → {preview}")
        # call_id 将返回值与模型原先请求的调用配对；output 是文本形式的 JSON。
        await self.command("conversation.item.create", item={
            "type": "function_call_output", "call_id": call_id,
            "output": json.dumps(output, ensure_ascii=False),
        })
        self.tool_pending[response_id] -= 1
        await self.maybe_respond()

    def pause_candidate(self, target: SpeechCandidate) -> None:
        # VAD 只证明“像持续人声”，还不能说明内容有意义，所以先暂停而不取消回答。
        if self.candidate is not target or target.paused or target.meaningful is not None:
            return
        target.paused = True
        self.player.pause()
        self.log(f"[暂停] 本地 VAD 人声 {self.activity.voiced_ms}ms，保留缓冲区，等待转录")

    async def observe_candidate(self, target: SpeechCandidate) -> None:
        # 每 20ms 看一次本地 VAD 状态；sleep 会让出事件循环，不阻塞网络事件接收。
        while self.candidate is target and not target.stopped and target.meaningful is None:
            if self.activity.voiced_ms >= self.threshold_ms:
                self.pause_candidate(target)
                return
            recent = time.perf_counter() - self.activity.last_voice_at < .15
            self.player.duck(recent)
            await asyncio.sleep(.02)

    async def confirm(self, target: SpeechCandidate) -> None:
        # 转写有意义后才真正打断：清空本地未播内容、取消生成、同步服务端已听位置。
        if self.candidate is not target or target.confirmed:
            return
        target.confirmed = True
        self.last_meaningful_candidate = target
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
        # 服务端的消息一条条到达；按 type 更新状态，不假定转写一定早于音频提交。
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
            # 先降低音量，保留旧回答；新的确认任务观察这次发声是否持续够久。
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
            # committed 表示服务端已将输入音频提交成会话条目，不代表已判断文字有意义。
            if self.candidate is not None and self.candidate.item_id == event["item_id"]:
                self.candidate.committed = True
                await self.maybe_respond()
        elif kind == "conversation.item.input_audio_transcription.completed":
            # 转写决定这次发声是否值得回答；噪音恢复旧播放，正常话语确认打断。
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
            # delta 是一小块新音频，不是完整回答；解码后入队，由播放器回调逐块消费。
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
        elif kind == "response.function_call_arguments.done":
            # 参数已完整才执行，不能拿 arguments.delta 的半截 JSON 去调用工具。
            self.schedule_tool(event, timers)
        elif kind == "response.done":
            response = event["response"]
            # done.output 兼作完整事件缺失时的兜底，call_id 去重避免重复写入。
            if response.get("status") == "completed" and response["id"] not in self.interrupted:
                for item in response.get("output", []):
                    if item.get("type") == "function_call":
                        self.schedule_tool({**item, "response_id": response["id"]}, timers)
            self.finished_responses[response["id"]] = (
                "cancelled" if response["id"] in self.interrupted else response.get("status", "")
            )
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
        # 一个任务专门收 WebSocket 消息；timers 管理确认任务和工具任务的生命周期。
        # 连接异常时 TaskGroup 取消同组后台任务，外层决定是否重连。
        try:
            async with asyncio.TaskGroup() as timers:
                async for message in self.ws:
                    await self.handle(json.loads(message), timers)
                raise SessionEnded("Qwen WebSocket 连接已关闭")
        finally:
            if self.timer:
                self.timer.cancel()
            self.player.resume()


def verify_session(session: dict, tools: list[ToolParam] | None = None) -> None:
    # 检查的是服务端确认后的配置，避免只看本地“发出了什么”就以为配置已经生效。
    detection = session.get("turn_detection") or {}
    if detection.get("create_response") is not False or detection.get("interrupt_response") is not False:
        raise QwenProtocolError(
            "服务端未确认 create_response=False / interrupt_response=False。"
            "不能保证暂停后噪音恢复，请检查模型/接口支持。实际 turn_detection="
            + json.dumps(detection, ensure_ascii=False)
        )
    if tools:
        expected = {tool["name"] for tool in tools} & PERSONAL_TOOLS
        actual = {tool.get("function", {}).get("name") for tool in session.get("tools", [])
                  if tool.get("type") == "function"}
        if not expected <= actual:
            raise QwenProtocolError(f"服务端未确认个人工具：{'、'.join(sorted(expected - actual))}")


def reconnectable(exc: BaseException) -> bool:
    # TaskGroup 可能把多个异常包成 ExceptionGroup；逐层检查，不能把真正的配置错误忽略。
    if isinstance(exc, BaseExceptionGroup):
        return all(reconnectable(error) for error in exc.exceptions)
    return isinstance(exc, (SessionEnded, ConnectionClosed, OSError, TimeoutError))


async def rotate_session(seconds: float) -> None:
    # 定时任务只负责到点通知；抛出的 SessionEnded 交给外层重连循环处理。
    await asyncio.sleep(seconds)
    raise SessionEnded("达到配置的会话轮换时间")


async def run_session(ws, player: StreamingPlayer, costs: CostTracker,
                      voice: str, region: str, session_seconds: float,
                      worker: PersonalWorker | None = None, tools: list[ToolParam] | None = None) -> None:
    # 每个新连接都重新完成“服务端建会话 → 发送配置 → 等确认”，之后才启动收音。
    created = await wait_for_event(ws, "session.created")
    session_id = created["session"]["id"]
    print(f"[会话] {session_id} ｜ {MODEL} ｜ {region}")
    await send_event(ws, "session.update", session=session_config(voice, tools))
    updated = await wait_for_event(ws, "session.updated")
    verify_session(updated["session"], tools)
    print(f"[配置] 中文 ｜ {voice} ｜ 16kHz 输入/24kHz 输出 ｜ semantic_vad ｜ 客户端控制回答与打断")
    activity = SpeechActivity(sample_rate=INPUT_RATE)
    receiver = QwenReceiver(ws, player, activity, costs, session_id, worker, tools)
    async with asyncio.TaskGroup() as tasks:
        # create_task 安排并发执行，不是为每个协程新开线程；这里不逐个 await 到结束。
        # 三个任务分别负责发麦克风音频、收服务端事件，以及定时结束本次会话。
        tasks.create_task(send_microphone(ws, activity))
        tasks.create_task(receiver.run())
        tasks.create_task(rotate_session(session_seconds))


async def reconnect_loop(url: str, key: str, player: StreamingPlayer,
                         costs: CostTracker, voice: str, region: str,
                         session_seconds: float = 1800, worker: PersonalWorker | None = None,
                         tools: list[ToolParam] | None = None) -> None:
    if session_seconds <= 0:
        raise ValueError("QWEN_SESSION_SECONDS 必须大于 0")
    while True:
        # 这个循环管理“一个会话”的生灭；播放器和 PersonalWorker 在外层持续运行。
        try:
            started = time.perf_counter()
            async with connect(url, additional_headers={"Authorization": f"Bearer {key}"},
                               open_timeout=30, max_size=16 * 1024 * 1024) as ws:
                print(f"[连接] WebSocket {time.perf_counter() - started:.2f}s")
                await run_session(ws, player, costs, voice, region, session_seconds, worker, tools)
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
    # 描述如何启动个人 server 子进程；此时还没启动。-m 让它按项目包结构导入模块。
    server = StdioServerParameters(
        command=str(VENV_PYTHON), args=["-m", "servers.personal_server"], cwd=str(PROJECT_ROOT),
    )
    async with AsyncExitStack() as stack:
        # 目录发现是一次短连接；worker 随后持有自己的长期连接，用于实际执行工具。
        catalog = await load_personal_catalog(server)
        tools = [tool for tool in catalog or [] if tool["name"] in PERSONAL_TOOLS]
        worker = PersonalWorker(server)
        # worker.run 会一直等待队列请求，所以安排为后台任务，不能在这里直接 await 它结束。
        worker_task = asyncio.create_task(worker.run(), name="qwen-personal-worker")
        # 登记退出清理：即使发生异常，也会取消并等待 worker 退出，关闭它的 MCP 子进程。
        stack.push_async_callback(stop_worker, worker_task)
        print("[个人工具] " + ("、".join(tool["name"] for tool in tools) or "暂不可用"))
        async with asyncio.TaskGroup() as tasks:
            # 打开并维持扬声器输出流；真正取缓冲区音频的是音频库调用的 player.callback。
            tasks.create_task(player.run())
            # 同时运行云端连接；它可以不断重连，而不用反复打开播放器和个人 server。
            tasks.create_task(reconnect_loop(
                url, key, player, costs, voice, region,
                float(os.environ.get("QWEN_SESSION_SECONDS", "1800")), worker, tools,
            ))


if __name__ == "__main__":
    try:
        # 创建并运行事件循环；普通调用 main() 只会得到协程对象，不会执行其中的流程。
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已退出")
