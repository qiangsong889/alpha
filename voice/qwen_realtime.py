"""Qwen3.8 Omni 语音 MVP：16kHz 麦克风 → 云端 → 24kHz 流式播放。

运行：cd alpha && .venv/bin/python -m voice.qwen_realtime
.env：DASHSCOPE_API_KEY、QWEN_WORKSPACE_ID。
可选：QWEN_REGION=singapore/beijing、QWEN_VOICE=Tina。
带噪音确认、暂停恢复、费用统计和自动重连；支持个人工具和 HA 设备工具。
"""

import asyncio
import base64
import json
import os
import re
import sys
import time
from typing import Any
import httpx2
from contextlib import AsyncExitStack
from datetime import datetime
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from uuid import uuid4
from decimal import Decimal
from websockets.exceptions import ConnectionClosed

import sounddevice as sd  # type: ignore[import-untyped]
from dotenv import load_dotenv

from anthropic.types import ToolParam
from mcp import StdioServerParameters
from core.paths import ENV_FILE, PROJECT_ROOT, VENV_PYTHON, OPERATIONS_DB
from core.plan import needs_plan
from core.plan_coordinator import PlanCoordinator
from core.task_coordinator import TaskCoordinator
from core.simple_coordinator import SimpleCoordinator
from integrations.mcp_client import (
    load_personal_catalog, format_error, load_ha_catalog, merge_tools,
)
from integrations.operations import OperationStore
from integrations.tool_executor import (
    HA_ACTION_TOOLS, HA_DEVICE_TOOLS, PERSONAL_TOOLS, make_executor, format_for_model,
)
from integrations.tool_gateway import ToolGateway
from integrations.tool_policy import get_policy
from integrations.personal_worker import PersonalWorker, stop_worker
from websockets.asyncio.client import connect

from voice.audio import (
    StreamingPlayer, audio_device, SpeechActivity, SpeechCandidate, is_meaningful,
)
from voice.commit_window import CommitReport, CommitWindow
from voice.plan_tools import call_control, control_output
from voice.batch_tools import CONTROL_TOOLS as LEGACY_CONTROL_TOOLS
from voice.simple_tools import CONTROL_TOOLS, RULES


@dataclass
class _Recap:
    plan_id: str
    version: int
    text: str
    items: list[str] = field(default_factory=list)
    invalid: bool = False
    checked: bool = False


MODEL = "qwen3.8-omni-flash-realtime"
INPUT_RATE = 16000
AUDIO_SEND_TIMEOUT_S = 2.0
def claims_completion(text: str) -> bool:
    """有限的中文完成声明检查；跳过否定和引用，不是完整语义验证。"""
    text = re.sub(r'[“「『"].*?[”」』"]', "", text)
    for clause in re.split(r"[，。！？；,!?;\n]", text):
        if re.search(r"没有|还没|未能|尚未|并未|不能|无法|不代表|不是|不会|未完成|未添加|未设置|未删除|未撤销|未恢复|不删|不撤|不恢复|不取消", clause):
            continue
        if re.search(
            r"(?:已经|已)[^，。！？；,!?;\n]{0,12}?(?:设好|设置|安排|创建|添加|加了|加好|加到|加进|加入|加回|放回|放进|放到|补回|完成|标记|勾选|打开|关闭|关掉|删除|删掉|撤销|撤掉|拿掉|移除|恢复|取消)"
            r"|(?:设置好|标记好|设好|添加好|加好|加回|放回|放进|补回|完成|删除|删掉|撤销|撤掉|拿掉|移除|恢复|取消)了", clause,
        ):
            return True
    return False


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
    vad = os.environ.get("QWEN_VAD", "server_vad").strip()
    if vad not in {"server_vad", "semantic_vad"}:
        raise ValueError("QWEN_VAD 必须是 server_vad 或 semantic_vad")
    try:
        silence_ms = int(os.environ.get("QWEN_SILENCE_MS", "800"))
    except ValueError as exc:
        raise ValueError("QWEN_SILENCE_MS 必须是 200 到 6000 的整数") from exc
    if not 200 <= silence_ms <= 6000:
        raise ValueError("QWEN_SILENCE_MS 必须是 200 到 6000 的整数")
    config = {"type": vad, "create_response": False, "interrupt_response": False}
    if vad == "server_vad":
        config["silence_duration_ms"] = silence_ms
    return config


def session_config(voice: str, tools: list[ToolParam] | None = None,
                   assist_prompt: str = "") -> dict:
    # 这里只组装配置字典，不建立连接、不录音，也不调用工具；session.update 才发送它。
    # 工具描述让模型决定“想调用什么”，真正执行仍要经过本地白名单和 MCP 路由。
    names = {tool["name"] for tool in tools or [] if get_policy(tool["name"]) is not None}
    return {
        "modalities": ["text", "audio"],
        "voice": voice,
        "instructions": (
            RULES
            + ("" if names & PERSONAL_TOOLS else "个人服务暂时不可用，不能声称已完成清单或提醒。")
            + ("" if names & HA_DEVICE_TOOLS else "家居设备暂时不可用，不能声称已操作设备。")
            + ("\nHome Assistant Assist prompt：\n" + assist_prompt if assist_prompt else "")
        ),
        "tools": [
            {"type": "function", "function": {
                "name": tool["name"], "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            }} for tool in tools or [] if get_policy(tool["name"]) is not None
        ] + [{"type": "function", "function": {"name": name, **tool}}
             for name, tool in CONTROL_TOOLS.items()],
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


async def send_microphone(ws, activity: SpeechActivity) -> None:
    loop = asyncio.get_running_loop()
    while True:
        # 音频线程产生数据，异步任务取数据并发送；队列把两者的速度差暂时缓冲起来。
        # 每块 20ms，最多存 100 块，即 2 秒；不是无限积压、越说延迟越大。
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
                        chunk = await asyncio.wait_for(queue.get(), timeout=1)
                    except TimeoutError as exc:
                        raise sd.PortAudioError("麦克风 1 秒未返回音频") from exc
                    if not stream.active:
                        raise sd.PortAudioError("麦克风流已停止")
                    if isinstance(chunk, Exception):
                        raise chunk
                    # PCM 是二进制，base64 将它变成能放进 JSON 的文本；没有先转成文字。
                    try:
                        async with asyncio.timeout(AUDIO_SEND_TIMEOUT_S):
                            await send_event(ws, "input_audio_buffer.append",
                                             audio=base64.b64encode(chunk).decode("ascii"))
                        sent += 1
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


class QwenReceiver:
    def __init__(self, ws, player: StreamingPlayer, activity: SpeechActivity,
                 costs: CostTracker, session_id: str, *, gateway: ToolGateway,
                 commit_window: CommitWindow,
                 coordinator: PlanCoordinator,
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
        self.gateway = gateway
        self.commit_window = commit_window
        self.coordinator = coordinator
        self.pending_recap: _Recap | None = None
        self.recaps: dict[str, _Recap] = {}
        self.response_items: dict[str, list[str]] = {}
        self.recap_needed = coordinator.plan is not None and coordinator.plan.status == "collecting"
        self.playback_tasks: set[asyncio.Task] = set()
        self.early_recap_turns: dict[str, tuple[str, int]] = {}
        self.finalized_responses: set[str] = set()
        self.turn_operations: dict[str, list[str]] = {}
        self.active_report: CommitReport | None = None
        self.report_response: str | None = None
        self.report_requested = False
        self.report_wakeup = asyncio.Event()
        self.tool_names = {tool["name"] for tool in tools or [] if get_policy(tool["name"]) is not None}
        self.timezone = ZoneInfo(os.environ.get("REMINDER_TIMEZONE", "America/Los_Angeles"))
        # call_id 去重，response_id 按一轮回答分组：一轮可能同时要求多个工具。
        # pending 归零且该轮 response.done 后，才继续请求模型组织回答。
        self.tool_calls: set[str] = set()
        self.tool_pending: dict[str, int] = {}
        self.tool_candidates: dict[str, SpeechCandidate | None] = {}
        self.finished_responses: dict[str, str] = {}
        self.tool_continued: set[str] = set()
        # 仅当这一轮全部是明确成功的设备操作时，静音处理工具后的模型回答。
        self.tool_silent: dict[str, bool] = {}
        # 静默表示用户不听到确认，不等于跳过服务端需要的工具后续轮次。
        self.silent_response_pending = False
        self.silent_responses: set[str] = set()
        self.response_requests = 0
        self._last_snapshot_text: str | None = None
        # 证据跨“调用工具”和“工具后续回答”两次 response 保留，但不跨用户请求。
        self.response_turns: dict[str, str] = {}
        self.turn_successes: dict[str, set[str]] = {}
        self.turn_attempts: dict[str, set[str]] = {}
        self.turn_pending: dict[str, int] = {}
        self.response_text: dict[str, dict[tuple[str, int], str]] = {}
        self.claims_to_check: dict[str, str] = {}
        self.checked_claims: set[str] = set()
        self.correction_counts: dict[str, int] = {}

    @staticmethod
    def turn_key(target: SpeechCandidate | None) -> str:
        return target.item_id if target is not None else "<no-user-item>"

    async def correct_unsupported_claim(self) -> bool:
        """等待本轮工具结束后核对声明；只为当前用户请求纠正一次，防止生成死循环。"""
        for response_id, text in self.claims_to_check.items():
            if response_id in self.recaps:
                continue  # 模板中可能有“标为已完成”，那是在复述计划，不是宣称已执行。
            if response_id in self.checked_claims:
                continue
            turn = self.response_turns[response_id]
            current = self.turn_key(self.candidate)
            if self.candidate is not None and self.candidate.meaningful is False:
                current = self.turn_key(self.last_meaningful_candidate)
            if turn != current or self.turn_pending.get(turn, 0):
                continue
            self.checked_claims.add(response_id)
            operations = [op for op_id in self.turn_operations.get(turn, [])
                          if (op := self.gateway.store.get(op_id)) is not None]
            pending = [op for op in operations if op.status in {"pending", "approved"}]
            if response_id == self.report_response and self.active_report is not None:
                if any(op.status == "succeeded" for op in self.active_report.operations):
                    continue
            succeeded = bool(self.turn_successes.get(turn)) or any(
                op.status == "succeeded" and (policy := get_policy(op.name)) is not None
                and policy.mutates for op in operations
            )
            if succeeded and not pending:
                continue
            self.log("[⚠️ 疑似虚构] 模型声称已完成，但本轮没有成功的工具调用")
            self.log(f"[核对] response_id={response_id} ｜ {text[:100]}")
            if self.correction_counts.get(turn, 0) >= 1:
                self.log("[核对] 已纠正一次，不再自动请求，避免重复调用或无限生成")
                continue
            self.correction_counts[turn] = 1
            attempted = bool(self.turn_attempts.get(turn)) or any(
                op.status in {"executing", "failed", "unknown"} for op in operations
            )
            correction = (
                "系统执行记录校验：你刚才宣称操作已完成，但本次用户请求没有成功的写入工具结果。"
                "必须先纠正刚才未经证实的说法，不得再次假称已完成。"
                + ("本轮已有写入尝试但结果失败或不明确，操作可能已生效。先查询状态或说明不确定，禁止盲目重复写入。"
                   if attempted else
                   "本轮没有执行写入工具。核对用户原始请求：若明确要求执行新操作，请调用对应工具后再回答；"
                   "若只是询问历史或讨论示例，不要重复执行操作。")
            )
            if pending:
                correction = (
                    "系统执行记录校验：以下操作已登记，还没有执行，稍后执行。"
                    "请纠正刚才的完成声明，只说已记下，不得再次调用工具或宣称完成："
                    + "、".join(f"#{op.id} {op.name}" for op in pending)
                )
            elif response_id == self.report_response and self.active_report is not None:
                correction = (
                    "系统执行记录校验：这些操作并没有成功，不得再次调用工具。"
                    "只根据以下真实状态纠正刚才的完成声明：\n" + self.active_report.text
                )
            # 当前纠正取代这轮原本的工具后续回答，避免稍后又发一次 response.create。
            self.tool_continued.update(r for r in self.tool_pending if self.response_turns.get(r) == turn)
            await self.request_response(self.candidate, correction=correction)
            return True
        return False

    def log(self, text: str) -> None:
        if (text.startswith(("[工具]", "[提交结果]", "[本地撤销]"))
                and sys.stdout.isatty() and "NO_COLOR" not in os.environ):
            text = f"\033[34m{text}\033[0m"
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
        effective = self.last_meaningful_candidate if target is not None and target.meaningful is False else target
        if self.turn_pending.get(self.turn_key(effective), 0):
            return  # 同一轮的 update/remove 全部完成后，才生成一次最新版本的复述。
        # 工具结果都已返回时先分流，再决定如何继续模型这一轮。
        # 多动作直接以计划复述完成工具后续轮次，不先生成“记下了/设好了”的自由回答。
        ready = [response_id for response_id, pending in self.tool_pending.items()
                 if pending == 0 and self.finished_responses.get(response_id) == "completed"
                 and response_id not in self.tool_continued
                 and (self.tool_candidates[response_id] is effective)]
        if (self.commit_window.pending and self.needs_plan_flow()
                and any(response_id not in self.recaps for response_id in ready)):
            self.tool_continued.update(ready)
            for response_id in ready:
                if response_id in self.claims_to_check:
                    self.log("[⚠️ 疑似虚构] 工具轮含未经核实的完成声明，直接用计划复述澄清")
                    self.checked_claims.add(response_id)
            await self.transfer_window_to_plan(effective)
            return
        if await self.correct_unsupported_claim():
            return
        # 工具已完成时继续旧问题；若用户已经提出新的有效问题，就不播过时的后续回答。
        for response_id, pending in self.tool_pending.items():
            if (pending == 0 and self.finished_responses.get(response_id) == "completed"
                    and response_id not in self.tool_continued
                    and (self.tool_candidates[response_id] is self.candidate
                         or (self.candidate is not None and self.candidate.meaningful is False
                             and self.tool_candidates[response_id] is self.last_meaningful_candidate))):
                self.tool_continued.add(response_id)
                if response_id in self.recaps:
                    # 拒绝不等于跳过协议收尾，否则连续违规会积压服务端轮次。
                    await self.request_response(self.tool_candidates[response_id], silent=True)
                    return
                if self.recap_needed and self.active_plan():
                    await self.request_recap(self.tool_candidates[response_id])
                    return
                if self.tool_silent.get(response_id, False):
                    self.log("[静默] 设备操作全部成功，完成后续轮次但不播放确认")
                    await self.request_response(self.tool_candidates[response_id], silent=True)
                    return
                await self.request_response(self.tool_candidates[response_id])
                return
        # 报告只在没有新话语、工具后续回答和纠正回答时送出，避免自己打断自己。
        if target is not None and (target.meaningful is None
                                   or (target.meaningful is True and not target.responded)):
            return
        if self.recap_needed and self.active_plan():
            await self.request_recap(target)
            return
        if self.active_report is None and not self.commit_window.reports.empty():
            self.active_report = self.commit_window.reports.get_nowait()
        if self.active_report is not None and self.report_response is None:
            self.report_requested = True
            self.log("[提交结果] " + self.active_report.text)
            await self.request_response(target, correction=(
                "系统执行结果（以这里的状态为准）：\n" + self.active_report.text
                + "\n只根据这份执行结果简短报告，不要再次执行工具；失败、未知或取消不得说成功。"
            ))

    async def request_response(self, target: SpeechCandidate | None, *, silent: bool = False,
                               correction: str = "") -> None:
        if self.current_response is not None or self.response_pending:
            return
        # 先占住“已请求”状态，再执行 await；其他异步任务此时也可能检查能否回答。
        self.response_pending = True
        self.silent_response_pending = silent
        self.requested_candidate = target
        self.response_requests += 1
        self.log(f"[轮次] response.create #{self.response_requests}（{'静默收尾' if silent else '正常回答'}）")
        if correction:
            await self.command("conversation.item.create", item={
                "type": "message", "role": "system",
                "content": [{"type": "input_text", "text": correction}],
            })
        if not silent:
            snapshot = self.coordinator.snapshot()
            if snapshot is not None:
                encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
                if encoded != self._last_snapshot_text:
                    await self.command("conversation.item.create", item={
                        "type": "message", "role": "system", "content": [{
                            "type": "input_text", "text": "当前操作记录：" + encoded,
                        }],
                    })
                    self._last_snapshot_text = encoded
            now = datetime.now(self.timezone)
            weekday = "一二三四五六日"[now.weekday()]
            text = f"[当前时间：{now:%Y-%m-%d} 星期{weekday} {now:%H:%M}，{self.timezone.key}]"
            # 时间作为系统上下文，不替代用户原始指令；静默收尾不增加时间条目。
            # Omni 当前文档未明确列出 system message 支持，服务端拒绝时会明确报错。
            await self.command("conversation.item.create", item={
                "type": "message", "role": "system",
                "content": [{"type": "input_text", "text": text}],
            })
        await self.command("response.create")

    def active_plan(self) -> bool:
        plan = self.coordinator.plan
        return plan is not None and plan.status in {"collecting", "awaiting_confirmation"}

    def coordinator_turn(self, target: SpeechCandidate | None) -> str:
        return f"{self.session_id}:{self.turn_key(target)}"

    async def request_recap(self, target: SpeechCandidate | None) -> None:
        if self.current_response is not None or self.response_pending or not self.active_plan():
            return
        plan = self.coordinator.plan
        assert plan is not None
        self.recap_needed = False
        if not plan.items:
            return
        self.pending_recap = _Recap(plan.id, plan.version, plan.recap(datetime.now(self.timezone)))
        # 即使是同一个版本再次复述，也必须先撤销上次的播放资格。
        self.coordinator.recap_played(plan.id, plan.version, interrupted=True)
        self.log(f"[计划复述] {plan.id} v{plan.version}：{self.pending_recap.text}")
        await self.request_response(target, correction=(
            "本轮只做计划复述。逐字念出下面这段话，不要增加或删减内容，不要调用工具：\n"
            + self.pending_recap.text
        ))

    async def wait_recap_played(self, response_id: str) -> None:
        recap = self.recaps[response_id]
        played = bool(recap.items) and not recap.invalid
        if played:
            played = all(await asyncio.gather(*(self.player.wait_played(item) for item in recap.items)))
        result = self.coordinator.recap_played(
            recap.plan_id, recap.version, interrupted=not played or recap.invalid,
        )
        recap.checked = True
        self.log(f"[计划播放] v{recap.version} played={played} reason={result.reason}")

    async def close_playback_tasks(self) -> None:
        tasks = tuple(self.playback_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def schedule_tool(self, event: dict, timers: asyncio.TaskGroup) -> None:
        # 这里只登记调用并安排任务，马上返回事件循环，不等 Google API 完成。
        call_id, response_id = event["call_id"], event["response_id"]
        if (call_id in self.tool_calls or response_id in self.interrupted
                or response_id in self.silent_responses):
            return
        self.tool_calls.add(call_id)
        if response_id in self.recaps:
            self.recaps[response_id].invalid = True
        self.tool_candidates.setdefault(response_id, self.requested_candidate)
        turn = self.response_turns.setdefault(response_id, self.turn_key(self.tool_candidates[response_id]))
        self.turn_pending[turn] = self.turn_pending.get(turn, 0) + 1
        self.tool_pending[response_id] = self.tool_pending.get(response_id, 0) + 1
        self.tool_silent.setdefault(response_id, True)
        timers.create_task(self.execute_tool(event), name=f"qwen-tool-{event['name']}")

    async def execute_tool(self, event: dict) -> None:
        # 模型给的是函数名和 JSON 参数，并没有直接执行 Python 函数的权限。
        # 只把请求交给 Gateway；路由、真实调用和持久执行状态都不再由语音层直接处理。
        name, call_id, response_id = event["name"], event["call_id"], event["response_id"]
        arguments = event.get("arguments", "")
        silent = False
        output: dict[str, Any]
        turn = self.response_turns[response_id]
        try:
            arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")
            if response_id in self.recaps:
                self.recaps[response_id].invalid = True
                snapshot = self.coordinator.snapshot()
                output = {"is_error": True, "reason": "recap_only", "snapshot": snapshot,
                          "version": snapshot["version"] if snapshot else None,
                          "text": "复述这一轮禁止调用工具，本次没有执行"}
            elif name in LEGACY_CONTROL_TOOLS:
                result = await call_control(self.coordinator, name, arguments)
                output = control_output(result)
                if result.ok and name in {"update_plan_item", "remove_plan_item"}:
                    self.recap_needed = True
                if result.ok and name == "cancel_plan":
                    self.recap_needed = False
                for op in result.operations:
                    self.turn_operations.setdefault(turn, []).append(op.id)
                    if op.status in {"succeeded", "failed", "unknown"}:
                        self.turn_attempts.setdefault(turn, set()).add(op.id)
                    if op.status == "succeeded":
                        self.turn_successes.setdefault(turn, set()).add(op.id)
            else:
                policy = get_policy(name)
                if name not in self.tool_names or policy is None:
                    raise ValueError(f"工具不在白名单或未加载：{name}")
                op = await self.gateway.submit(f"{self.session_id}:{turn}", name, arguments)
                self.turn_operations.setdefault(turn, []).append(op.id)
                duplicate = None
                if policy.commit in {"deferred", "confirm"} and op.status == "pending":
                    duplicate = (self.gateway.store.recent_succeeded(name, arguments)
                                 if policy.commit == "deferred" else None)
                    if duplicate is not None:
                        self.gateway.cancel(op.id)
                        op = self.gateway.store.get(op.id) or op
                    else:
                        self.commit_window.add(op)
                silent = op.status == "succeeded" and name in HA_ACTION_TOOLS
                if policy.mutates and op.status in {"executing", "succeeded", "failed", "unknown"}:
                    self.turn_attempts.setdefault(turn, set()).add(call_id)
                if policy.mutates and op.status == "succeeded":
                    self.turn_successes.setdefault(turn, set()).add(call_id)
                output = dict(format_for_model(op, duplicate_of=duplicate))
        except Exception as exc:
            output = {"is_error": True, "text": (
                f"工具执行失败：{format_error(exc)}。操作可能已生效，写入操作不要盲目重试。"
            )}
            if name in LEGACY_CONTROL_TOOLS:
                snapshot = self.coordinator.snapshot()
                output.update(reason="invalid_arguments" if isinstance(exc, (ValueError, TypeError)) else "control_error",
                              snapshot=snapshot, version=snapshot["version"] if snapshot else None)
        preview = str(output["text"])[:100]
        self.log(f"[工具] {name} {json.dumps(arguments, ensure_ascii=False)} → {preview}")
        # call_id 将返回值与模型原先请求的调用配对；output 是文本形式的 JSON。
        await self.command("conversation.item.create", item={
            "type": "function_call_output", "call_id": call_id,
            "output": json.dumps(output, ensure_ascii=False),
        })
        # 查询、个人工具、异常或 action_done 以外的结果都保留模型后续回答。
        self.tool_silent[response_id] = self.tool_silent.get(response_id, True) and silent
        self.tool_pending[response_id] -= 1
        self.turn_pending[turn] -= 1
        await self.maybe_respond()
        await self.maybe_finish_turn()
        self.report_wakeup.set()

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
        self.commit_window.speech_accepted()
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
        try:
            await self._handle(event, timers)
            await self.maybe_finish_turn()
        finally:
            self.report_wakeup.set()

    async def maybe_finish_turn(self) -> None:
        """回答结束和噪音转写可能乱序；两种事件之后都重新检查提交条件。"""
        if self.current_response is not None or self.response_pending:
            return
        target = self.candidate
        if target is not None:
            if target.meaningful is None:
                return  # 先等文字；不能把尚未识别的新话语当成噪音。
            if target.meaningful is False:
                target = self.last_meaningful_candidate
            elif not target.responded:
                return
        turn = self.turn_key(target)
        if self.turn_pending.get(turn, 0):
            return
        completed = [response_id for response_id, status in self.finished_responses.items()
                     if status == "completed" and self.response_turns.get(response_id) == turn
                     and response_id not in self.tool_pending
                     and response_id not in self.finalized_responses]
        if not completed:
            return
        self.finalized_responses.update(completed)
        self.coordinator.turn_answered(self.coordinator_turn(target))
        if all(response_id in self.recaps for response_id in completed):
            return  # 复述的生成结束不是再次发起复述，也不等于扬声器播完。
        if not self.needs_plan_flow():
            self.commit_window.turn_finished()
            return
        await self.transfer_window_to_plan(target)

    def needs_plan_flow(self) -> bool:
        try:
            to_plan = self.active_plan() or needs_plan(self.commit_window.pending)
        except ValueError:
            to_plan = True  # 无效参数交给 collect 拒绝并取消，不能落入自动执行。
        if self.coordinator.plan is not None and self.coordinator.plan.status == "executing":
            to_plan = True
        return to_plan

    async def transfer_window_to_plan(self, target: SpeechCandidate | None) -> None:
        rejected = []
        for op in self.commit_window.take_all():
            result = self.coordinator.collect(op)
            if not result.ok:
                self.gateway.cancel(op.id)
                rejected.append(f"#{op.id} {op.name}：这一项没有加进计划（{result.reason}），"
                                f"当前状态 {(self.gateway.store.get(op.id) or op).status}")
        if rejected:
            text = "\n".join(rejected)
            self.log("[计划登记] " + text)
            await self.command("conversation.item.create", item={
                "type": "message", "role": "system", "content": [{"type": "input_text", "text": text}],
            })
        if self.active_plan():
            self.recap_needed = True
            await self.request_recap(target)
        elif rejected:
            await self.request_response(target, correction="只根据刚才的计划登记失败结果报告，不要重新调用工具。")

    async def _handle(self, event: dict, timers: asyncio.TaskGroup) -> None:
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
            if operation == "conversation.item.create" and (
                "role" in str(error).lower() or "system" in str(error).lower()
            ):
                raise QwenProtocolError(
                    "Qwen 拒绝系统上下文条目；当前 Omni 接口的 system 消息兼容性需要确认："
                    + json.dumps(error, ensure_ascii=False)
                )
            raise_api_error(error)
        if kind == "input_audio_buffer.speech_started":
            self.commit_window.speech_started()
            # 先降低音量，保留旧回答；新的确认任务观察这次发声是否持续够久。
            if self.timer:
                self.timer.cancel()
            self.candidate = SpeechCandidate(event["item_id"], event["audio_start_ms"])
            open_recap = self.pending_recap or next(
                (recap for recap in reversed(list(self.recaps.values())) if not recap.checked), None,
            )
            if open_recap is not None:
                self.early_recap_turns[event["item_id"]] = (open_recap.plan_id, open_recap.version)
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
            user_text = f"你：{text!r}"
            if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
                user_text = f"\033[32m{user_text}\033[0m"
            self.log(f"{user_text} ｜ meaningful={meaningful}")
            target = self.candidate
            if target is not None and target.item_id == event["item_id"]:
                target.meaningful = meaningful
                if self.timer:
                    self.timer.cancel()
                if meaningful:
                    early = self.early_recap_turns.pop(target.item_id, None)
                    if early is not None:
                        self.coordinator.recap_played(*early, interrupted=True)
                        for recap in self.recaps.values():
                            if (recap.plan_id, recap.version) == early:
                                recap.invalid = True
                    self.coordinator.user_turn(self.coordinator_turn(target))
                    await self.confirm(target)
                else:
                    self.commit_window.speech_ignored()
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
                self.commit_window.speech_ignored()
                if self.timer:
                    self.timer.cancel()
                self.player.resume()
        elif kind == "conversation.item.deleted":
            self.log(f"conversation.item.deleted（{event['item_id']}）")
        elif kind == "response.created":
            self.current_response = event["response"]["id"]
            if self.pending_recap is not None:
                self.recaps[self.current_response] = self.pending_recap
                self.pending_recap = None
            if self.report_requested:
                self.report_response = self.current_response
                self.report_requested = False
            self.response_turns[self.current_response] = self.turn_key(self.requested_candidate)
            self.first_audio = True
            self.log(f"response.created（客户端请求={self.response_pending}）")
            if not self.response_pending:
                raise QwenProtocolError("服务端自动创建了回答，create_response=False 未生效")
            if self.silent_response_pending:
                self.silent_responses.add(self.current_response)
            self.silent_response_pending = False
            self.response_pending = False
            # 旧的 response.create 与新的用户输入交错。
            if (self.candidate is not None and self.candidate is not self.requested_candidate
                    and self.candidate.confirmed):
                self.interrupted.add(self.current_response)
                await self.command("response.cancel")
        elif kind == "response.audio.delta":
            # delta 是一小块新音频，不是完整回答；解码后入队，由播放器回调逐块消费。
            if event.get("response_id") in self.interrupted | self.silent_responses:
                return
            chunk = base64.b64decode(event["delta"])
            if chunk:
                if self.first_audio and self.stopped_at is not None:
                    self.log(f"[延迟] speech_stopped 到首段音频：{time.perf_counter() - self.stopped_at:.2f}s")
                self.first_audio = False
                items = self.response_items.setdefault(event["response_id"], [])
                if event["item_id"] not in items:
                    items.append(event["item_id"])
                self.player.append(chunk, event["item_id"], event.get("content_index", 0))
        elif kind == "response.audio.done":
            self.player.finish(event["item_id"])
        elif kind in {"response.audio_transcript.delta", "response.audio_transcript.done", "response.text.delta", "response.text.done"}:
            response_id = event["response_id"]
            key = (event.get("item_id", ""), event.get("content_index", 0))
            parts = self.response_text.setdefault(response_id, {})
            if kind.endswith(".delta"):
                parts[key] = parts.get(key, "") + event.get("delta", "")
            else:
                parts[key] = event.get("transcript", event.get("text", ""))
            if event.get("response_id") not in self.interrupted | self.silent_responses:
                if kind.endswith(".done"):
                    self.log(f"模型：{parts[key]}")
        elif kind == "response.function_call_arguments.done":
            # 参数已完整才执行，不能拿 arguments.delta 的半截 JSON 去调用工具。
            self.schedule_tool(event, timers)
        elif kind == "response.done":
            response = event["response"]
            items = self.response_items.setdefault(response["id"], [])
            for item in response.get("output", []):
                if (item.get("type") == "message" and item.get("id")
                        and any(part.get("type") == "audio" for part in item.get("content", []))
                        and item["id"] not in items):
                    items.append(item["id"])
            for item_id in items:
                self.player.finish(item_id)
            self.response_turns.setdefault(response["id"], self.turn_key(self.requested_candidate))
            silent_response = response["id"] in self.silent_responses
            # 优先读最终 response.output；若服务端只在转写事件中给文字，则使用按 response_id 收集的内容。
            final_text = "".join(
                part.get("transcript", part.get("text", ""))
                for item in response.get("output", []) if item.get("type") == "message"
                for part in item.get("content", [])
            ) or "".join(self.response_text.pop(response["id"], {}).values())
            self.response_text.pop(response["id"], None)
            if (response.get("status") == "completed" and not silent_response
                    and response["id"] not in self.interrupted and claims_completion(final_text)):
                self.claims_to_check[response["id"]] = final_text
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
            self.silent_responses.discard(response["id"])
            if silent_response:
                if any(item.get("type") == "function_call" for item in response.get("output", [])):
                    # 收尾只允许完成回答，不能因为模型重复调用而再次执行设备写入。
                    raise SessionEnded("静默收尾再次请求工具，未执行重复操作，重新建立会话")
                self.log("[静默] 收尾完成，继续监听下一句")
            if response["id"] in self.recaps:
                recap = self.recaps[response["id"]]
                recap.items = list(items)
                recap.invalid |= self.finished_responses[response["id"]] != "completed"
                self.coordinator.turn_answered(f"{self.session_id}:{self.response_turns[response['id']]}")
                if not recap.checked and not any(
                    task.get_name() == f"recap-played-{response['id']}" for task in self.playback_tasks
                ):
                    task = (timers.create_task(self.wait_recap_played(response["id"]),
                                               name=f"recap-played-{response['id']}") if timers is not None
                            else asyncio.create_task(self.wait_recap_played(response["id"]),
                                                     name=f"recap-played-{response['id']}"))
                    self.playback_tasks.add(task)
                    task.add_done_callback(self.playback_tasks.discard)
            await self.maybe_respond()
            if response["id"] == self.report_response:
                # 只有完整回答才算报告送达；打断/断线时把事实留给下一次连接。
                if self.finished_responses[response["id"]] == "completed":
                    self.active_report = None
                self.report_response = None

    async def watch_reports(self) -> None:
        while True:
            # 队列不属于 receiver；重连时未报告的结果不会丢失。
            if self.active_report is None:
                self.active_report = await self.commit_window.reports.get()
            self.report_wakeup.clear()
            await self.maybe_respond()
            if self.active_report is not None:
                await self.report_wakeup.wait()

    async def run(self) -> None:
        # 一个任务专门收 WebSocket 消息；timers 管理确认任务和工具任务的生命周期。
        # 连接异常时 TaskGroup 取消同组后台任务，外层决定是否重连。
        try:
            async with asyncio.TaskGroup() as timers:
                timers.create_task(self.watch_reports())
                await self.maybe_respond()  # 重连后将保留的计划重新复述。
                async for message in self.ws:
                    await self.handle(json.loads(message), timers)
                raise SessionEnded("Qwen WebSocket 连接已关闭")
        finally:
            await self.close_playback_tasks()
            if self.active_report is not None:
                self.commit_window.reports.put_nowait(self.active_report)
                self.active_report = None
            if self.timer:
                self.timer.cancel()
            self.player.resume()


def verify_session(session: dict, tools: list[ToolParam] | None = None) -> None:
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
    if tools is not None:
        expected = {tool["name"] for tool in tools if get_policy(tool["name"]) is not None} | set(CONTROL_TOOLS)
        actual = {tool.get("function", {}).get("name") for tool in session.get("tools", [])
                  if tool.get("type") == "function"}
        if not expected <= actual:
            raise QwenProtocolError(f"服务端未确认工具：{'、'.join(sorted(expected - actual))}")


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
                      tools: list[ToolParam] | None = None, *, gateway: ToolGateway,
                      commit_window: Any,
                      coordinator: Any,
                      assist_prompt: str = "") -> None:
    # 每个新连接都重新完成“服务端建会话 → 发送配置 → 等确认”，之后才启动收音。
    created = await wait_for_event(ws, "session.created")
    session_id = created["session"]["id"]
    print(f"[会话] {session_id} ｜ {MODEL} ｜ {region}")
    await send_event(ws, "session.update", session=session_config(voice, tools, assist_prompt))
    updated = await wait_for_event(ws, "session.updated")
    verify_session(updated["session"], tools)
    detection = updated["session"]["turn_detection"]
    vad_label = detection["type"] + (f" / 静音 {detection['silence_duration_ms']}ms" if detection["type"] == "server_vad" else "")
    print(f"[配置] 中文 ｜ {voice} ｜ 16kHz 输入/24kHz 输出 ｜ {vad_label} ｜ 客户端控制回答与打断")
    activity = SpeechActivity(sample_rate=INPUT_RATE)
    if isinstance(coordinator, SimpleCoordinator):
        from voice.simple_receiver import receiver_type
        receiver = receiver_type(QwenReceiver, claims_completion=claims_completion)(
            ws, player, activity, costs, session_id, gateway=gateway, workflow=coordinator, tools=tools)
    elif isinstance(coordinator, TaskCoordinator):
        from voice.batch_receiver import receiver_type
        receiver = receiver_type(QwenReceiver, claims_completion=claims_completion)(ws, player, activity, costs, session_id,
                                               gateway=gateway, workflow=coordinator, tools=tools)
    else:
        # 兼容旧的离线回归夹具；生产 main 只创建 SimpleCoordinator。
        receiver = QwenReceiver(ws, player, activity, costs, session_id, gateway=gateway,
                                commit_window=commit_window, coordinator=coordinator, tools=tools)
    async with asyncio.TaskGroup() as tasks:
        # create_task 安排并发执行，不是为每个协程新开线程；这里不逐个 await 到结束。
        # 三个任务分别负责发麦克风音频、收服务端事件，以及定时结束本次会话。
        tasks.create_task(send_microphone(ws, activity))
        tasks.create_task(receiver.run())
        tasks.create_task(rotate_session(session_seconds))


async def reconnect_loop(url: str, key: str, player: StreamingPlayer,
                         costs: CostTracker, voice: str, region: str,
                         session_seconds: float = 1800, tools: list[ToolParam] | None = None, *,
                         gateway: ToolGateway, commit_window: Any,
                         coordinator: Any,
                         assist_prompt: str = "") -> None:
    if session_seconds <= 0:
        raise ValueError("QWEN_SESSION_SECONDS 必须大于 0")
    while True:
        # 这个循环管理“一个会话”的生灭；播放器和 PersonalWorker 在外层持续运行。
        try:
            started = time.perf_counter()
            async with connect(url, additional_headers={"Authorization": f"Bearer {key}"},
                               open_timeout=30, max_size=16 * 1024 * 1024) as ws:
                print(f"[连接] WebSocket {time.perf_counter() - started:.2f}s")
                await run_session(ws, player, costs, voice, region, session_seconds, tools,
                                  gateway=gateway, commit_window=commit_window,
                                  coordinator=coordinator, assist_prompt=assist_prompt)
        except Exception as exc:
            if not reconnectable(exc):
                raise
            print("[重连] 会话已结束，重新连接")
            print(f"[重连原因] {format_error(exc)}；本次连接持续 {time.perf_counter() - started:.2f}s；"
                  "新会话的对话历史将重新开始，累计费用保留。")
        finally:
            coordinator.on_reconnect()
            player.clear()
        await asyncio.sleep(1)


async def main() -> None:
    load_dotenv(ENV_FILE)
    turn_detection_config()  # 在启动 MCP/联网之前检查本地 VAD 参数。
    print(f"[运行代码] {__file__} ｜ 本地购物路由 v1")
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    workspace = os.environ.get("QWEN_WORKSPACE_ID", "").strip()
    ha_root = os.environ.get("HA_URL", "").strip()
    ha_token = os.environ.get("HA_TOKEN", "").strip()
    ha_url = ha_root.rstrip("/") + "/api/mcp"

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
        OPERATIONS_DB.parent.mkdir(parents=True, exist_ok=True)
        store = OperationStore(OPERATIONS_DB)
        stack.callback(store.close)
        for recovered in store.recover_on_startup():
            print(f"[恢复] 操作 {recovered.id} {recovered.name}：executing → unknown，不自动重试")
        for discarded in store.discard_unsubmitted():
            print(f"[恢复] 旧进程未提交操作 {discarded.id} 已取消，不恢复旧计划")
        # HTTP 客户端跨 Qwen 重连复用，离开 stack 才关闭；HA 会话则在每次工具调用时创建。
        ha_http_client = await stack.enter_async_context(httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {ha_token}"},
            timeout=httpx2.Timeout(30.0, read=300.0),
        ))
        # 目录发现是一次短连接；worker 随后持有自己的长期连接，用于实际执行工具。
        personal_catalog = await load_personal_catalog(server)
        ha_catalog = await load_ha_catalog(ha_url, ha_http_client) if ha_root and ha_token else None
        # 先检查全部目录的重名，再按来源和白名单筛选，避免工具被覆盖或路由到错误服务器。
        merged, owners = merge_tools({
            "personal": personal_catalog or [],
            "home_assistant": ha_catalog[0] if ha_catalog is not None else [],
        })
        tools = [tool for tool in merged if (
            owners[tool["name"]] == "personal" and tool["name"] in PERSONAL_TOOLS
        ) or (
            owners[tool["name"]] == "home_assistant" and tool["name"] in HA_DEVICE_TOOLS
        )]
        # 与 OpenAI 语音版本一致，只取 HA 的 Assist 内容，不重复导入 Claude 的 Alpha 规则。
        assist_prompt = (ha_catalog[1][0]["text"].partition("Home Assistant Assist prompt：\n")[2]
                         if ha_catalog is not None else "")
        worker = PersonalWorker(server)
        # worker.run 会一直等待队列请求，所以安排为后台任务，不能在这里直接 await 它结束。
        worker_task = asyncio.create_task(worker.run(), name="qwen-personal-worker")
        # 登记退出清理：即使发生异常，也会取消并等待 worker 退出，关闭它的 MCP 子进程。
        stack.push_async_callback(stop_worker, worker_task)
        gateway = ToolGateway(store, make_executor(ha_url=ha_url, http_client=ha_http_client, worker=worker))
        coordinator = SimpleCoordinator(
            gateway, timezone_name=os.environ.get("REMINDER_TIMEZONE", "America/Los_Angeles"),
            window_s=float(os.environ.get("COLLECTION_WINDOW_S", "2")),
            incomplete_s=float(os.environ.get("COLLECTION_INCOMPLETE_S", "5")),
        )
        commit_window = coordinator  # 只有简单写入窗口；不启动计划模式和复述。
        # 先关窗口，再停 worker/HTTP/store；WebSocket 重连不会触发这项程序级清理。
        stack.push_async_callback(commit_window.close)
        print("[个人工具] " + ("、".join(tool["name"] for tool in tools if tool["name"] in PERSONAL_TOOLS) or "暂不可用"))
        print("[HA 工具] " + ("、".join(tool["name"] for tool in tools if tool["name"] in HA_DEVICE_TOOLS) or "暂不可用"))
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(coordinator.run(), name="alpha-task-coordinator")
            # 打开并维持扬声器输出流；真正取缓冲区音频的是音频库调用的 player.callback。
            tasks.create_task(player.run())
            # 同时运行云端连接；它可以不断重连，而不用反复打开播放器和个人 server。
            tasks.create_task(reconnect_loop(
                url, key, player, costs, voice, region,
                float(os.environ.get("QWEN_SESSION_SECONDS", "1800")), tools,
                gateway=gateway, commit_window=commit_window, coordinator=coordinator,
                assist_prompt=assist_prompt,
            ))


if __name__ == "__main__":
    try:
        # 创建并运行事件循环；普通调用 main() 只会得到协程对象，不会执行其中的流程。
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已退出")
