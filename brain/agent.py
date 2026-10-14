"""AlphaBrain：管理 Claude、MCP 目录、客户端生命周期与多轮对话。"""

import asyncio
import json
import logging
import os
import time
from contextlib import AsyncExitStack
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx2
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlockParam, ToolParam, ToolResultBlockParam
from dotenv import load_dotenv

from core.paths import ENV_FILE, PROJECT_ROOT, VENV_PYTHON
from mcp import StdioServerParameters
from mcp.types import CallToolResult, TextContent
from integrations.personal_worker import PersonalWorker, stop_worker
from integrations.mcp_client import (
    ALLOWED_TOOLS, call_ha_tool, catalog_system, format_error,
    load_ha_catalog, load_personal_catalog, merge_tools,
)

logger = logging.getLogger("alpha.brain")

def device_action_reply(name: str, arguments: dict, result: CallToolResult) -> str | None:
    """只对明确成功的 HA 设备操作生成简短确认，其他结果交给模型。"""
    # 只有明确成功的设备操作才由代码直接说确认语，省掉再请求一次 Claude 的时间。
    # 查询、个人清单、失败或无法判断的结果返回 None，继续交给模型组织回答。
    actions = {
        "intent__HassTurnOn": "打开",
        "intent__HassTurnOff": "关闭",
        "light__HassLightSet": "调整",
    }
    if name not in actions or result.is_error:
        return None
    payloads = []
    if isinstance(result.structured_content, dict):
        payloads.append(result.structured_content)
    for content in result.content:
        if isinstance(content, TextContent):
            try:
                payloads.append(json.loads(content.text))
            except (ValueError, TypeError):
                return None
    if not payloads:
        return None
    for payload in payloads:
        if not isinstance(payload, dict):
            return None
        response = payload.get("response", payload)
        if not isinstance(response, dict) or response.get("response_type") != "action_done":
            return None
        data = response.get("data", {})
        if isinstance(data, dict) and (data.get("failed") or data.get("errors")):
            return None
    target = arguments.get("name")
    if not isinstance(target, str) or not target.strip():
        target = arguments.get("area") or "设备"
    if not isinstance(target, str):
        target = "设备"
    if name == "light__HassLightSet":
        brightness = arguments.get("brightness")
        # 同时调整颜色等参数时使用通用确认，避免遗漏其他操作。
        if type(brightness) in (int, float) and set(arguments) <= {
            "name", "area", "floor", "domain", "brightness"
        }:
            return f"已把{target}调到 {brightness:g}%"
        return f"已调整{target}的灯光"
    return f"已{actions[name]}{target}"


class AlphaBrain:
    # 大脑不读取麦克风、不直接朗读；CLI 或 HTTP 语音层都只需要调用 start/ask/close。
    def __init__(self, *, model: str | None = None):
        self._model_override = model
        self.model = ""
        self.ha_url = ""
        self.reminder_tz: ZoneInfo | None = None
        self.personal_server: StdioServerParameters | None = None
        self.client: AsyncAnthropic | None = None
        self.http_client: httpx2.AsyncClient | None = None
        self.worker: PersonalWorker | None = None
        self.worker_task: asyncio.Task[None] | None = None
        # 放在实例上的对话历史和目录会跨 ask 保留，而不是每句话都重新建立一份。
        self.messages: list[MessageParam] = []
        self.personal_tools: list[ToolParam] | None = None
        self.ha_catalog: tuple[list[ToolParam], list[TextBlockParam]] | None = None
        self.tools: list[ToolParam] = []
        self.tool_owners: dict[str, str] = {}
        self.system: list[TextBlockParam] = []
        self.last_timings: list[tuple[str, float]] = []
        self._stack: AsyncExitStack | None = None
        self._started = False
        # 同一大脑一次处理一个 ask，避免两个请求交错修改同一份历史。
        # 这是异步锁：等锁的任务暂停，事件循环中的其他任务仍可运行。
        self._lock = asyncio.Lock()

    @property
    def ha_available(self) -> bool:
        return self.ha_catalog is not None

    def _prepare_catalog(self) -> None:
        # 合并描述给模型，同时保留路由表供客户端执行工具；模型不负责选择具体连接对象。
        self.tools, self.tool_owners = merge_tools({
            "personal": self.personal_tools if self.personal_tools is not None else [],
            "home_assistant": self.ha_catalog[0] if self.ha_catalog is not None else [],
        })
        self.system = catalog_system(self.ha_catalog, self.personal_tools)

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            load_dotenv(ENV_FILE)
            ha_root = os.environ.get("HA_URL", "").strip()
            token = os.environ.get("HA_TOKEN", "").strip()
            if not ha_root or not token or not os.environ.get("ANTHROPIC_API_KEY", "").strip():
                raise SystemExit("请在 alpha/.env 中设置 HA_URL、HA_TOKEN 和 ANTHROPIC_API_KEY")
            self.model = self._model_override or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
            self.reminder_tz = ZoneInfo(os.environ.get("REMINDER_TIMEZONE", "America/Los_Angeles"))
            self.ha_url = ha_root.rstrip("/") + "/api/mcp"
            # 以模块方式启动（-m），工作目录设为项目根目录，server 才能 import core / servers 包。
            self.personal_server = StdioServerParameters(
                command=str(VENV_PYTHON),
                args=["-m", "servers.personal_server"], cwd=str(PROJECT_ROOT),
            )
            # 统一管理不同客户端和 worker 的退出清理，按登记的逆序释放资源。
            stack = AsyncExitStack()
            await stack.__aenter__()
            self._stack = stack
            try:
                self.http_client = await stack.enter_async_context(httpx2.AsyncClient(
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx2.Timeout(30.0, read=300.0),
                ))
                self.personal_tools = await load_personal_catalog(self.personal_server)
                self.worker = PersonalWorker(self.personal_server)
                # run 长期等队列，必须安排为后台任务；直接 await 会停在这里，无法完成启动。
                self.worker_task = asyncio.create_task(self.worker.run(), name="personal-worker")
                stack.push_async_callback(stop_worker, self.worker_task)
                self.ha_catalog = await load_ha_catalog(self.ha_url, self.http_client)
                self._prepare_catalog()
                logger.info("可用工具：%s", "、".join(tool["name"] for tool in self.tools) or "无")
                self.client = await stack.enter_async_context(AsyncAnthropic())
                self._started = True
            except BaseException:
                try:
                    await stack.aclose()
                finally:
                    self._stack = None
                    self.client = self.http_client = self.worker = None
                    self.worker_task = None
                raise

    async def ask(self, request: str) -> str:
        # 一次用户提问的入口：确认启动状态、重试失败的目录发现、注入时间、进入工具循环。
        async with self._lock:
            if (not self._started or self.http_client is None or self.personal_server is None
                    or self.reminder_tz is None):
                raise RuntimeError("请先调用 AlphaBrain.start()")
            started = time.perf_counter()
            self.last_timings = []
            # 发现失败的服务在下一次 ask 时重试；成功的目录继续缓存。
            if self.personal_tools is None or self.ha_catalog is None:
                if self.personal_tools is None:
                    self.personal_tools = await load_personal_catalog(self.personal_server)
                if self.ha_catalog is None:
                    self.ha_catalog = await load_ha_catalog(self.ha_url, self.http_client)
                self._prepare_catalog()
            current_time = datetime.now(self.reminder_tz).isoformat(timespec="seconds")
            # 时间放进本轮用户消息；system 的静态规则保持稳定，便于利用提示缓存。
            timed_request = (
                f"[当前本地时间：{current_time}；时区：{self.reminder_tz.key}]\n{request}"
            )
            answer = await self._agent_turn(timed_request, self.last_timings)
            logger.debug("[耗时] %s ｜ brain总计 %.2fs", " ｜ ".join(
                f"{label} {elapsed:.2f}s" for label, elapsed in self.last_timings
            ), time.perf_counter() - started)
            return answer

    async def mark_interrupted(self, spoken: str) -> bool:
        """语音层报告最后一条回答被打断：把它截断成用户实际听到的部分。"""
        async with self._lock:
            if not self.messages or self.messages[-1]["role"] != "assistant":
                return False
            heard = spoken.strip()
            # 修改的是对话记忆，不是撤销已执行的工具；让后续推理知道用户只听到了前半句。
            self.messages[-1] = {
                "role": "assistant",
                "content": (f"{heard}……" if heard else "")
                + "（回答到这里被用户打断，后面的内容用户没有听到）",
            }
            return True

    async def close(self) -> None:
        # 即使使用者异常退出，也应调用 close；清理 callback 会停止后台 worker。
        async with self._lock:
            self._started = False
            stack, self._stack = self._stack, None
            try:
                if stack is not None:
                    await stack.aclose()
            finally:
                self.client = self.http_client = self.worker = None
                self.worker_task = None

    async def _agent_turn(self, user_text: str, timings: list[tuple[str, float]]) -> str:
        # 一轮用户输入可能需要多次 Claude 请求：先决策工具 → 执行工具 → 再生成回答。
        client = self.client
        worker = self.worker
        http_client = self.http_client
        if client is None or worker is None or http_client is None:
            raise RuntimeError("请先调用 AlphaBrain.start()")
        tool_owners, model, system = self.tool_owners, self.model, self.system
        tools, messages, ha_url = self.tools, self.messages, self.ha_url
        # 成功完成这一轮后才保存，避免 API 失败留下不完整的工具消息。
        current: list[MessageParam] = [{"role": "user", "content": user_text}]
        available_names = {tool["name"] for tool in tools}
        for request_index in range(1, 11):
            # 设置轮数上限，避免模型一直调用工具而不结束；每次请求保留历史和本轮中间消息。
            started = time.perf_counter()
            response = await client.messages.create(
                # await 等这次 Claude 返回，但不阻止 MCP worker 等其他任务继续执行。
                model=model,
                max_tokens=4096,
                system=system,
                tools=tools,
                messages=messages + current,
                cache_control={"type": "ephemeral"}
            )
            if timings is not None:
                timings.append((f"Claude 第{request_index}次", time.perf_counter() - started))
            logger.info(
                "[缓存] Claude 第%d次：cache_read=%d tokens ｜ cache_write=%d tokens",
                request_index, response.usage.cache_read_input_tokens or 0,
                response.usage.cache_creation_input_tokens or 0,
            )
            if response.stop_reason == "max_tokens":
                raise RuntimeError("回复被截断，请提高 max_tokens 后重试")
            if response.stop_reason not in {"end_turn", "tool_use"}:
                raise RuntimeError(f"未处理的停止原因：{response.stop_reason}")

            current.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "end_turn":
                # end_turn 才是模型结束本轮；tool_use 则需要客户端执行动作后再提供结果。
                messages.extend(current)
                return "".join(
                    block.text for block in response.content if block.type == "text"
                )

            calls = [block for block in response.content if block.type == "tool_use"]
            if not calls:
                raise RuntimeError("模型要求调用工具，但没有返回工具调用")

            results: list[ToolResultBlockParam] = []
            action_replies: list[str | None] = []
            for block in calls:
                logger.info("[工具] %s 参数：%s", block.name, json.dumps(block.input, ensure_ascii=False))
                tool_started = time.perf_counter()
                action_reply = None
                try:
                    if block.name not in ALLOWED_TOOLS or block.name not in available_names:
                        raise ValueError(f"工具不在可用白名单中：{block.name}")
                    if not isinstance(block.input, dict):
                        raise ValueError("工具参数必须是对象")
                    owner = tool_owners.get(block.name)
                    # 根据启动时建立的路由表分发，不把工具名当 Python 代码直接执行。
                    if owner == "home_assistant":
                        result = await call_ha_tool(
                            block.name, block.input, url=ha_url, http_client=http_client
                        )
                    elif owner == "personal":
                        result = await worker.call(block.name, block.input)
                    else:
                        raise ValueError(f"工具没有对应服务器：{block.name}")
                    if not isinstance(result, CallToolResult):
                        raise RuntimeError("工具未返回完整结果")
                    text = "\n".join(
                        content.text for content in result.content
                        if isinstance(content, TextContent)
                    )
                    is_error = result.is_error
                    if owner == "home_assistant":
                        action_reply = device_action_reply(block.name, block.input, result)
                except Exception as exc:
                    text = f"工具执行失败：{format_error(exc)}。操作可能已生效，写入操作不要盲目重试。"
                    is_error = True
                finally:
                    if timings is not None:
                        timings.append(("工具", time.perf_counter() - tool_started))

                logger.info("[结果] isError=%s %s", is_error, text[:100])
                action_replies.append(action_reply)
                results.append({
                    # tool_use_id 必须对应模型发来的 block.id，模型才能把结果配回原调用。
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": text,
                    "is_error": is_error,
                })
            current.append({"role": "user", "content": results})
            # Claude 协议中工具结果放在 user 消息里；它是程序回传，不是用户又输入了一句。
            if all(reply is not None for reply in action_replies):
                answer = "，".join(reply for reply in action_replies if reply is not None) + "。"
                current.append({"role": "assistant", "content": answer})
                messages.extend(current)
                return answer

        raise RuntimeError("达到最大调用次数，仍未生成最终回答")
