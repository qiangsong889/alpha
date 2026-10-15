"""MCP 工具发现、协议转换与 HA 按需调用，不涉及终端或音频。"""

import logging
import json
import httpx2
from anthropic.types import TextBlockParam, ToolParam
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, GetPromptResult, PaginatedRequestParams, TextContent
from core.rules import ALPHA_RULES
from integrations.tool_gateway import ToolNotSent
from integrations.tool_policy import TOOL_POLICY

logger = logging.getLogger("alpha.mcp")

# 保留旧导出供 Brain 使用，工具名由同一份策略表定义，不再维护两张白名单。
ALLOWED_TOOLS = set(TOOL_POLICY)


def device_action_succeeded(name: str, result: CallToolResult) -> bool:
    """只认可 HA 明确返回 action_done 且没有失败明细的开关、调光操作。"""
    if name not in {"intent__HassTurnOn", "intent__HassTurnOff", "light__HassLightSet"} or result.is_error:
        return False
    payloads = []
    if isinstance(result.structured_content, dict):
        payloads.append(result.structured_content)
    for content in result.content:
        if isinstance(content, TextContent):
            try:
                payloads.append(json.loads(content.text))
            except (ValueError, TypeError):
                return False
    if not payloads:
        return False
    for payload in payloads:
        if not isinstance(payload, dict):
            return False
        response = payload.get("response", payload)
        if not isinstance(response, dict) or response.get("response_type") != "action_done":
            return False
        data = response.get("data", {})
        if isinstance(data, dict) and (data.get("failed") or data.get("errors")):
            return False
    return True


async def load_tools(session: ClientSession) -> list[ToolParam]:
    """读取全部工具，包括非白名单工具，供合并时检查重名。"""
    tools: list[ToolParam] = []
    cursor = None
    while True:
        # 工具目录也可能分页；next_cursor 为空才代表已经读完全部工具。
        result = await session.list_tools(
            params=PaginatedRequestParams(cursor=cursor) if cursor else None
        )
        for tool in result.tools:
            # MCP schema 转成 Claude 所需的字段名；这里只转换描述，不执行工具。
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
            })
        cursor = result.next_cursor
        if not cursor:
            return tools


def merge_tools(
    catalogs: dict[str, list[ToolParam]],
) -> tuple[list[ToolParam], dict[str, str]]:
    tools: list[ToolParam] = []
    # owners 是“工具名 → 所属服务器”的路由表，执行时据此选择正确的 MCP 连接。
    owners: dict[str, str] = {}
    for server_name, catalog in catalogs.items():
        for tool in catalog:
            name = tool["name"]
            if name in owners:
                # 白名单过滤之前就检查重名，防止两个 server 的同名工具被悄悄覆盖。
                raise RuntimeError(f"工具重名：{name}（{owners[name]} / {server_name}）")
            owners[name] = server_name
            if name in ALLOWED_TOOLS:
                tools.append(tool)
    return tools, {tool["name"]: owners[tool["name"]] for tool in tools}


async def call_ha_tool(
    name: str, arguments: dict, *, url: str, http_client: httpx2.AsyncClient,
    classify_not_sent: bool = False,
) -> CallToolResult:
    """每次调用新建 transport 和会话，退出时关闭，复用 HTTP 客户端。"""
    tool_started = False
    try:
        async with streamable_http_client(url, http_client=http_client) as (read, write):
            # HTTP 客户端可复用；每次新建会话，隔离 HA 重启影响。
            async with ClientSession(read, write) as session:
                await session.initialize()
                # 从进入 call_tool 开始，就不能再断言请求没有发出。
                tool_started = True
                result = await session.call_tool(name, arguments=arguments)
                if not isinstance(result, CallToolResult):
                    raise RuntimeError("HA 工具未返回完整结果")
    except Exception as exc:
        if classify_not_sent and not tool_started:
            raise ToolNotSent(f"HA 工具请求尚未发出：{format_error(exc)}") from exc
        raise
    return result


async def call_personal_tool(
    name: str, arguments: dict, *, server: StdioServerParameters,
) -> CallToolResult:
    """每次调用启动子进程、完成握手，退出时清理会话和进程。"""
    async with stdio_client(server) as (read, write):
        # 这是每次启动子进程的版本；需要长期复用连接时，主流程改用 PersonalWorker。
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=arguments)
            if not isinstance(result, CallToolResult):
                raise RuntimeError("personal 工具未返回完整结果")
    return result


async def load_personal_catalog(
    server: StdioServerParameters,
) -> list[ToolParam] | None:
    try:
        # 只在此连接里读取目录；离开 async with 会关闭会话和这次启动的子进程。
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_tools(session)
        return tools
    except Exception as exc:
        logger.warning("个人服务暂时不可用：%s", format_error(exc))
        return None


def catalog_system(
    ha_catalog: tuple[list[ToolParam], list[TextBlockParam]] | None,
    personal_tools: list[ToolParam] | None,
) -> list[TextBlockParam]:
    # HA 目录中的提示已包含 Alpha 规则和 Assist 内容；发现失败时退回基础规则。
    text = ha_catalog[1][0]["text"] if ha_catalog is not None else (
        ALPHA_RULES + "\n家居设备暂时不可用。"
    )
    if personal_tools is None:
        text += "\n个人任务和日历服务暂时不可用。不能声称已完成相关操作。"
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


async def load_ha_catalog(
    url: str, http_client: httpx2.AsyncClient,
) -> tuple[list[ToolParam], list[TextBlockParam]] | None:
    try:
        async with streamable_http_client(url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_tools(session)
                system = await load_system(session)
        return tools, system
    except Exception as exc:
        logger.warning("HA 暂时不可用：%s", format_error(exc))
        return None


async def load_system(session: ClientSession) -> list[TextBlockParam]:
    # Assist prompt 提供 HA 暴露的设备背景；工具目录描述“可调用什么”，prompt 补充上下文。
    prompt = await session.get_prompt("Assist")
    if not isinstance(prompt, GetPromptResult):
        raise RuntimeError("Assist prompt 未返回完整内容")
    assist_text = "\n".join(
        message.content.text
        for message in prompt.messages
        if isinstance(message.content, TextContent)
    )
    return [{
        "type": "text",
        "text": ALPHA_RULES + "\nHome Assistant Assist prompt：\n" + assist_text,
        "cache_control": {"type": "ephemeral"},
    }]


def format_error(exc: BaseException) -> str:
    """展开异步任务组异常，显示底层错误而不是整段 traceback。"""
    if isinstance(exc, BaseExceptionGroup):
        return "；".join(format_error(error) for error in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"

