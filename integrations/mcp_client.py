"""MCP 工具发现、协议转换与 HA 按需调用，不涉及终端或音频。"""

import logging
import httpx2
from anthropic.types import TextBlockParam, ToolParam
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, GetPromptResult, PaginatedRequestParams, TextContent
from core.rules import ALPHA_RULES

logger = logging.getLogger("alpha.mcp")

ALLOWED_TOOLS = {
    "homeassistant__GetLiveContext",
    "intent__HassTurnOn",
    "intent__HassTurnOff",
    "light__HassLightSet",
    "list_items",
    "add_items",
    "create_reminder",
    "complete_items",
}

async def load_tools(session: ClientSession) -> list[ToolParam]:
    """读取全部工具，包括非白名单工具，供合并时检查重名。"""
    tools: list[ToolParam] = []
    cursor = None
    while True:
        result = await session.list_tools(
            params=PaginatedRequestParams(cursor=cursor) if cursor else None
        )
        for tool in result.tools:
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
    owners: dict[str, str] = {}
    for server_name, catalog in catalogs.items():
        for tool in catalog:
            name = tool["name"]
            if name in owners:
                raise RuntimeError(f"工具重名：{name}（{owners[name]} / {server_name}）")
            owners[name] = server_name
            if name in ALLOWED_TOOLS:
                tools.append(tool)
    return tools, {tool["name"]: owners[tool["name"]] for tool in tools}


async def call_ha_tool(
    name: str, arguments: dict, *, url: str, http_client: httpx2.AsyncClient,
) -> CallToolResult:
    """每次调用新建 transport 和会话，退出时关闭，复用 HTTP 客户端。"""
    async with streamable_http_client(url, http_client=http_client) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=arguments)
            if not isinstance(result, CallToolResult):
                raise RuntimeError("HA 工具未返回完整结果")
    return result


async def call_personal_tool(
    name: str, arguments: dict, *, server: StdioServerParameters,
) -> CallToolResult:
    """每次调用启动子进程、完成握手，退出时清理会话和进程。"""
    async with stdio_client(server) as (read, write):
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

