"""Alpha：通过 Claude、Home Assistant 和 Google Tasks MCP 管理家居与任务。

运行：alpha/.venv/bin/python alpha/alpha.py
同目录 .env：HA_URL、HA_TOKEN、ANTHROPIC_API_KEY。
可选：ANTHROPIC_MODEL（默认沿用知识助手的 claude-sonnet-5）。
"""

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

import httpx2
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlockParam, ToolParam, ToolResultBlockParam
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, GetPromptResult, PaginatedRequestParams, TextContent


ALLOWED_TOOLS = {
    "homeassistant__GetLiveContext",
    "intent__HassTurnOn",
    "intent__HassTurnOff",
    "light__HassLightSet",
    "llm__GetDateTime",   
    "list_items",
    "add_items",
    "create_reminder",
}

ALPHA_RULES = """你是 Alpha，用户的家居与个人任务助手。
始终用中文回答，回答简短，只告诉用户结果，不解释内部决策和工具调用过程。
需要家居状态或执行操作时使用提供的工具，不猜测状态，不虚构执行成功。 \n
目标不明确时：开关灯、调亮度这类可撤销的操作，选最合理的理解直接执行，执行后告诉用户做了什么；
以后的发邮件、删除文件这类不可撤销的操作，必须先确认。
创建提醒时，必须使用明确日期和时区偏移。对于“明天”等相对时间，
如果当前可用工具中有 llm__GetDateTime，先调用它获取当前日期和时区，
再计算目标日期的正确偏移并调用 create_reminder；不要直接沿用跨夏令时变更前的偏移。
如果时间工具不可用或返回的信息不足，询问用户明确日期和时区，不猜测。
"""


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
        print(f"个人服务暂时不可用：{format_error(exc)}")
        return None


def catalog_system(
    ha_catalog: tuple[list[ToolParam], list[TextBlockParam]] | None,
    personal_tools: list[ToolParam] | None,
) -> list[TextBlockParam]:
    text = ha_catalog[1][0]["text"] if ha_catalog is not None else (
        ALPHA_RULES + "\n家居设备暂时不可用。不要猜测当前时间或时区；信息不足时询问用户。"
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
        print(f"HA 暂时不可用：{format_error(exc)}")
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


async def agent_turn(
    client: AsyncAnthropic,
    tool_owners: dict[str, str],
    personal_server: StdioServerParameters,
    model: str,
    system: list[TextBlockParam],
    tools: list[ToolParam],
    messages: list[MessageParam],
    user_text: str,
    *,
    ha_url: str,
    http_client: httpx2.AsyncClient,
) -> str:
    # 成功完成这一轮后才保存，避免 API 失败留下不完整的工具消息。
    current: list[MessageParam] = [{"role": "user", "content": user_text}]
    available_names = {tool["name"] for tool in tools}
    for _ in range(10):
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages + current,
        )
        if response.stop_reason == "max_tokens":
            raise RuntimeError("回复被截断，请提高 max_tokens 后重试")
        if response.stop_reason not in {"end_turn", "tool_use"}:
            raise RuntimeError(f"未处理的停止原因：{response.stop_reason}")

        current.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            messages.extend(current)
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        calls = [block for block in response.content if block.type == "tool_use"]
        if not calls:
            raise RuntimeError("模型要求调用工具，但没有返回工具调用")

        results: list[ToolResultBlockParam] = []
        for block in calls:
            print(f"[工具] {block.name} 参数：{json.dumps(block.input, ensure_ascii=False)}")
            try:
                if block.name not in ALLOWED_TOOLS or block.name not in available_names:
                    raise ValueError(f"工具不在可用白名单中：{block.name}")
                if not isinstance(block.input, dict):
                    raise ValueError("工具参数必须是对象")
                owner = tool_owners.get(block.name)
                if owner == "home_assistant":
                    started = time.perf_counter()
                    result = await call_ha_tool(
                        block.name, block.input, url=ha_url, http_client=http_client
                    )
                    finished = time.perf_counter()
                    print(f"[耗时] {block.name}：{finished - started:.3f} 秒")
                elif owner == "personal":
                    started = time.perf_counter()
                    result = await call_personal_tool(
                        block.name, block.input, server=personal_server
                    )
                    finished = time.perf_counter()
                    print(f"[耗时] {block.name}：{finished - started:.3f} 秒")
                else:
                    raise ValueError(f"工具没有对应服务器：{block.name}")
                if not isinstance(result, CallToolResult):
                    raise RuntimeError("工具未返回完整结果")
                text = "\n".join(
                    content.text for content in result.content
                    if isinstance(content, TextContent)
                )
                is_error = result.is_error
            except Exception as exc:
                text = f"工具执行失败：{format_error(exc)}。操作可能已生效，写入操作不要盲目重试。"
                is_error = True

            print(f"[结果] isError={is_error} {text[:100]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": text,
                "is_error": is_error,
            })
        current.append({"role": "user", "content": results})

    raise RuntimeError("达到最大调用次数，仍未生成最终回答")


def format_error(exc: BaseException) -> str:
    """展开异步任务组异常，显示底层错误而不是整段 traceback。"""
    if isinstance(exc, BaseExceptionGroup):
        return "；".join(format_error(error) for error in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


async def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    ha_url = os.environ.get("HA_URL", "").strip()
    token = os.environ.get("HA_TOKEN", "").strip()
    if not ha_url or not token or not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise SystemExit("请在 alpha/.env 中设置 HA_URL、HA_TOKEN 和 ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

    url = ha_url.rstrip("/") + "/api/mcp"
    directory = Path(__file__).resolve().parent
    personal_server = StdioServerParameters(
        command=str(directory / ".venv/bin/python"),
        args=[str(directory / "personal_server.py")],
        cwd=str(directory),
    )
    async with AsyncExitStack() as stack:
        http_client = await stack.enter_async_context(httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx2.Timeout(30.0, read=300.0),
        ))
        personal_tools = await load_personal_catalog(personal_server)
        ha_catalog = await load_ha_catalog(url, http_client)
        tools, tool_owners = merge_tools({
            "personal": personal_tools if personal_tools is not None else [],
            "home_assistant": ha_catalog[0] if ha_catalog is not None else [],
        })
        system = catalog_system(ha_catalog, personal_tools)
        print("Alpha 已启动。输入 exit / quit / 退出 结束对话。")
        print("可用工具：" + ("、".join(tool["name"] for tool in tools) or "无"))
        messages: list[MessageParam] = []
        client = await stack.enter_async_context(AsyncAnthropic())
        while True:
            try:
                user_text = (await asyncio.to_thread(input, "你：")).strip()
            except EOFError:
                break
            if user_text.lower() in {"exit", "quit", "退出"}:
                break
            if not user_text:
                continue
            # 两边独立重试发现；成功后缓存，空工具列表也表示加载成功。
            if personal_tools is None or ha_catalog is None:
                if personal_tools is None:
                    personal_tools = await load_personal_catalog(personal_server)
                if ha_catalog is None:
                    ha_catalog = await load_ha_catalog(url, http_client)
                # 重名是配置错误，不能作为服务离线忽略。
                tools, tool_owners = merge_tools({
                    "personal": personal_tools if personal_tools is not None else [],
                    "home_assistant": ha_catalog[0] if ha_catalog is not None else [],
                })
                system = catalog_system(ha_catalog, personal_tools)
            try:
                answer = await agent_turn(
                    client, tool_owners, personal_server, model, system, tools,
                    messages, user_text, ha_url=url, http_client=http_client,
                )
                print(f"Alpha：{answer}")
            except Exception as exc:
                print(f"本轮失败：{format_error(exc)}")
                print("本轮未自动重试；已执行的操作可能仍然生效。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出")
