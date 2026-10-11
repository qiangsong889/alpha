"""Alpha：通过 Claude、Home Assistant 和 Google Tasks MCP 管理家居与任务。

运行：alpha/.venv/bin/python alpha/alpha.py
同目录 .env：HA_URL、HA_TOKEN、ANTHROPIC_API_KEY。
可选：ANTHROPIC_MODEL（默认沿用知识助手的 claude-sonnet-5）。
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

import anthropic
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
}

ALPHA_RULES = """你是 Alpha，用户的家居与个人任务助手。
始终用中文回答，回答简短，只告诉用户结果，不解释内部决策和工具调用过程。
需要家居状态或执行操作时使用提供的工具，不猜测状态，不虚构执行成功。 \n
目标不明确时：开关灯、调亮度这类可撤销的操作，选最合理的理解直接执行，执行后告诉用户做了什么；
以后的发邮件、删除文件这类不可撤销的操作，必须先确认。
"""


async def load_tools(
    sessions: dict[str, ClientSession],
) -> tuple[list[ToolParam], dict[str, ClientSession]]:
    """检查所有工具重名，合并白名单工具并记录所属会话。"""
    tools: list[ToolParam] = []
    tool_sessions: dict[str, ClientSession] = {}
    owners: dict[str, str] = {}
    for server_name, session in sessions.items():
        cursor = None
        while True:
            result = await session.list_tools(
                params=PaginatedRequestParams(cursor=cursor) if cursor else None
            )
            for tool in result.tools:
                if tool.name in owners:
                    raise RuntimeError(
                        f"工具重名：{tool.name}（{owners[tool.name]} / {server_name}）"
                    )
                owners[tool.name] = server_name
                if tool.name in ALLOWED_TOOLS:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.input_schema,
                    })
                    tool_sessions[tool.name] = session
            cursor = result.next_cursor
            if not cursor:
                break
    return tools, tool_sessions


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
    tool_sessions: dict[str, ClientSession],
    model: str,
    system: list[TextBlockParam],
    tools: list[ToolParam],
    messages: list[MessageParam],
    user_text: str,
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
                session = tool_sessions.get(block.name)
                if session is None:
                    raise ValueError(f"工具没有对应会话：{block.name}")
                result = await session.call_tool(block.name, arguments=block.input)
                if not isinstance(result, CallToolResult):
                    raise RuntimeError("工具未返回完整结果")
                text = "\n".join(
                    content.text for content in result.content
                    if isinstance(content, TextContent)
                )
                is_error = result.is_error
            except Exception as exc:
                text = f"工具执行失败：{exc}"
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
        read_ha, write_ha = await stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        session_ha = await stack.enter_async_context(ClientSession(read_ha, write_ha))
        await session_ha.initialize()

        read_personal, write_personal = await stack.enter_async_context(
            stdio_client(personal_server)
        )
        session_personal = await stack.enter_async_context(
            ClientSession(read_personal, write_personal)
        )
        await session_personal.initialize()

        tools, tool_sessions = await load_tools({
            "home_assistant": session_ha,
            "personal": session_personal,
        })
        system = await load_system(session_ha)
        print("Alpha 已连接 Home Assistant 和个人服务。输入 exit / quit / 退出 结束对话。")
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
            try:
                answer = await agent_turn(
                    client, tool_sessions, model, system, tools, messages, user_text
                )
                print(f"Alpha：{answer}")
            except (anthropic.APIError, RuntimeError) as exc:
                print(f"请求失败：{exc}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出")
