"""Alpha：通过 Claude、Home Assistant 和 Google Tasks MCP 管理家居与任务。

运行：alpha/.venv/bin/python alpha/alpha.py
同目录 .env：HA_URL、HA_TOKEN、ANTHROPIC_API_KEY。
可选：ANTHROPIC_MODEL（默认沿用知识助手的 claude-sonnet-5）。
"""

import asyncio
import argparse
import json
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from personal_worker import PersonalWorker, stop_worker

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
    "list_items",
    "add_items",
    "create_reminder",
    "complete_items",
}

ALPHA_RULES = """你是 Alpha，用户的家居与个人任务助手。
始终用中文回答，回答简短，只告诉用户结果，不解释内部决策和工具调用过程。
回答会被朗读出来，所以不要用 Markdown、列表或表情符号，用一两句口语化的话说完。
需要家居状态或执行操作时使用提供的工具，不猜测状态，不虚构执行成功。 \n
目标不明确时：开关灯、调亮度这类可撤销的操作，选最合理的理解直接执行，执行后告诉用户做了什么；
以后的发邮件、删除文件这类不可撤销的操作，必须先确认。
创建提醒时，传入不带时区的 local_time；时区由 personal 服务配置并计算偏移。
每条用户消息前缀包含程序提供的当前本地时间和 IANA 时区。
对于“明天”等相对时间，以本条消息前缀为基准计算日期，不使用历史消息中的时间。
提醒采用前缀中的配置时区；不需要查询 HA 时间或自行计算 UTC 偏移。
用户指定其他时区或时间含义不明确时，先询问确认。
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
    worker: PersonalWorker,
    model: str,
    system: list[TextBlockParam],
    tools: list[ToolParam],
    messages: list[MessageParam],
    user_text: str,
    *,
    ha_url: str,
    http_client: httpx2.AsyncClient,
    timings: list[tuple[str, float]] | None = None,
) -> str:
    # 成功完成这一轮后才保存，避免 API 失败留下不完整的工具消息。
    current: list[MessageParam] = [{"role": "user", "content": user_text}]
    available_names = {tool["name"] for tool in tools}
    for request_index in range(1, 11):
        started = time.perf_counter()
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages + current,
        )
        if timings is not None:
            timings.append((f"Claude 第{request_index}次", time.perf_counter() - started))
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
            tool_started = time.perf_counter()
            try:
                if block.name not in ALLOWED_TOOLS or block.name not in available_names:
                    raise ValueError(f"工具不在可用白名单中：{block.name}")
                if not isinstance(block.input, dict):
                    raise ValueError("工具参数必须是对象")
                owner = tool_owners.get(block.name)
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
            except Exception as exc:
                text = f"工具执行失败：{format_error(exc)}。操作可能已生效，写入操作不要盲目重试。"
                is_error = True
            finally:
                if timings is not None:
                    timings.append(("工具", time.perf_counter() - tool_started))

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


async def speak(text: str) -> None:
    """通过 macOS say 的标准输入朗读，避免把回答当命令参数。"""
    process = await asyncio.create_subprocess_exec(
        "say", stdin=asyncio.subprocess.PIPE,
    )
    try:
        await process.communicate(text.encode("utf-8"))
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"朗读程序退出码：{process.returncode}")


async def main(voice: bool = False) -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    ha_url = os.environ.get("HA_URL", "").strip()
    token = os.environ.get("HA_TOKEN", "").strip()
    if not ha_url or not token or not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise SystemExit("请在 alpha/.env 中设置 HA_URL、HA_TOKEN 和 ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    reminder_tz = ZoneInfo(os.environ.get("REMINDER_TIMEZONE", "America/Los_Angeles"))

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
        worker = PersonalWorker(personal_server)
        worker_task = asyncio.create_task(worker.run(), name="personal-worker")
        stack.push_async_callback(stop_worker, worker_task)
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
            timings: list[tuple[str, float]] = []
            try:
                if voice:
                    from listen import record, transcribe

                    audio = await asyncio.to_thread(record)
                    turn_started = time.perf_counter()
                    text = await asyncio.to_thread(transcribe, audio)
                    timings.append(("转文字", time.perf_counter() - turn_started))
                    user_text = text.strip()
                    print(f"识别文字：{user_text or '（未识别到文字）'}")
                    print(f"录音时长：{audio.size / 16000:.2f} 秒")
                else:
                    user_text = (await asyncio.to_thread(input, "你：")).strip()
                    turn_started = time.perf_counter()
            except EOFError:
                break
            except Exception as exc:
                print(f"输入失败：{format_error(exc)}")
                continue
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
                current_time = datetime.now(reminder_tz).isoformat(timespec="seconds")
                timed_user_text = (
                    f"[当前本地时间：{current_time}；时区：{reminder_tz.key}]\n{user_text}"
                )
                answer = await agent_turn(
                    client, tool_owners, worker, model, system, tools,
                    messages, timed_user_text, ha_url=url, http_client=http_client,
                    timings=timings,
                )
                print(f"Alpha：{answer}")
            except Exception as exc:
                print(f"本轮失败：{format_error(exc)}")
                print("本轮未自动重试；已执行的操作可能仍然生效。")
                continue
            total_label = "朗读开始前总计" if voice else "总计"
            timings.append((total_label, time.perf_counter() - turn_started))
            print("[耗时] " + " ｜ ".join(
                f"{label} {seconds:.2f}s" for label, seconds in timings
            ))
            if voice and answer:
                try:
                    await speak(answer)
                except Exception as exc:
                    print(f"朗读失败：{format_error(exc)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha 家居与个人任务助手")
    parser.add_argument("--voice", action="store_true", help="录音输入并朗读回答（macOS）")
    options = parser.parse_args()
    try:
        asyncio.run(main(voice=options.voice))
    except KeyboardInterrupt:
        print("\n已退出")
