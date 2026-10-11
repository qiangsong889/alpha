"""连接 Home Assistant MCP，列工具并读取 Assist prompt。

运行：.venv/bin/python ha_client.py
配置：同目录 .env 中的 HA_URL（HA 根地址）和 HA_TOKEN。
接口已按已安装的 MCP Python SDK 2.3.0 核对。
"""

import asyncio
import os
from pathlib import Path

import httpx2
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import GetPromptResult, TextContent


async def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    ha_url = os.environ.get("HA_URL", "").strip()
    token = os.environ.get("HA_TOKEN", "").strip()
    if not ha_url or not token:
        raise SystemExit("请在 alpha/.env 中设置 HA_URL 和 HA_TOKEN")

    url = ha_url.rstrip("/") + "/api/mcp"
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx2.Timeout(30.0, read=300.0),
    ) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("已连接 Home Assistant MCP")

                print("\n工具：")
                tools = await session.list_tools()
                for tool in tools.tools:
                    print(f"- {tool.name}")

                prompt = await session.get_prompt("Assist")
                if not isinstance(prompt, GetPromptResult):
                    raise RuntimeError("Assist prompt 未返回完整内容")
                text = "\n".join(
                    message.content.text
                    for message in prompt.messages
                    if isinstance(message.content, TextContent)
                )
                print("\nAssist prompt（前 200 字符）：")
                print(text[:200])


if __name__ == "__main__":
    asyncio.run(main())
