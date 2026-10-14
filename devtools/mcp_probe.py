"""手动探测 Home Assistant 的 MCP Server（原始 JSON-RPC，不依赖 MCP SDK）。

用法：
    python mcp_probe.py init                                   # 握手，看 serverInfo / capabilities
    python mcp_probe.py list                                   # 列出工具名和描述
    python mcp_probe.py schema intent__HassTurnOn              # 看某个工具的 inputSchema
    python mcp_probe.py call homeassistant__GetLiveContext     # 调用工具（无参数）
    python mcp_probe.py call intent__HassTurnOff '{"name": "客厅灯"}'
    python mcp_probe.py prompts                                # 列出服务端提供的 prompt
    python mcp_probe.py prompt <名字>                           # 取出某个 prompt 的内容

环境变量（可写在同目录 .env 里）：
    HA_URL    例如 http://192.168.1.100:8123
    HA_TOKEN  长期访问令牌（eyJ 开头）
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_env(path: Path) -> None:
    """极简 .env 读取：KEY=VALUE，已存在的环境变量优先。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_body(body: str, content_type: str) -> dict[str, Any]:
    """服务端可以回 JSON，也可以回 SSE（text/event-stream），两种都处理。"""
    if "text/event-stream" in content_type:
        data_lines = [line[5:].strip() for line in body.splitlines() if line.startswith("data:")]
        if not data_lines:
            raise ValueError(f"SSE 响应里没有 data 行：{body[:200]}")
        return json.loads(data_lines[-1])
    return json.loads(body)


def rpc(method: str, params: dict[str, Any] | None = None, request_id: int = 1) -> dict[str, Any]:
    # 这个调试脚本直接组装 JSON-RPC 请求，展示 SDK 平时替你处理的协议字段。
    # id 用于把响应配回请求；method 是协议方法，params 是它需要的参数。
    url = os.environ["HA_URL"].rstrip("/") + "/api/mcp"
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    # JSON-RPC 是消息格式，HTTP 是这里选用的传输方式，两者不是同一层概念。
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {os.environ['HA_TOKEN']}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    try:
        # urllib 是同步网络调用：当前线程会等到响应。这个独立探测脚本不运行异步循环。
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            message = parse_body(body, response.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} {exc.reason}：{exc.read().decode('utf-8', 'replace')[:300]}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"连不上 {url}：{exc.reason}")

    if "error" in message:
        # JSON-RPC 的 error 表示协议请求失败；工具结果里的 isError 则是工具执行失败。
        raise SystemExit(f"JSON-RPC 错误：{json.dumps(message['error'], ensure_ascii=False)}")
    result: dict[str, Any] = message["result"]
    return result


def main(argv: list[str]) -> None:
    load_env(Path(__file__).resolve().parent.parent / ".env")
    for key in ("HA_URL", "HA_TOKEN"):
        if not os.environ.get(key):
            raise SystemExit(f"缺少环境变量 {key}（可以写进 alpha/.env）")

    if len(argv) < 2 or argv[1] not in {"init", "list", "schema", "call", "prompts", "prompt"}:
        raise SystemExit(__doc__)
    command = argv[1]

    if command == "init":
        result = rpc("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "mcp_probe", "version": "0.1"},
        })
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif command == "list":
        for tool in rpc("tools/list", request_id=2)["tools"]:
            print(f"{tool['name']} - {tool.get('description', '')[:80]}")

    elif command == "schema":
        if len(argv) < 3:
            raise SystemExit("用法：python mcp_probe.py schema <工具名>")
        tools = {tool["name"]: tool for tool in rpc("tools/list", request_id=2)["tools"]}
        tool = tools.get(argv[2])
        if tool is None:
            raise SystemExit(f"没有这个工具：{argv[2]}")
        print(json.dumps(tool.get("inputSchema", {}), ensure_ascii=False, indent=2))

    elif command == "call":
        if len(argv) < 3:
            raise SystemExit("用法：python mcp_probe.py call <工具名> ['{\"参数\": \"值\"}']")
        arguments = json.loads(argv[3]) if len(argv) > 3 else {}
        result = rpc("tools/call", {"name": argv[2], "arguments": arguments}, request_id=3)
        if result.get("isError"):
            print("⚠️ 工具返回了错误（isError=true）：")
        for block in result.get("content", []):
            print(block.get("text", block))

    elif command == "prompts":
        for prompt in rpc("prompts/list", request_id=4).get("prompts", []):
            arguments = ", ".join(arg["name"] for arg in prompt.get("arguments", []))
            print(f"{prompt['name']} - {prompt.get('description', '')[:80]}  参数: [{arguments}]")

    elif command == "prompt":
        if len(argv) < 3:
            raise SystemExit("用法：python mcp_probe.py prompt <名字> ['{\"参数\": \"值\"}']")
        params: dict[str, Any] = {"name": argv[2]}
        if len(argv) > 3:
            params["arguments"] = json.loads(argv[3])
        result = rpc("prompts/get", params, request_id=5)
        # prompts/get 返回的是一组消息（role + content），通常直接放进对话或 system prompt。
        for message in result.get("messages", []):
            content = message.get("content", {})
            print(f"--- role: {message.get('role')} ---")
            print(content.get("text", content))


if __name__ == "__main__":
    main(sys.argv)
