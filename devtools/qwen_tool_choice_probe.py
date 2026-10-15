"""最小在线协议探测，不启动麦克风/MCP，也不执行任何业务工具。

运行：.venv/bin/python -m devtools.qwen_tool_choice_probe
只输出事件类型、响应文本/工具名和脱敏错误，不打印密钥或连接地址。
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from websockets.asyncio.client import connect

from core.paths import ENV_FILE
from voice.qwen_realtime import session_config, websocket_url


async def probe(label, extra):
    key = os.environ["DASHSCOPE_API_KEY"]
    workspace = os.environ["QWEN_WORKSPACE_ID"]
    url = websocket_url(workspace, os.environ.get("QWEN_REGION", "singapore"))
    def safe(value):
        return str(value).replace(key, "[REDACTED]").replace(workspace, "[WORKSPACE]")
    try:
        async with asyncio.timeout(40):
            async with connect(url, additional_headers={"Authorization": f"Bearer {key}"},
                               open_timeout=20, max_size=16 * 1024 * 1024) as ws:
                async def send(kind, **payload):
                    await ws.send(json.dumps({"type": kind, **payload}, ensure_ascii=False))

                async def wait(kind):
                    while True:
                        event = json.loads(await ws.recv())
                        if event["type"] == "error":
                            print(label, "ERROR", safe(json.dumps(event.get("error", {}), ensure_ascii=False)))
                            return False
                        if event["type"] == kind:
                            return event

                await wait("session.created")
                config = session_config(os.environ.get("QWEN_VOICE", "Tina"))
                config["instructions"] = "这是无副作用协议测试。请只说你好，不要调用工具。"
                config["tools"] = [{"type": "function", "function": {
                    "name": "probe_ping", "description": "无副作用的测试工具，不操作任何真实服务。",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                }}]
                await send("session.update", session=config)
                updated = await wait("session.updated")
                if not updated:
                    return
                print(label, "VAD", json.dumps(updated["session"].get("turn_detection"), ensure_ascii=False))
                await send("conversation.item.create", item={"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "你好，请打个招呼。"}]})
                # 对照当前补救请求的 system 条目，探测 response.create 的参数兼容性。
                await send("response.create", **extra)
                while True:
                    event = json.loads(await ws.recv())
                    kind = event["type"]
                    if kind == "error":
                        print(label, "ERROR", safe(json.dumps(event.get("error", {}), ensure_ascii=False)))
                        return
                    if kind == "response.created":
                        print(label, "response.created", "tool_choice=", event.get("response", {}).get("tool_choice", "[not echoed]"))
                    if kind == "response.done":
                        response = event["response"]
                        print(label, "response.done", response.get("status"))
                        for item in response.get("output", []):
                            if item.get("type") == "function_call":
                                print(label, "function_call", item.get("name"))
                            elif item.get("type") == "message":
                                text = "".join(p.get("transcript", p.get("text", "")) for p in item.get("content", []))
                                print(label, "text", safe(text))
                        return
    except Exception as exc:
        print(label, type(exc).__name__, safe(exc))


async def main():
    load_dotenv(ENV_FILE)
    if not os.environ.get("DASHSCOPE_API_KEY") or not os.environ.get("QWEN_WORKSPACE_ID"):
        raise SystemExit("缺少 DASHSCOPE_API_KEY 或 QWEN_WORKSPACE_ID")
    await probe("nested-required", {"response": {"tool_choice": "required"}})
    await probe("top-level-required", {"tool_choice": "required"})


if __name__ == "__main__":
    asyncio.run(main())
