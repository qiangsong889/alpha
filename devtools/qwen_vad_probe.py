"""60 秒麦克风/VAD 实测：不播放回答，不加载 MCP，不操作业务数据。"""

import asyncio
import json
import os
import time

from dotenv import load_dotenv
from websockets.asyncio.client import connect

from core.paths import ENV_FILE
from voice.audio import SpeechActivity
from voice.qwen_realtime import (
    send_event, send_microphone, session_config, verify_session, wait_for_event, websocket_url,
)


async def main():
    load_dotenv(ENV_FILE)
    config = session_config(os.environ.get("QWEN_VOICE", "Tina"))
    config["tools"] = []
    key = os.environ["DASHSCOPE_API_KEY"]
    workspace = os.environ["QWEN_WORKSPACE_ID"]
    started = time.perf_counter()
    async with connect(websocket_url(workspace, os.environ.get("QWEN_REGION", "singapore")),
                       additional_headers={"Authorization": f"Bearer {key}"}, open_timeout=20) as ws:
        await wait_for_event(ws, "session.created")
        await send_event(ws, "session.update", session=config)
        updated = await wait_for_event(ws, "session.updated")
        verify_session(updated["session"])
        print("[VAD 实测] " + json.dumps(updated["session"]["turn_detection"], ensure_ascii=False), flush=True)
        print("请先单独说‘对’，等两秒，再说‘把牛奶……还有鸡蛋，加到购物清单’，中间停一秒。", flush=True)

        async def receive():
            async for message in ws:
                event = json.loads(message)
                kind = event["type"]
                stamp = f"[{time.perf_counter() - started:.3f}]"
                if kind in {"input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped", "input_audio_buffer.committed"}:
                    print(stamp, kind, event.get("item_id"), flush=True)
                elif kind == "conversation.item.input_audio_transcription.completed":
                    print(stamp, "你：", repr(event.get("transcript", "")), flush=True)
                elif kind == "error":
                    text = json.dumps(event.get("error", {}), ensure_ascii=False)
                    print(stamp, "ERROR", text.replace(key, "[REDACTED]").replace(workspace, "[WORKSPACE]"), flush=True)

        try:
            async with asyncio.timeout(60):
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(send_microphone(ws, SpeechActivity(16000)))
                    tasks.create_task(receive())
        except TimeoutError:
            print("[VAD 实测] 60 秒结束，麦克风已关闭", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
