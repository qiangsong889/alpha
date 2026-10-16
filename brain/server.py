"""把 AlphaBrain 包成一个本机 HTTP 服务，供语音层（LiveKit 等）调用。

为什么单独开一个进程：LiveKit 依赖旧版 anthropic / openai / mcp，和 Alpha 用的新版冲突，
两边不能装在同一个虚拟环境里。用 HTTP 隔开，两边各用各的依赖。

运行（alpha 的虚拟环境）：
    .venv/bin/python -m brain.server
接口：
    POST /ask          {"text": "..."}    → {"answer": "...", "timings": [...]}
    POST /ask/stream   {"text": "..."}    → NDJSON delta / done / error
    POST /interrupted  {"spoken": "..."}  → {"ok": true}
    GET  /health
"""

import logging
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from brain.agent import AlphaBrain

logger = logging.getLogger("alpha.brain_server")
brain = AlphaBrain(style="voice")
stream_tasks: set[asyncio.Task[None]] = set()


class AskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    timings: list[tuple[str, float]]


class InterruptedRequest(BaseModel):
    spoken: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # async context manager：yield 前是服务启动；yield 期间 FastAPI 接请求；finally 是关闭。
    # 大脑跨 HTTP 请求复用，因此多轮历史和 MCP 连接不会每次重新初始化。
    await brain.start()
    try:
        yield
    finally:
        # 语音断开后生产任务仍完成当前工具请求；退出前等它写好操作记录。
        if stream_tasks:
            await asyncio.gather(*stream_tasks, return_exceptions=True)
        await brain.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True, "ha_available": brain.ha_available}


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    # FastAPI 把请求 JSON 转成 AskRequest，并根据 Field 检查长度；这里只交给大脑处理。
    # 语音层被打断时会断开连接，但这里不会被取消：正在执行的工具调用会完整做完，
    # 避免“写到一半”的状态；之后语音层再通过 /interrupted 告诉大脑用户实际听到了哪里。
    try:
        answer = await brain.ask(body.text)
    except Exception as exc:
        logger.exception("brain.ask 失败")
        raise HTTPException(status_code=502, detail="大脑处理失败") from exc
    return AskResponse(answer=answer, timings=brain.last_timings)


@app.post("/interrupted")
async def interrupted(body: InterruptedRequest) -> dict[str, bool]:
    return {"ok": await brain.mark_interrupted(body.spoken)}


@app.post("/ask/stream")
async def ask_stream(body: AskRequest) -> StreamingResponse:
    """NDJSON：delta（文字增量）、done（完成及耗时）、error（不可自动重试）。"""
    events: asyncio.Queue[dict] = asyncio.Queue()

    async def emit(text: str) -> None:
        events.put_nowait({"type": "delta", "text": text})

    async def signal(name: str) -> None:
        events.put_nowait({"type": "signal", "name": name})

    async def produce() -> None:
        try:
            await brain.ask(body.text, on_text=emit, on_signal=signal)
            events.put_nowait({"type": "done", "timings": brain.last_timings})
        except Exception:
            logger.exception("brain.ask 流式处理失败")
            events.put_nowait({"type": "error", "text": "未能拿到完整结果，操作可能已生效，请先查询确认。"})

    # 与 HTTP 消费者分开，断线不取消正在发出的工具操作，也不会自动重发。
    task = asyncio.create_task(produce(), name="brain-stream")
    stream_tasks.add(task)
    task.add_done_callback(stream_tasks.discard)

    async def content() -> AsyncIterator[str]:
        while True:
            event = await events.get()
            yield json.dumps(event, ensure_ascii=False) + "\n"
            if event["type"] in {"done", "error"}:
                return

    return StreamingResponse(content(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8765)
