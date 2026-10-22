"""把 AlphaBrain 包成一个本机 HTTP 服务，供语音层（LiveKit 等）调用。

为什么单独开一个进程：LiveKit 依赖旧版 anthropic / openai / mcp，和 Alpha 用的新版冲突，
两边不能装在同一个虚拟环境里。用 HTTP 隔开，两边各用各的依赖。

运行（alpha 的虚拟环境）：
    .venv/bin/python -m brain.server
接口：
    POST /ask          {"text": "..."}    → {"answer": "...", "timings": [...]}
    POST /ask/stream   {"text": "..."}    → NDJSON delta / done / error
    POST /interrupted  {"spoken": "..."}  → {"ok": true}
    POST /delivered    {"turn_id": "...", "delivery": "played|interrupted|not_played"} → {"ok": bool}
    GET  /health
    GET  /status       运行状态（配置指纹、生效中的参数、依赖状态、心跳）
"""

import logging
import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Literal
from uuid import uuid4
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from brain.agent import AlphaBrain
from core.tool_timing import TOOL_TIMING_SINK
from core.costs import configure_costs, cost_summary, log_summary
from core.paths import ARCHIVE_DB

logger = logging.getLogger("alpha.brain_server")
brain = AlphaBrain(style="voice", archive_path=ARCHIVE_DB)
stream_tasks: set[asyncio.Task[None]] = set()


class AskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    search_announced: bool = False
    request_id: str | None = Field(default=None, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    interruptions: list['InterruptedRequest'] = Field(default_factory=list, max_length=200)


class AskResponse(BaseModel):
    answer: str
    timings: list[tuple[str, float]]


class InterruptedRequest(BaseModel):
    spoken: str = Field(default='', max_length=32000)
    turn_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$')
    session_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$')
    uncertain_text: str = Field(default='', max_length=32000)
    delivery: Literal['interrupted', 'not_played'] = 'interrupted'


AskRequest.model_rebuild()


async def apply_interruptions(items: list[InterruptedRequest]) -> None:
    for item in items:
        if item.turn_id is not None:  # 请求附带的兜底通知必须指定回答，不能套用到最新一条。
            await brain.mark_interrupted(item.spoken, turn_id=item.turn_id,
                                         session_id=item.session_id, uncertain_text=item.uncertain_text,
                                         delivery=item.delivery)


class DeliveredRequest(BaseModel):
    turn_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    delivery: Literal["played", "interrupted", "not_played"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # async context manager：yield 前是服务启动；yield 期间 FastAPI 接请求；finally 是关闭。
    # 大脑跨 HTTP 请求复用，因此多轮历史和 MCP 连接不会每次重新初始化。
    configure_costs()
    await brain.start()
    try:
        yield
    finally:
        # 语音断开后生产任务仍完成当前工具请求；退出前等它写好操作记录。
        if stream_tasks:
            await asyncio.gather(*stream_tasks, return_exceptions=True)
        await brain.close()
        log_summary()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True, "ha_available": brain.ha_available}


@app.get("/status")
async def status() -> dict:
    """只读的运行状态：配置指纹、实际在用的参数（密钥只说明是否已设置）、各依赖最近一次的状态和心跳。"""
    return brain.status()


@app.get("/costs")
async def costs() -> dict:
    return {"currency": "USD", "basis": "standard_price_estimate", "components": cost_summary()}


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    # FastAPI 把请求 JSON 转成 AskRequest，并根据 Field 检查长度；这里只交给大脑处理。
    # 语音层被打断时会断开连接，但这里不会被取消：正在执行的工具调用会完整做完，
    # 避免“写到一半”的状态；之后语音层再通过 /interrupted 告诉大脑用户实际听到了哪里。
    try:
        await apply_interruptions(body.interruptions)
        answer = await brain.ask(body.text, search_announced=body.search_announced, turn_id=uuid4().hex)
    except Exception as exc:
        logger.exception("brain.ask 失败")
        raise HTTPException(status_code=502, detail="大脑处理失败") from exc
    return AskResponse(answer=answer, timings=brain.last_timings)


@app.post("/interrupted")
async def interrupted(body: InterruptedRequest) -> dict[str, bool]:
    return {"ok": await brain.mark_interrupted(body.spoken, turn_id=body.turn_id,
                                               session_id=body.session_id, uncertain_text=body.uncertain_text,
                                               delivery=body.delivery)}


@app.post("/delivered")
async def delivered(body: DeliveredRequest) -> dict[str, bool]:
    """语音层报告某一轮回答的实际播放结果；只更新存档，不改变对话历史或已执行的操作。"""
    return {"ok": await brain.mark_delivery(body.turn_id, body.delivery)}


@app.post("/ask/stream")
async def ask_stream(body: AskRequest) -> StreamingResponse:
    """NDJSON：delta（文字增量）、done（完成及耗时）、error（不可自动重试）。"""
    events: asyncio.Queue[dict] = asyncio.Queue()

    async def emit(text: str) -> None:
        events.put_nowait({"type": "delta", "text": text})

    async def signal(name: str) -> None:
        events.put_nowait({"type": "signal", "name": name})

    turn_id = uuid4().hex
    if body.request_id is not None:
        # 只有声明了 request_id 的客户端（Qwen 语音层）才会收到 turn 事件；其他客户端的协议不变。
        events.put_nowait({"type": "turn", "request_id": body.request_id, "turn_id": turn_id})

    async def produce() -> None:
        started = time.perf_counter()
        def tool_timing(event):
            events.put_nowait({'type': 'tool_timing', 'request_id': body.request_id,
                              **event, 'elapsed_s': time.perf_counter() - started})
        token = TOOL_TIMING_SINK.set(tool_timing if body.request_id is not None else None)
        try:
            await apply_interruptions(body.interruptions)
            await brain.ask(body.text, on_text=emit, on_signal=signal,
                            search_announced=body.search_announced, turn_id=turn_id)
            done = {"type": "done", "timings": brain.last_timings}
            if body.request_id is not None:
                done['session_id'] = brain.session_id
            events.put_nowait(done)
        except Exception:
            logger.exception("brain.ask 流式处理失败")
            events.put_nowait({"type": "error", "text": "未能拿到完整结果，操作可能已生效，请先查询确认。"})
        finally:
            TOOL_TIMING_SINK.reset(token)

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
