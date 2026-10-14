"""把 AlphaBrain 包成一个本机 HTTP 服务，供语音层（LiveKit 等）调用。

为什么单独开一个进程：LiveKit 依赖旧版 anthropic / openai / mcp，和 Alpha 用的新版冲突，
两边不能装在同一个虚拟环境里。用 HTTP 隔开，两边各用各的依赖。

运行（alpha 的虚拟环境）：
    .venv/bin/python brain_server.py
接口：
    POST /ask          {"text": "..."}    → {"answer": "...", "timings": [...]}
    POST /interrupted  {"spoken": "..."}  → {"ok": true}
    GET  /health
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from brain.agent import AlphaBrain

logger = logging.getLogger("alpha.brain_server")
brain = AlphaBrain()


class AskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    timings: list[tuple[str, float]]


class InterruptedRequest(BaseModel):
    spoken: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await brain.start()
    try:
        yield
    finally:
        await brain.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True, "ha_available": brain.ha_available}


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8765)
