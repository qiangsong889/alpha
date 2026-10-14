"""在一个后台任务中持有 personal MCP 会话，串行处理请求并重连。"""

import asyncio
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult


logger = logging.getLogger("alpha.personal_worker")


class PersonalWorker:
    def __init__(self, server: StdioServerParameters, timeout: float = 60.0):
        self.server = server
        self.timeout = timeout
        self.queue: asyncio.Queue[
            tuple[str, dict[str, Any], asyncio.Future[CallToolResult]]
        ] = asyncio.Queue()
        self.closed = False

    async def call(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        if self.closed:
            raise RuntimeError("PersonalWorker 已关闭")
        future: asyncio.Future[CallToolResult] = asyncio.get_running_loop().create_future()
        try:
            # 超时包括排队、等待连接和实际调用。
            async with asyncio.timeout(self.timeout):
                await self.queue.put((name, arguments, future))
                return await future
        finally:
            if not future.done():
                future.cancel()

    async def run(self) -> None:
        current: asyncio.Future[CallToolResult] | None = None
        try:
            while True:
                try:
                    # 在同一个任务中进入、使用和退出连接上下文。
                    async with stdio_client(self.server) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            while True:
                                name, arguments, current = await self.queue.get()
                                try:
                                    # 已超时且尚未执行的请求直接丢弃。
                                    if current.done():
                                        continue
                                    result = await session.call_tool(name, arguments=arguments)
                                    if not isinstance(result, CallToolResult):
                                        raise RuntimeError("personal 工具未返回完整结果")
                                    if not current.done():
                                        current.set_result(result)
                                finally:
                                    self.queue.task_done()
                                current = None
                except Exception as exc:
                    if current is not None and not current.done():
                        current.set_exception(exc)
                    current = None
                    # 不重放当前请求；服务恢复后继续处理队列中的新请求。
                    logger.warning("个人服务连接失败，1 秒后重连：%s", exc)
                    await asyncio.sleep(1)
        finally:
            self.closed = True
            if current is not None and not current.done():
                current.cancel()
            while not self.queue.empty():
                _, _, future = self.queue.get_nowait()
                if not future.done():
                    future.cancel()
                self.queue.task_done()


async def stop_worker(task: asyncio.Task[None]) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
