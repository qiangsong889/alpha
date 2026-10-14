"""在一个后台任务中持有 personal MCP 会话，串行处理请求并重连。"""

import asyncio
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult


logger = logging.getLogger("alpha.personal_worker")


class PersonalWorker:
    # 把 MCP 长连接交给一个固定的后台任务管理；其他任务通过 call() 提交请求。
    # Google 工具在 server 端执行，这里负责消息传递，不在客户端实现 Google API。
    def __init__(self, server: StdioServerParameters, timeout: float = 60.0):
        self.server = server
        self.timeout = timeout
        # 一项队列数据有三个部分：工具名、参数，以及“这次请求将来的结果”future。
        self.queue: asyncio.Queue[
            tuple[str, dict[str, Any], asyncio.Future[CallToolResult]]
        ] = asyncio.Queue()
        self.closed = False

    async def call(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        if self.closed:
            raise RuntimeError("PersonalWorker 已关闭")
        # future 像一张待填的结果单：call() 等它，run() 拿到结果后负责填入。
        future: asyncio.Future[CallToolResult] = asyncio.get_running_loop().create_future()
        try:
            # 超时包括排队、等待连接和实际调用。
            async with asyncio.timeout(self.timeout):
                await self.queue.put((name, arguments, future))
                # 暂停的是这次 call 所在任务；worker.run、网络接收、播放任务仍可继续。
                return await future
        finally:
            if not future.done():
                future.cancel()

    async def run(self) -> None:
        current: asyncio.Future[CallToolResult] | None = None
        try:
            while True:
                # 外层循环负责重连；内层队列循环负责在同一个连接上依次执行工具。
                try:
                    # 在同一个任务中进入、使用和退出连接上下文。
                    async with stdio_client(self.server) as (read, write):
                        # stdio_client 启动子进程并提供读写通道；ClientSession 在其上说 MCP 协议。
                        async with ClientSession(read, write) as session:
                            # 握手后才知道对端协议和能力；连接建立不等于握手已经完成。
                            await session.initialize()
                            while True:
                                name, arguments, current = await self.queue.get()
                                # 没有请求时只挂起 worker，不会卡住整个进程或其他异步任务。
                                try:
                                    # 已超时且尚未执行的请求直接丢弃。
                                    if current.done():
                                        continue
                                    result = await session.call_tool(name, arguments=arguments)
                                    if not isinstance(result, CallToolResult):
                                        raise RuntimeError("personal 工具未返回完整结果")
                                    if not current.done():
                                        # 唤醒等待此 future 的 call()，它接着返回结果给模型处理代码。
                                        current.set_result(result)
                                finally:
                                    self.queue.task_done()
                                current = None
                except Exception as exc:
                    if current is not None and not current.done():
                        # 不是在这里直接 raise 给调用方；给 future 填异常后，await future 会抛出。
                        current.set_exception(exc)
                    current = None
                    # 不重放当前请求；服务恢复后继续处理队列中的新请求。
                    logger.warning("个人服务连接失败，1 秒后重连：%s", exc)
                    await asyncio.sleep(1)
        finally:
            # 退出或被 cancel 时，取消还在等待的 future，避免调用方永远等不到结果。
            self.closed = True
            if current is not None and not current.done():
                current.cancel()
            while not self.queue.empty():
                _, _, future = self.queue.get_nowait()
                if not future.done():
                    future.cancel()
                self.queue.task_done()


async def stop_worker(task: asyncio.Task[None]) -> None:
    # cancel 是发出取消请求；再 await 才能等 worker 跑完 finally 和连接上下文的清理。
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
