"""延迟提交写操作；生命周期属于程序，不属于某一条 WebSocket 连接。"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import math

from integrations.operations import Operation
from integrations.tool_executor import format_for_model
from integrations.tool_gateway import ToolGateway


@dataclass(frozen=True)
class CommitReport:
    operations: tuple[Operation, ...]
    text: str


class CommitWindow:
    def __init__(self, gateway: ToolGateway, window_s: float = 2.0,
                 on_report: Callable[[CommitReport], Awaitable[None]] | None = None):
        if not math.isfinite(window_s) or window_s < 0:
            raise ValueError("window_s 必须是有限的非负数")
        self.gateway, self.window_s = gateway, window_s
        # 默认报告收件箱也跨重连保留，新的 receiver 可以继续领取。
        self.reports: asyncio.Queue[CommitReport] = asyncio.Queue()
        self.on_report = on_report or self.reports.put
        self._pending: dict[str, None] = {}
        self._timer: asyncio.Task[None] | None = None
        self._execution: asyncio.Task[None] | None = None
        self._waiting_for_turn = True
        self._speech = False
        self._closed = False

    def add(self, op: Operation) -> None:
        if self._closed:
            raise RuntimeError("提交窗口已经关闭")
        if op.status != "pending":
            raise ValueError("只接受 pending 操作")
        if op.id in self._pending:
            return
        self._pending[op.id] = None
        # 新操作必须等包含它的最终回答结束，不能搭上旧一轮的计时器。
        self._waiting_for_turn = True
        self._stop_timer()

    def turn_finished(self) -> None:
        if self._closed:
            return
        self._waiting_for_turn = False
        self._start_timer()

    def speech_started(self) -> None:
        self._speech = True
        self._stop_timer()

    def speech_ignored(self) -> None:
        self._speech = False
        self._start_timer()

    def speech_accepted(self) -> None:
        self._speech = False
        self._waiting_for_turn = True
        self._stop_timer()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _start_timer(self) -> None:
        if (self._closed or self._speech or self._waiting_for_turn or not self._pending
                or self._timer is not None or self._execution is not None):
            return
        self._timer = asyncio.create_task(self._wait(), name="commit-window-timer")

    async def _wait(self) -> None:
        try:
            await asyncio.sleep(self.window_s)
            # 从检查到创建执行任务没有 await。speech_started 只取消计时任务。
            self._timer = None
            if not self._speech and not self._waiting_for_turn and not self._closed:
                self._execution = asyncio.create_task(self._execute(), name="commit-window-execute")
        except asyncio.CancelledError:
            pass

    async def _execute(self) -> None:
        completed: list[Operation] = []
        try:
            # 固定本批 ID；执行期间新加的操作需要等新一轮结束。
            for op_id in tuple(self._pending):
                if self._speech or self._waiting_for_turn or self._closed:
                    break
                self._pending.pop(op_id)
                op: Operation | None
                # approve 与 execute 的 approved→executing 之间没有让出事件循环。
                if self.gateway.approve(op_id):
                    op = await self.gateway.execute(op_id)
                else:
                    op = self.gateway.store.get(op_id)
                if op is not None:
                    completed.append(op)
            if completed:
                text = "\n".join(
                    f"#{op.id} {op.name} [{op.status}]：{format_for_model(op)['text']}"
                    for op in completed
                )
                await self.on_report(CommitReport(tuple(completed), text))
        finally:
            self._execution = None
            self._start_timer()

    async def close(self) -> None:
        """退出时停止后台任务；已发出的请求取消后由 gateway 记为 unknown。"""
        self._closed = True
        timer, execution = self._timer, self._execution
        self._stop_timer()
        if execution is not None:
            execution.cancel()
        await asyncio.gather(*(task for task in (timer, execution) if task is not None),
                             return_exceptions=True)
