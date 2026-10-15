"""串行执行已提交批次；尚未接入 Qwen，不管理说话、等待窗口或用户确认。"""

import asyncio
from dataclasses import dataclass
import json

from core.task_batch import Task, TaskChain
from integrations.operations import Operation
from integrations.tool_executor import format_for_model
from integrations.tool_gateway import ToolGateway


@dataclass(frozen=True)
class TaskReport:
    task: Task
    operation: Operation
    text: str


@dataclass(frozen=True)
class BatchReport:
    batch_id: str
    tasks: tuple[TaskReport, ...]

    @property
    def text(self) -> str:
        return "\n".join(f"#{item.task.task_id} {item.task.name} "
                         f"[{item.task.status}]：{item.text}" for item in self.tasks)


def _same_arguments(first: dict, second: dict) -> bool:
    # 比较 JSON 含义，同时保留数组顺序；不能让 Python 把 True 与 1 当作相同参数。
    return json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False,
    )


class BatchExecutor:
    """一个实例只有一个 run 任务，同一时间最多发送一个写操作。

    入队前由业务层完成窗口、确认和 chain.queue。这里只通过 gateway.execute
    发送请求，不调用 gateway.submit，避免 immediate 策略绕过任务收集。
    执行中的取消不能撤销远端请求；cancel 只处理尚未发出的任务。
    """

    def __init__(self, chain: TaskChain, gateway: ToolGateway) -> None:
        self.chain, self.gateway = chain, gateway
        self.reports: asyncio.Queue[BatchReport] = asyncio.Queue()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._accepted: dict[str, None] = {}
        self._reported: set[str] = set()
        self._reasons: dict[int, str] = {}
        self._runner: asyncio.Task | None = None
        self._closed = False

    def _operation(self, task: Task) -> Operation:
        op = self.gateway.store.get(task.operation_id)
        if (op is None or op.name != task.name
                or not _same_arguments(op.arguments, task.arguments)):
            raise ValueError(f"任务 #{task.task_id} 与数据库操作不匹配")
        return op

    def enqueue(self, batch_id: str) -> bool:
        """只接收已提交批次，整个批次校验通过后才入队；重复入队不重放。"""
        if self._closed:
            raise RuntimeError("执行队列已经关闭")
        if batch_id in self._accepted:
            return False
        if self.chain.batch(batch_id).status != "queued":
            raise ValueError("批次必须先通过提交检查")
        tasks = self.chain.tasks(batch_id)
        for task in tasks:
            op = self._operation(task)
            if task.status in {"cancelled", "skipped"} and op.status not in {
                "pending", "approved", "cancelled",
            }:
                raise ValueError("任务已被标为取消，但对应操作已经执行，需先核实")
            for dependency in task.depends_on:
                previous = self.chain.task(dependency)
                if (previous.batch_id != batch_id and previous.status in {"pending", "queued", "executing"}
                        and previous.batch_id not in self._accepted):
                    raise ValueError("依赖的前置批次尚未进入执行队列")
        # 先完成全部校验，再同步收集阶段取消/替换留下的 pending 记录。
        for task in tasks:
            if task.status in {"cancelled", "skipped"}:
                self.gateway.cancel(task.operation_id)
        self._accepted[batch_id] = None
        self._queue.put_nowait(batch_id)
        return True

    def cancel(self, task_id: int) -> bool:
        """先取消真实操作，再同步内存；检查和登记之间没有 await。"""
        task = self.chain.task(task_id)
        op = self._operation(task)
        if self.gateway.cancel(op.id):
            return self.chain.cancel(task_id)
        self.chain.observe_operation(task_id, self._operation(task))
        return False

    def cancel_remaining(self, batch_id: str | None = None) -> tuple[int, ...]:
        """取消已入队批次的剩余任务；收集中的下一批由收集协调层另行处理。"""
        if batch_id is not None and batch_id not in self._accepted:
            raise ValueError("批次尚未进入执行队列")
        cancelled = []
        for key in self._accepted:
            if batch_id is not None and key != batch_id:
                continue
            for task in self.chain.tasks(key):
                if task.status in {"pending", "queued"} and self.cancel(task.task_id):
                    cancelled.append(task.task_id)
        return tuple(cancelled)

    async def run(self) -> None:
        if self._closed:
            raise RuntimeError("执行队列已经关闭")
        if self._runner is not None:
            raise RuntimeError("执行队列已在运行")
        self._runner = asyncio.current_task()
        try:
            while True:
                batch_id = await self._queue.get()
                try:
                    await self._execute_batch(batch_id)
                finally:
                    self._queue.task_done()
        finally:
            # run 被取消也按退出处理：未发送的操作明确取消，不能遗留无人管理的 pending。
            self._closed = True
            self._cancel_unstarted()
            self._runner = None

    def _skip(self, task: Task, reason: str) -> None:
        op = self._operation(task)
        if self.gateway.cancel(op.id):
            self.chain.transition_task(task.task_id, "skipped")
            self._reasons[task.task_id] = reason
        else:
            self.chain.observe_operation(task.task_id, self._operation(task))

    async def _execute_batch(self, batch_id: str) -> None:
        while True:
            # 重新读取每一项；执行期间 cancel 或模型对下一批的修改不会使用旧快照。
            pending = [task for task in self.chain.tasks(batch_id) if task.status == "queued"]
            if not pending:
                self._report(batch_id)
                return
            progressed = False
            for task in pending:
                op = self._operation(task)
                if op.status == "cancelled" or task.status == "cancelled":
                    self.chain.observe_operation(task.task_id, op)
                    progressed = True
                    continue
                dependencies = [self.chain.task(key) for key in task.depends_on]
                if any(dep.status in {"failed", "cancelled", "skipped", "unknown"}
                       for dep in dependencies):
                    self._skip(task, "依赖未成功，本项没有执行；依赖结果未知时需先查询核实")
                    progressed = True
                    continue
                if any(dep.status != "succeeded" for dep in dependencies):
                    continue
                if op.status in {"succeeded", "failed", "unknown"}:
                    self.chain.observe_operation(task.task_id, op)
                    progressed = True
                    continue
                if op.status == "executing":
                    # 此操作已被其他入口认领，不能重放或越过它继续执行依赖项。
                    raise RuntimeError(f"任务 #{task.task_id} 已由其他执行者认领")
                if op.status == "pending" and not self.gateway.approve(op.id):
                    self._skip(task, "操作未获批准，本项没有执行")
                    progressed = True
                    continue
                if not self.chain.transition_task(task.task_id, "executing"):
                    raise RuntimeError("任务状态或依赖在执行前发生变化")
                # 从最后一次检查、approve 到 gateway 的 executing 登记之间没有 await。
                try:
                    result = await self.gateway.execute(op.id)
                finally:
                    # Gateway 在任务被取消时记 unknown 后重抛；先同步事实再退出。
                    observed = self._operation(task)
                    if observed.status in {"pending", "approved"}:
                        self.gateway.cancel(observed.id)
                        observed = self._operation(task)
                    self.chain.observe_operation(task.task_id, observed)
                if result.status in {"pending", "approved", "executing"}:
                    raise RuntimeError(f"任务 #{task.task_id} 未获得最终结果")
                progressed = True
                break  # 一次只执行一项，下一次从头检查依赖和取消。
            if not progressed:
                # 例如依赖的前一批根本没有进入此队列。不要忙循环或跳过顺序执行。
                raise RuntimeError("批次依赖无法推进，请检查前置批次是否已提交执行")

    def _report(self, batch_id: str) -> None:
        if batch_id in self._reported:
            return
        entries = tuple(TaskReport(
            task, self._operation(task), self._reasons.get(
                task.task_id, str(format_for_model(self._operation(task))["text"]),
            ),
        ) for task in self.chain.tasks(batch_id))
        self.reports.put_nowait(BatchReport(batch_id, entries))
        self._reported.add(batch_id)

    def _cancel_unstarted(self) -> None:
        for batch_id in self._accepted:
            for task in self.chain.tasks(batch_id):
                if task.status in {"pending", "queued"}:
                    self.cancel(task.task_id)
            self._report(batch_id)

    async def close(self) -> None:
        """关闭独立执行任务；请求已发出则记录 unknown，后续项取消。"""
        self._closed = True
        runner = self._runner
        if runner is not None:
            if runner is asyncio.current_task():
                raise RuntimeError("请从执行队列任务之外调用 close")
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)
        else:
            self._cancel_unstarted()
