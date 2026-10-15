"""任务链与批次的纯逻辑，暂未接入语音流程。

TaskChain 是一次持续安排任务的上下文，多个批次共享递增序号。
这里只记录意图和状态，不执行工具、不改 OperationStore、不管理计时或确认资格。
取消已登记的任务后，接线层仍需通过 gateway.cancel 同步数据库。
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from typing import Any, Literal, cast
from uuid import uuid4

from core.plan import action_count, validate_arguments
from integrations.operations import Operation
from integrations.tool_policy import get_policy


BatchStatus = Literal[
    "collecting", "waiting_for_clarification", "awaiting_confirmation",
    "queued", "executing", "done", "cancelled",
]
TaskStatus = Literal[
    "pending", "queued", "executing", "succeeded", "failed", "unknown",
    "cancelled", "skipped",
]
CancellationEffect = Literal["cancel", "compensate", "verify", "none"]

_EDITABLE = {"collecting", "waiting_for_clarification", "awaiting_confirmation"}
_TERMINAL = {"succeeded", "failed", "unknown", "cancelled", "skipped"}
_TASK_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    "pending": {"queued", "cancelled"},
    "queued": {"executing", "cancelled", "skipped"},
    "executing": {"succeeded", "failed", "unknown"},
}


@dataclass(frozen=True)
class Task:
    task_id: int
    sequence: int
    batch_id: str
    source_turn_id: str
    operation_id: str
    name: str
    arguments: dict[str, Any]
    status: TaskStatus = "pending"
    depends_on: tuple[int, ...] = ()
    replaces_task_id: int | None = None
    compensates_task_id: int | None = None


@dataclass(frozen=True)
class TaskBatch:
    id: str
    version: int = 0
    status: BatchStatus = "collecting"
    # 包括已取消、被替换的记录；effective_tasks 才是当前有效任务。
    task_ids: tuple[int, ...] = ()


class TaskChain:
    """同一条任务链内收集多个批次；历史任务链应使用另一个实例。

sequence 包括用户输入边界和任务变更。调用方按转写中的顺序调用方法，
不要把同一句里的所有新增排在所有取消之前，否则会丢失用户的原始先后关系。
返回值均为快照；外部修改嵌套 arguments 不会修改内部记录。
"""

    def __init__(self) -> None:
        self.id = uuid4().hex
        self._sequence = 0
        self._tasks: dict[int, Task] = {}
        self._batches: dict[str, TaskBatch] = {}

    def mark_input(self) -> int:
        """为当前指令分配先后边界；取消目标必须早于这个边界。"""
        self._sequence += 1
        return self._sequence

    def new_batch(self) -> TaskBatch:
        if any(batch.status in _EDITABLE for batch in self._batches.values()):
            raise ValueError("已有正在收集的批次")
        batch = TaskBatch(uuid4().hex)
        self._batches[batch.id] = batch
        return batch

    def batch(self, batch_id: str) -> TaskBatch:
        return self._batches[batch_id]

    @property
    def batches(self) -> tuple[TaskBatch, ...]:
        return tuple(self._batches.values())

    @property
    def all_tasks(self) -> tuple[Task, ...]:
        return tuple(self.task(key) for key in self._tasks)

    def task(self, task_id: int) -> Task:
        return deepcopy(self._tasks[task_id])

    def tasks(self, batch_id: str) -> tuple[Task, ...]:
        return tuple(self.task(key) for key in self.batch(batch_id).task_ids)

    def effective_tasks(self, batch_id: str) -> tuple[Task, ...]:
        return tuple(task for task in self.tasks(batch_id)
                     if task.status not in {"cancelled", "skipped"})

    def _changed(self, batch_id: str, *, task_ids: tuple[int, ...] | None = None) -> None:
        batch = self.batch(batch_id)
        self._batches[batch_id] = replace(
            batch, version=batch.version + 1,
            status="collecting" if batch.status in _EDITABLE else batch.status,
            task_ids=batch.task_ids if task_ids is None else task_ids,
        )

    def _validate_new(
        self, batch_id: str, op: Operation, dependencies: tuple[int, ...],
    ) -> None:
        if self.batch(batch_id).status not in _EDITABLE:
            raise ValueError("批次已提交，新增任务应放入下一批")
        policy = get_policy(op.name)
        if policy is None or not policy.mutates or op.status != "pending":
            raise ValueError("只收集白名单内尚未执行的写操作")
        validate_arguments(op)
        json.dumps(op.arguments, allow_nan=False)
        if any(task.operation_id == op.id for task in self._tasks.values()):
            raise ValueError("同一 operation 不能登记两次")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("依赖不能重复")
        for key in dependencies:
            if type(key) is not int or key not in self._tasks:
                raise ValueError("依赖目标不存在")
            target = self._tasks[key]
            batch_ids = list(self._batches)
            if batch_ids.index(target.batch_id) > batch_ids.index(batch_id):
                raise ValueError("不能依赖未来批次的任务")
            if target.status in {"cancelled", "skipped", "failed", "unknown"}:
                raise ValueError("依赖目标不可用")

    def add(
        self, batch_id: str, op: Operation, *, depends_on: tuple[int, ...] = (),
        compensates_task_id: int | None = None,
        replaces_task_id: int | None = None,
    ) -> Task:
        self._validate_new(batch_id, op, depends_on)
        if compensates_task_id is not None:
            target = self._tasks.get(compensates_task_id)
            if target is None or target.status != "succeeded":
                raise ValueError("只能为已成功任务登记撤销任务")
        if replaces_task_id is not None:
            old = self._tasks.get(replaces_task_id)
            if old is None or old.status != "cancelled":
                raise ValueError("替换关系必须指向已取消的旧任务")
        key = len(self._tasks) + 1  # 保留取消记录，编号不回收。
        sequence = self.mark_input()
        task = Task(key, sequence, batch_id, op.turn_id, op.id, op.name,
                    deepcopy(op.arguments), depends_on=tuple(depends_on),
                    compensates_task_id=compensates_task_id, replaces_task_id=replaces_task_id)
        self._tasks[key] = task
        self._changed(batch_id, task_ids=(*self.batch(batch_id).task_ids, key))
        return self.task(key)

    def replace(self, task_id: int, op: Operation) -> Task:
        """替换收集中的任务，保留旧记录；已有依赖者时要求先明确调整依赖。"""
        old = self._tasks[task_id]
        if old.status != "pending" or self.batch(old.batch_id).status not in _EDITABLE:
            raise ValueError("只能替换尚在收集的任务")
        if self.dependents(task_id):
            raise ValueError("存在依赖此任务的操作，请先调整依赖")
        self._validate_new(old.batch_id, op, old.depends_on)
        # add 成功之后才取消旧记录；参数无效时旧任务和版本均保持原样。
        new = self.add(old.batch_id, op, depends_on=old.depends_on,
                       compensates_task_id=old.compensates_task_id)
        self._tasks[task_id] = replace(old, status="cancelled")
        self._tasks[new.task_id] = replace(new, replaces_task_id=task_id)
        return self.task(new.task_id)

    def dependents(self, task_id: int) -> tuple[Task, ...]:
        return tuple(self.task(task.task_id) for task in self._tasks.values()
                     if task_id in task.depends_on and task.status not in _TERMINAL)

    def set_dependencies(self, task_id: int, dependencies: tuple[int, ...]) -> None:
        """支持后补“开灯之前先关窗帘”；只改收集中的任务，拒绝依赖环。"""
        task = self._tasks[task_id]
        if task.status != "pending" or self.batch(task.batch_id).status not in _EDITABLE:
            raise ValueError("只能修改收集中的任务依赖")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("依赖不能重复")
        batch_ids = list(self._batches)
        for key in dependencies:
            if type(key) is not int or key not in self._tasks:
                raise ValueError("依赖目标不存在")
            target = self._tasks[key]
            if (batch_ids.index(target.batch_id) > batch_ids.index(task.batch_id)
                    or target.status in {"cancelled", "skipped", "failed", "unknown"}):
                raise ValueError("依赖目标不可用")
            todo, seen = [key], set()
            while todo:
                current = todo.pop()
                if current == task_id:
                    raise ValueError("依赖不能成环")
                if current not in seen:
                    seen.add(current)
                    todo.extend(self._tasks[current].depends_on)
        self._tasks[task_id] = replace(task, depends_on=tuple(dependencies))
        self._changed(task.batch_id)

    def targets_before(self, sequence: int, *, name: str | None = None) -> tuple[Task, ...]:
        """只返回可供定位的历史候选，不猜测“牛奶”指哪项，也不自动取消同名项。"""
        if type(sequence) is not int or not 0 < sequence <= self._sequence:
            raise ValueError("必须使用本任务链已分配的输入边界")
        return tuple(self.task(task.task_id) for task in self._tasks.values()
                     if task.sequence < sequence
                     and task.status not in {"failed", "cancelled", "skipped"}
                     and (name is None or task.name == name))

    def cancellation_effect(self, task_id: int) -> CancellationEffect:
        status = self._tasks[task_id].status
        if status in {"pending", "queued"}:
            return "cancel"
        if status == "succeeded":
            return "compensate"
        if status in {"executing", "unknown"}:
            return "verify"
        return "none"

    def cancel(self, task_id: int) -> bool:
        """只取消未发送的任务；不会声称已撤销执行中的请求或已完成的操作。"""
        if self.cancellation_effect(task_id) != "cancel":
            return False
        task = self._tasks[task_id]
        self._tasks[task_id] = replace(task, status="cancelled")
        self.mark_input()
        self._changed(task.batch_id)
        batch = self.batch(task.batch_id)
        if batch.status in {"queued", "executing"} and all(
            self._tasks[key].status in _TERMINAL for key in batch.task_ids
        ):
            self._batches[batch.id] = replace(
                batch, status="cancelled" if all(
                    self._tasks[key].status == "cancelled" for key in batch.task_ids
                ) else "done",
            )
        return True

    def requires_confirmation(self, batch_id: str) -> bool:
        """新流程的门槛为三个动作；旧 needs_plan 的语音行为暂时保持原样。"""
        total = 0
        for task in self.effective_tasks(batch_id):
            policy = get_policy(task.name)
            if policy is None:
                raise ValueError("任务策略已不可用")
            if policy.commit == "confirm":
                return True
            total += action_count(Operation(
                task.operation_id, task.source_turn_id, task.name, task.arguments,
                "pending", None, 0, 0,
            ))
        return total >= 3

    def set_collection_status(
        self, batch_id: str,
        status: Literal["collecting", "waiting_for_clarification", "awaiting_confirmation"],
    ) -> None:
        if status not in _EDITABLE or self.batch(batch_id).status not in _EDITABLE:
            raise ValueError("只能调整收集中的批次")
        self._batches[batch_id] = replace(self.batch(batch_id), status=status)

    def queue(self, batch_id: str, *, version: int, confirmed: bool = False) -> bool:
        """登记提交快照；计时、完整性和播放资格由后续协调层检查。

        confirmed 只能由确认协调层给出，不能直接采用模型传来的布尔值。
        """
        batch = self.batch(batch_id)
        tasks = self.effective_tasks(batch_id)
        if (batch.version != version or batch.status not in {"collecting", "awaiting_confirmation"}
                or not tasks or (self.requires_confirmation(batch_id) and not confirmed)):
            return False
        self._batches[batch_id] = replace(batch, status="queued")
        for task in tasks:
            self._tasks[task.task_id] = replace(task, status="queued")
        return True

    def transition_task(self, task_id: int, status: TaskStatus) -> bool:
        """由未来执行层同步真实状态；自身绝不发送工具请求。"""
        task = self._tasks[task_id]
        if status not in _TASK_TRANSITIONS.get(task.status, set()):
            return False
        if status == "cancelled":
            return self.cancel(task_id)
        if status == "queued":  # 必须通过整个批次的提交检查。
            return False
        if status == "executing" and any(
            self._tasks[key].status != "succeeded" for key in task.depends_on
        ):
            return False
        self._tasks[task_id] = replace(task, status=status)
        batch = self.batch(task.batch_id)
        self._batches[batch.id] = replace(batch, status="executing")
        if all(self._tasks[key].status in _TERMINAL for key in batch.task_ids):
            self._batches[batch.id] = replace(self.batch(batch.id), status="done")
        return True

    def observe_operation(self, task_id: int, op: Operation) -> None:
        """同步 gateway/store 的真实结果，不代表批准或发送请求。

        执行队列调用此方法，避免内存说 cancelled、数据库却已经 succeeded。
        已发生的事实不再受依赖检查阻挡；依赖检查仍在发送请求前完成。
        """
        task = self._tasks[task_id]
        if (task.operation_id != op.id or task.name != op.name
                or json.dumps(task.arguments, sort_keys=True, allow_nan=False)
                != json.dumps(op.arguments, sort_keys=True, allow_nan=False)):
            raise ValueError("操作与任务不匹配")
        if op.status in {"pending", "approved"}:
            return
        status = cast(TaskStatus, op.status)
        if task.status in _TERMINAL:
            if task.status == "unknown" and status == "succeeded":
                self._tasks[task_id] = replace(task, status=status)
                return  # 可信核实入口已更新数据库；不重新执行旧批次。
            if task.status != status and not (task.status == "skipped" and status == "cancelled"):
                raise ValueError("不能覆盖任务的最终结果")
            return
        self._tasks[task_id] = replace(task, status=status)
        batch = self.batch(task.batch_id)
        if all(self._tasks[key].status in _TERMINAL for key in batch.task_ids):
            end: BatchStatus = "cancelled" if all(
                self._tasks[key].status == "cancelled" for key in batch.task_ids
            ) else "done"
            self._batches[batch.id] = replace(batch, status=end)
        elif status == "executing":
            self._batches[batch.id] = replace(batch, status="executing")
