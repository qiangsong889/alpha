"""收集、确认、执行和撤销的统一业务入口；不依赖 WebSocket。"""

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import time
from typing import Any
from zoneinfo import ZoneInfo

from core.plan import Plan, PlanItem, validate_arguments
from core.plan_coordinator import CoordinatorResult
from core.task_batch import TaskChain
from core.task_compensation import undo_add_items
from integrations.batch_executor import BatchExecutor
from integrations.tool_executor import format_for_model
from integrations.operations import Operation
from integrations.tool_gateway import ToolGateway
from integrations.tool_policy import get_policy
from voice.collection_window import CollectionWindow
from voice.commit_window import CommitReport


@dataclass(frozen=True)
class TaskNotice:
    kind: str
    batch_id: str
    version: int
    text: str = ""
    turn_id: str | None = None


class BatchView:
    def __init__(self, owner: "TaskCoordinator"):
        self.owner = owner

    @property
    def id(self):
        return self.owner.batch_id

    @property
    def version(self):
        return self.owner.chain.batch(self.id).version

    @property
    def status(self):
        return self.owner.chain.batch(self.id).status

    @property
    def items(self):
        return self.owner.chain.effective_tasks(self.id)

    @property
    def has_duplicates(self):
        keys = [json.dumps([task.name, task.arguments], sort_keys=True) for task in self.items]
        return len(keys) != len(set(keys))

    def recap(self, now: datetime) -> str:
        formatter = Plan(timezone_name=self.owner._timezone_name)
        local = now.astimezone(ZoneInfo(self.owner._timezone_name))
        parts = []
        for task in self.items:
            text = formatter.describe_item(PlanItem(task.task_id, task.operation_id, task.name, task.arguments), local)
            if task.name == "delete_task":
                text = "撤销刚才新增的指定清单条目"
            elif task.name == "delete_reminder":
                text = "撤销刚才创建的指定提醒"
            if task.depends_on:
                text += "（等" + "、".join(f"第{key}项" for key in task.depends_on) + "成功后执行）"
            parts.append(f"第{task.task_id}项，{text}")
        duplicate = "有相同操作，请说明要做几次。" if self.has_duplicates else ""
        return "我确认一下：" + "；".join(parts) + "。" + duplicate + "对吗？"


class TaskCoordinator:
    def __init__(self, gateway: ToolGateway, *, timezone_name="America/Los_Angeles",
                 window_s=2.0, incomplete_s=5.0, clock=time.monotonic, auto_verify=True):
        self.gateway, self._timezone_name, self.clock = gateway, timezone_name, clock
        self.chain = TaskChain()
        self.batch_id = self.chain.new_batch().id
        self.window_s, self.incomplete_s = window_s, incomplete_s
        self.window = self._window()
        self.executor = BatchExecutor(self.chain, gateway)
        self.reports: asyncio.Queue[CommitReport] = asyncio.Queue()
        self.notices: asyncio.Queue[TaskNotice] = asyncio.Queue()
        self.plan = BatchView(self)
        self.current_turn: str | None = None
        self._speech_id: str | None = None
        self._sequence = 0
        self._turn_sequence = 0
        self._used: set[str] = set()
        self._answered: set[str] = set()
        self._texts: dict[str, str] = {}
        self._decisions: dict[str, tuple[bool, str, bool]] = {}
        self._played: tuple[str, int, int, float] | None = None
        self._expired = False
        self._confirmation_request: tuple[str, str, int] | None = None
        self._force_confirm: set[str] = set()
        self._undo_after: set[int] = set()
        self._compensations: dict[tuple[int, str], str] = {}
        self._closed = False
        self._retired: list[CollectionWindow] = []
        self._runner: asyncio.Task | None = None
        self._jobs: asyncio.TaskGroup | None = None
        self._report_lock = asyncio.Lock()
        self.auto_verify = auto_verify

    def _window(self):
        return CollectionWindow(self.chain, self.batch_id, window_s=self.window_s,
                                incomplete_s=self.incomplete_s, clock=self.clock)

    def result(self, ok: bool, reason: str, operations=()) -> CoordinatorResult:
        return CoordinatorResult(ok, reason, tuple(operations), self.plan.version, self.snapshot())

    def snapshot(self):
        history = self.chain.all_tasks
        by_id = {task.task_id: task for task in history}
        selected = {task.task_id for task in history[-40:]}
        selected.update(task.task_id for task in history if task.status in {"pending", "queued", "executing"})
        # 长期开机时不把全部已结束历史反复发给模型；仍保留活动任务的完整依赖背景。
        todo = list(selected)
        while todo:
            for key in by_id[todo.pop()].depends_on:
                if key not in selected:
                    selected.add(key)
                    todo.append(key)
        visible = [task for task in history if task.task_id in selected]
        batch_ids = {task.batch_id for task in visible} | {self.batch_id}
        return {"id": self.batch_id, "version": self.plan.version, "status": self.plan.status,
                "batches": [asdict(batch) for batch in self.chain.batches if batch.id in batch_ids],
                "unknown_operations": [asdict(op) for op in self.gateway.store.unresolved()
                                       if (policy := get_policy(op.name)) is not None and policy.mutates],
                "items": [dict(asdict(task), item_id=task.task_id,
                               result=(op.result if (op := self.gateway.store.get(task.operation_id)) else None))
                          for task in visible]}

    def speech_started(self, turn_id: str | None = None):
        if turn_id is None:
            return  # 音频基类兼容事件；带输入编号的调用已由接线层完成。
        self._speech_id = turn_id
        self.window.speech_started(turn_id)

    def speech_stopped(self, turn_id: str):
        self.window.speech_stopped(turn_id)

    def speech_ignored(self):
        if self._speech_id:
            self.window.speech_ignored(self._speech_id)

    def speech_accepted(self):
        pass  # speech_started 已暂停窗口；等待解释完成后由 finish_turn 重新计时。

    def user_turn(self, turn_id: str):
        if turn_id == self.current_turn:
            return self.result(False, "duplicate_turn")
        self.current_turn = turn_id
        self._sequence += 1
        self._turn_sequence = self._sequence
        return self.result(True, "turn_received")

    def record_text(self, turn_id: str, text: str):
        self._texts[turn_id] = text

    def turn_answered(self, turn_id: str):
        self._answered.add(turn_id)

    def _input_error(self, turn_id=None):
        turn = turn_id or self.current_turn
        if not turn or turn != self.current_turn or turn in self._answered:
            return "no_user_turn"
        if turn in self._used:
            return "turn_used"
        return None

    def add(self, turn_id: str, name: str, arguments: dict) -> CoordinatorResult:
        if error := self._input_error(turn_id):
            return self.result(False, error)
        op = self.gateway.register(turn_id, name, arguments)
        try:
            task = self.chain.add(self.batch_id, op)
        except Exception:
            self.gateway.cancel(op.id)
            raise
        self._played = None
        previous = self.gateway.store.recent_succeeded(name, arguments)
        if previous and name in {"add_items", "create_reminder"}:
            self._force_confirm.add(self.batch_id)
        reason = "duplicate_requires_confirmation" if previous and self.batch_id in self._force_confirm else "collected"
        return self.result(True, reason, (op,))

    def finish_collection(self, complete: bool, question: str = "", execute_now: bool = False):
        if type(complete) is not bool or type(execute_now) is not bool:
            return self.result(False, "invalid_arguments")
        if error := self._input_error():
            return self.result(False, error)
        assert self.current_turn is not None
        self._decisions[self.current_turn] = (complete, question, execute_now)
        return self.result(True, "interpretation_recorded")

    def finish_turn(self, turn_id: str):
        if turn_id != self.current_turn or turn_id in self._used:
            return
        if not self.plan.items and turn_id not in self._decisions:
            return  # 普通聊天或查询不应在五秒后被追问“任务还完整吗”。
        # 未显式调用完整性工具时保守等待澄清，不能凭工具调用就推断长句已整理完。
        decision = self._decisions.get(turn_id, (False, "这些任务还需要补充或修改吗？", False))
        if (decision[0] and self._confirmation_request == (turn_id, self.batch_id, self.plan.version)
                and self._played is not None and self._played[:2] == (self.batch_id, self.plan.version)):
            self._used.add(turn_id)
            self._confirmation_request = None
            self._submit(confirmed=True)
            return
        if self.plan.status == "awaiting_confirmation" and turn_id not in self._decisions:
            return  # 用户只是提问，不把未确认批次自动提交。
        self.window.interpreted(turn_id, version=self.plan.version, complete=decision[0],
                                question=decision[1], execute_now=decision[2])

    def recap_played(self, plan_id: str, version: int, interrupted: bool):
        if plan_id != self.batch_id or version != self.plan.version:
            return self.result(False, "version_mismatch")
        if interrupted:
            self._played = None
            self.chain.set_collection_status(self.batch_id, "collecting")
            return self.result(False, "not_played")
        if self._played is not None and self._played[:2] == (plan_id, version):
            return self.result(True, "already_played")
        self._sequence += 1
        self._played = (plan_id, version, self._sequence, self.clock())
        self._expired = False
        self.chain.set_collection_status(self.batch_id, "awaiting_confirmation")
        return self.result(True, "played")

    def expire(self):
        if self._played and self.clock() - self._played[3] >= 120:
            self._played = None
            self._expired = True
            self.chain.set_collection_status(self.batch_id, "collecting")
            return True
        return False

    async def confirm(self, plan_id: str, version: int, duplicates_intended=False):
        if plan_id != self.batch_id or type(version) is not int or version != self.plan.version:
            return self.result(False, "version_mismatch")
        self.expire()
        if self.current_turn in self._used:
            return self.result(False, "turn_used")
        if self._expired:
            return self.result(False, "expired")
        if self._played is None or self.plan.status != "awaiting_confirmation":
            return self.result(False, "not_played")
        if error := self._input_error():
            return self.result(False, error)
        if self._turn_sequence <= self._played[2]:
            return self.result(False, "no_user_turn")
        if self.plan.has_duplicates and duplicates_intended is not True:
            return self.result(False, "quantity_ambiguous")
        assert self.current_turn is not None
        # 确认也是本句的解释结果：等整句整理完，避免“对，等等第二项改一下”提前执行。
        self._confirmation_request = (self.current_turn, plan_id, version)
        return self.result(True, "confirmation_recorded_not_submitted")

    def _submit(self, *, confirmed=False):
        old = self.batch_id
        if not self.chain.queue(old, version=self.plan.version, confirmed=confirmed):
            return False
        self.executor.enqueue(old)
        self.window.suspend()
        self._retired.append(self.window)
        self.batch_id = self.chain.new_batch().id
        self.window = self._window()
        self._played = None
        self._expired = False
        return True

    async def update(self, plan_id: str, item_id: int, arguments: dict):
        if error := self._input_error():
            return self.result(False, error)
        if plan_id != self.batch_id:
            return self.result(False, "plan_mismatch")
        old = self.chain.task(item_id)
        if old.status not in {"pending", "queued"}:
            return self.result(False, "operation_unavailable")
        if old.batch_id != self.batch_id or self.chain.dependents(item_id):
            return self._replan(item_id, arguments)
        assert self.current_turn is not None
        op = self.gateway.register(self.current_turn, old.name, arguments)
        try:
            validate_arguments(op)
            new = self.chain.replace(item_id, op)
        except Exception:
            self.gateway.cancel(op.id)
            raise
        self.gateway.cancel(old.operation_id)
        self._played = None
        return self.result(True, "updated", (op,))

    def _replan(self, task_id: int, arguments: dict):
        """修改旧批次中未开始的任务时，把它和未执行的依赖者移到收集批次。"""
        assert self.current_turn is not None
        ids, todo = {task_id}, [task_id]
        while todo:
            for child in self.chain.dependents(todo.pop()):
                if child.task_id not in ids:
                    ids.add(child.task_id)
                    todo.append(child.task_id)
        originals = [self.chain.task(key) for key in sorted(ids)]
        if any(task.status not in {"pending", "queued"} or
               (op := self.gateway.store.get(task.operation_id)) is None or
               op.status not in {"pending", "approved"} for task in originals):
            return self.result(False, "operation_unavailable")
        for task in originals:
            if any(self.chain.task(key).status in {"failed", "unknown", "cancelled", "skipped"}
                   for key in task.depends_on if key not in ids):
                return self.result(False, "dependency_unavailable")
        staged = {}
        try:
            for task in originals:
                op = self.gateway.register(self.current_turn, task.name,
                                           arguments if task.task_id == task_id else task.arguments)
                staged[task.task_id] = op
                validate_arguments(op)
        except Exception:
            for op in staged.values():
                self.gateway.cancel(op.id)
            raise
        # 从最后检查到取消、重新登记之间没有 await，正在执行的任务不参与迁移。
        for task in originals:
            self.executor.cancel(task.task_id)
        remap: dict[int, int] = {}
        remaining = list(originals)
        while remaining:
            task = next(task for task in remaining if all(key not in ids or key in remap for key in task.depends_on))
            new = self.chain.add(self.batch_id, staged[task.task_id],
                                 depends_on=tuple(remap.get(key, key) for key in task.depends_on),
                                 replaces_task_id=task.task_id, compensates_task_id=task.compensates_task_id)
            remap[task.task_id] = new.task_id
            remaining.remove(task)
        self._played = None
        return self.result(True, "replanned", staged.values())

    def remove(self, plan_id: str, item_id: int):
        if error := self._input_error():
            return self.result(False, error)
        if plan_id != self.batch_id:
            return self.result(False, "plan_mismatch")
        task = self.chain.task(item_id)
        if task.batch_id != self.batch_id or task.status != "pending":
            return self.result(False, "operation_unavailable")
        self.executor.cancel(item_id)
        self._played = None
        return self.result(True, "removed")

    def cancel_plan(self, plan_id: str):
        if error := self._input_error():
            return self.result(False, error)
        if plan_id != self.batch_id:
            return self.result(False, "plan_mismatch")
        for task in self.plan.items:
            self.executor.cancel(task.task_id)
        self._played = None
        self._force_confirm.discard(self.batch_id)
        self.window.suspend()
        return self.result(True, "cancelled")

    def depend(self, task_id: int, depends_on: list[int]):
        if error := self._input_error():
            return self.result(False, error)
        self.chain.set_dependencies(task_id, tuple(depends_on))
        self._played = None
        return self.result(True, "dependencies_updated")

    async def verify_operation(self, operation_id: str):
        op = self.gateway.store.get(operation_id)
        if op is None:
            return self.result(False, "operation_missing")
        if op.status != "unknown":
            return self.result(True, "known_result", (op,))
        if op.name == "add_items":
            name, args = "find_created_items", dict(op.arguments, operation_id=op.id)
        elif op.name == "create_reminder":
            name, args = "inspect_reminder", {"operation_id": op.id}
        elif op.name == "delete_task":
            name, args = "inspect_task", op.arguments
        elif op.name == "delete_reminder":
            name, args = "inspect_reminder", op.arguments
        else:
            return self.result(False, "verification_requires_manual_query", (op,))
        query = await self.gateway.submit("verify:" + op.id, name, args)
        data = query.result.get("structured_content", {}) if isinstance(query.result, dict) else {}
        success = False
        if query.status == "succeeded":
            if op.name == "add_items":
                success = data.get("status") == "succeeded"
            elif op.name == "create_reminder" and data.get("present") is True:
                start = data.get("start", {}).get("dateTime", "")
                try:
                    local = datetime.fromisoformat(start).astimezone(ZoneInfo(self._timezone_name)).replace(tzinfo=None)
                    success = local == datetime.fromisoformat(op.arguments["local_time"]) and data.get("title") == op.arguments["title"].strip()
                except (ValueError, TypeError):
                    success = False
            elif op.name.startswith("delete_"):
                success = data.get("present") is False
        value = {"text": "查询已证实操作生效" if success else "查询尚不能确定原操作结果，不要重试",
                 "structured_content": data, "previous_result": op.result}
        if success:
            if op.name == "create_reminder":
                value["structured_content"] = dict(data, version=1, status="succeeded")
        updated = self.gateway.store.record_verification(op.id, value, succeeded=success)
        if success:
            for task in self.chain.all_tasks:
                if task.operation_id == op.id:
                    self.chain.observe_operation(task.task_id, updated)
        return self.result(success, "verified" if success else "still_unknown", (updated,))

    async def undo(self, task_id: int, task_ids: list[str] | None = None):
        if error := self._input_error():
            return self.result(False, error)
        task = self.chain.task(task_id)
        # 固定边界：只允许指向当前已经出现的具体编号，不按标题波及未来任务。
        if task.status in {"pending", "queued"}:
            self.executor.cancel(task_id)
            self._played = None
            return self.result(True, "cancelled_unsent")
        if task.status == "executing":
            self._undo_after.add(task_id)
            return self.result(True, "waiting_for_outcome")
        turn = self.current_turn
        op = self.gateway.store.get(task.operation_id)
        if op is None:
            return self.result(False, "operation_missing")
        if op.status == "unknown":
            checked = await self.verify_operation(op.id)
            if not checked.ok:
                return checked
            op = self.gateway.store.get(op.id)
            assert op is not None
            if self.current_turn != turn or self._input_error(turn):
                return self.result(False, "turn_changed")
        return self._stage_undo(task_id, op, task_ids)

    def _stage_undo(self, task_id: int, op: Operation, task_ids=None):
        if op.status != "succeeded":
            return self.result(False, "operation_not_succeeded", (op,))
        if op.name == "add_items":
            requests = [(request.name, request.arguments) for request in undo_add_items(op, task_ids=task_ids)]
        elif op.name == "create_reminder":
            data = op.result.get("structured_content", {}) if isinstance(op.result, dict) else {}
            if not data.get("event_id"):
                return self.result(False, "missing_receipt")
            requests = [("delete_reminder", {"event_id": data["event_id"]})]
        else:
            return self.result(False, "undo_not_supported")
        registered = []
        for name, args in requests:
            key = (task_id, json.dumps(args, sort_keys=True))
            if key in self._compensations:
                previous = self.gateway.store.get(self._compensations[key])
                if previous is not None and previous.status not in {"cancelled", "failed"}:
                    continue
            new = self.gateway.register(self.current_turn or "deferred-undo", name, args)
            try:
                # 核实后的真实成功可能发生在原先的 unknown 内存状态。
                self.chain.observe_operation(task_id, op)
                self.chain.add(self.batch_id, new, compensates_task_id=task_id)
            except Exception:
                self.gateway.cancel(new.id)
                raise
            self._compensations[key] = new.id
            registered.append(new)
        self._played = None
        return self.result(True, "undo_pending_confirmation", registered)

    def on_reconnect(self):
        self._played = None
        self._confirmation_request = None
        self.current_turn = None
        self.window.suspend()
        if self.plan.items:
            self.chain.set_collection_status(self.batch_id, "collecting")
            self._force_confirm.add(self.batch_id)
            self.notices.put_nowait(TaskNotice("recap", self.batch_id, self.plan.version))

    async def tick(self):
        for window in self._retired:
            await window.close()
        self._retired.clear()
        self.expire()
        while not self.window.events.empty():
            event = self.window.events.get_nowait()
            if not self.window.is_current(event):
                continue
            if event.kind == "ready" and self.batch_id not in self._force_confirm:
                self._submit()
            elif event.kind in {"ready", "confirmation_required"}:
                self.window.suspend()
                self.notices.put_nowait(TaskNotice("recap", self.batch_id, self.plan.version, turn_id=event.turn_id))
            elif event.kind == "clarification_required":
                self.notices.put_nowait(TaskNotice("clarify", self.batch_id, self.plan.version, event.question, event.turn_id))
        while not self.executor.reports.empty():
            report = self.executor.reports.get_nowait()
            if self._jobs is not None:
                self._jobs.create_task(self._forward_report(report), name="verify-and-report-batch")
            else:
                await self._forward_report(report)

    async def _forward_report(self, report):
        # 核实可能耗时，单独运行，不阻塞收集窗口；报告仍按批次顺序输出。
        async with self._report_lock:
            operations, lines = [], []
            for item in report.tasks:
                outcome = item.operation
                if outcome.status == "unknown" and (self.auto_verify or item.task.task_id in self._undo_after):
                    try:
                        await self.verify_operation(outcome.id)
                    except Exception:
                        pass  # 查询失败不是原写入失败，继续如实报告 unknown。
                    outcome = self.gateway.store.get(outcome.id) or outcome
                operations.append(outcome)
                text = item.text if item.task.status == "skipped" else format_for_model(outcome)["text"]
                lines.append(f"#{item.task.task_id} {item.task.name} [{outcome.status}]：{text}")
                if item.task.task_id in self._undo_after:
                    self._undo_after.remove(item.task.task_id)
                    if outcome.status == "succeeded":
                        try:
                            staged = self._stage_undo(item.task.task_id, outcome)
                            if not staged.ok:
                                raise ValueError(staged.reason)
                        except (ValueError, TypeError) as exc:
                            self.notices.put_nowait(TaskNotice("clarify", self.batch_id, self.plan.version,
                                                              "原操作完成了，但缺少可用撤销回执：" + str(exc)))
                        else:
                            self.notices.put_nowait(TaskNotice("recap", self.batch_id, self.plan.version))
                    else:
                        self.notices.put_nowait(TaskNotice("clarify", self.batch_id, self.plan.version,
                                                          "刚才的操作结果未确定，暂时不能撤销，需要先核实。"))
            self.reports.put_nowait(CommitReport(tuple(operations), "\n".join(lines)))

    async def run(self):
        self._runner = asyncio.current_task()
        async def watch():
            while True:
                await self.tick()
                await asyncio.sleep(.02)
        try:
            async with asyncio.TaskGroup() as tasks:
                self._jobs = tasks
                tasks.create_task(self.executor.run())
                tasks.create_task(watch())
        finally:
            self._runner = None
            self._jobs = None

    async def close(self):
        self._closed = True
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
        await self.executor.close()
        await self.window.close()
        for window in self._retired:
            await window.close()
        for task in self.chain.all_tasks:
            if task.status in {"pending", "queued"}:
                self.executor.cancel(task.task_id)
