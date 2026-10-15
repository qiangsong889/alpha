"""日常任务协调：设备立即执行、普通写入短暂等待、自动分批执行、精确删除和恢复。"""

import asyncio
from dataclasses import dataclass
from collections import deque
import re
import time
from typing import Any

from core.plan import validate_arguments
from core.task_compensation import created_tasks
from integrations.operations import Operation
from integrations.tool_executor import HA_ACTION_TOOLS, format_for_model
from integrations.tool_gateway import ToolGateway
from integrations.tool_policy import get_policy
from integrations.verification import verify_operation
from voice.commit_window import CommitReport


@dataclass(frozen=True)
class Clarification:
    turn_id: str
    generation: int
    text: str


class SimpleCoordinator:
    # 兼容音频基类的只读接口；主流程不再创建任何计划或复述。
    plan = None

    def __init__(self, gateway: ToolGateway, *, timezone_name="America/Los_Angeles",
                 window_s=2.0, incomplete_s=5.0, clock=time.monotonic):
        import math
        if not all(math.isfinite(value) and value >= 0 for value in (window_s, incomplete_s)) or incomplete_s < window_s:
            raise ValueError("等待时间必须有限非负，未说完的等待不能短于普通窗口")
        self.gateway, self._timezone_name, self.clock = gateway, timezone_name, clock
        self.window_s, self.incomplete_s = window_s, incomplete_s
        self.current_turn: str | None = None
        self._speech_id: str | None = None
        self._speaking = False
        self._stopped_at: float | None = None
        self._ready = False
        self._incomplete = False
        self._execute_now = False
        self._previous_input: tuple[Any, ...] | None = None
        self._texts: dict[str, str] = {}
        self._sequence = 0
        self._turn_sequence = 0
        self._started_sequences: dict[str, int] = {}
        self._generation = 0
        self._timer: asyncio.Task | None = None
        self._execution: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._runner: asyncio.Task | None = None
        self._pending: dict[str, None] = {}
        self._created_sources: list[str] = []
        self._mutations: list[str] = []
        self._deletions: dict[tuple[str, str, str], str] = {}
        self._batches: deque[tuple[str, ...]] = deque()
        self._queued: set[str] = set()
        self._report_pending = False
        self._deleted_sources: list[str] = []
        self._restorations: dict[str, str] = {}
        self._closed = False
        self._paused_for_reconnect = False
        self.reports: asyncio.Queue[CommitReport] = asyncio.Queue()
        self.notices: asyncio.Queue[Clarification] = asyncio.Queue()

    def _spawn(self, coro, name):
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _invalidate(self):
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def speech_started(self, turn_id=None):
        if turn_id is None or turn_id == self._speech_id:
            return
        self._previous_input = (self._speech_id, self._speaking, self._stopped_at, self._ready)
        self._speech_id = turn_id
        self._sequence += 1
        self._started_sequences[turn_id] = self._sequence
        self._speaking, self._ready = True, False
        self._stopped_at = None
        self._invalidate()

    def speech_stopped(self, turn_id):
        if turn_id == self._speech_id and self._speaking:
            self._speaking = False
            self._stopped_at = self.clock()
            self._arm()

    def speech_accepted(self):
        pass  # speech_started 已暂停；工具和 response.done 完成后由代码放行。

    def speech_ignored(self):
        if self._previous_input is not None:
            ready_during_noise = self._ready
            self._speech_id, self._speaking, self._stopped_at, self._ready = self._previous_input
            if self.current_turn == self._speech_id:
                self._ready |= ready_during_noise
            self._previous_input = None
            if self._ready:
                self._stopped_at = self.clock()
            self._arm()
            self._start_next()

    def record_text(self, turn_id, text):
        self._texts[turn_id] = text

    def user_text(self, turn_id) -> str:
        return self._texts.get(turn_id, "")

    def user_turn(self, turn_id):
        if turn_id == self.current_turn:
            return
        self.current_turn = turn_id
        if turn_id not in self._started_sequences:
            self._sequence += 1
            self._started_sequences[turn_id] = self._sequence
        self._turn_sequence = self._started_sequences[turn_id]
        text = "".join(char for char in self._texts.get(turn_id, "") if char.isalnum())
        self._incomplete = bool(re.search(r"(?:等等|等一下|稍等|还有|然后|或者|跟|和|把|给|再)$", text))
        self._execute_now = text in {"执行吧", "现在执行", "开始执行", "可以执行了",
                                     "继续", "继续吧", "继续执行", "好继续", "好的继续"}
        if self._execute_now:
            self._paused_for_reconnect = False

    def turn_answered(self, turn_id):
        pass  # 执行无需用户再次同意。

    def turn_finished(self, turn_id):
        if turn_id == self.current_turn:
            self._ready = True
            self._arm()
            self._start_next()

    @property
    def has_pending(self):
        return bool(self._pending)

    @property
    def executing(self):
        return self._execution is not None

    def snapshot(self):
        def summary(op):
            args = {key: value for key, value in op.arguments.items() if key != "resource"}
            resource = op.arguments.get("resource")
            if isinstance(resource, dict):
                args["resource_title"] = resource.get("title", resource.get("summary", ""))
            item = {"id": op.id, "name": op.name, "arguments": args, "status": op.status,
                    "text": format_for_model(op)["text"][:200]}
            data = op.result.get("structured_content", {}) if isinstance(op.result, dict) else {}
            receipt = {key: data[key] for key in ("list_id", "event_id", "task_id", "title") if key in data}
            if isinstance(data.get("created"), list):
                receipt["created"] = data["created"][:5]
                receipt["created_count"] = len(data["created"])
            if receipt:
                item["receipt"] = receipt
            return item
        return {
            "pending": [summary(op) for key in self._pending if (op := self.gateway.store.get(key))],
            "recent_created": [summary(op) for key in self._created_sources[-3:]
                               if (op := self.gateway.store.get(key))],
            "batches": [list(batch) for batch in self._batches],
            "recent_deletions": [summary(op) for key in self._deleted_sources[-3:]
                                 if (op := self.gateway.store.get(key))],
            "paused_after_reconnect": self._paused_for_reconnect,
        }

    async def invoke(self, turn_id, name, arguments):
        if self._closed or turn_id != self.current_turn:
            return {"is_error": True, "text": "旧输入已被取代，本次未执行"}
        policy = get_policy(name)
        if policy is None:
            raise ValueError("工具不在白名单")
        if policy.mutates and name not in HA_ACTION_TOOLS:
            if policy.commit == "confirm":
                raise ValueError("此工具暂不支持自动执行")
            op = self.gateway.register(turn_id, name, arguments)
            try:
                validate_arguments(op)
            except Exception:
                self.gateway.cancel(op.id)
                raise
            self._pending[op.id] = None
            self._mutations.append(op.id)
            if name in {"add_items", "create_reminder"}:
                self._created_sources.append(op.id)
            self._ready = False
            self._paused_for_reconnect = False
            self._invalidate()
        else:
            # HA 的 immediate 不经过两秒窗口，仍由 Gateway 记录真实成功/失败。
            op = await self.gateway.submit(turn_id, name, arguments)
            if policy.mutates:
                self._mutations.append(op.id)
        return self._output(op)

    def _output(self, op):
        result = dict(format_for_model(op), operation_id=op.id, status=op.status)
        if isinstance(op.result, dict) and "structured_content" in op.result:
            result["data"] = op.result["structured_content"]
        return result

    def cancel_pending(self):
        # 同步检查并取消，不让计时器在中间认领这些操作；已发出的请求不在此集合里。
        self._invalidate()
        cancelled = 0
        for key in tuple(self._pending):
            if self.gateway.cancel(key):
                cancelled += 1
            self._pending.pop(key, None)
        self._ready = False
        text = f"已取消 {cancelled} 项尚未执行的操作" if cancelled else "当前没有等待执行的操作"
        if self.executing:
            text += "；另有操作已开始执行，无法保证取消，确定结果后才能撤销"
        return {"is_error": False, "text": text}

    def _arm(self):
        if (self._closed or self._speaking or not self._ready or self._stopped_at is None
                or self._timer is not None or self._paused_for_reconnect
                or (not (self._pending.keys() - self._queued) and not self._incomplete)):
            return
        delay = self.incomplete_s if self._incomplete else (0 if self._execute_now else self.window_s)
        self._timer = self._spawn(self._wait(self._stopped_at + delay, self._generation), "simple-write-window")

    async def _wait(self, deadline, generation):
        try:
            await asyncio.sleep(max(0, deadline - self.clock()))
            if generation != self._generation or self._speaking or self._closed:
                return
            self._timer = None
            if self._incomplete:
                self._ready = False
                self.notices.put_nowait(Clarification(self.current_turn or "", generation, "你还想补充什么？"))
            else:
                batch = tuple(key for key in self._pending if key not in self._queued)
                if batch:
                    self._batches.append(batch)
                    self._queued.update(batch)
                self._start_next()
        except asyncio.CancelledError:
            pass
        finally:
            if self._timer is asyncio.current_task():
                self._timer = None

    async def _verify(self, op):
        if op.status == "unknown":
            try:
                return await verify_operation(self.gateway, op, self._timezone_name)
            except Exception:
                pass
        return op

    def _start_next(self):
        if (self._closed or self._execution is not None or self._report_pending or self._speaking
                or self._paused_for_reconnect or not self._ready or self._incomplete):
            return
        while self._batches:
            batch = self._batches.popleft()
            if any(key in self._pending for key in batch):
                self._execution = self._spawn(self._flush(batch), "simple-write-execution")
                return
            self._queued.difference_update(batch)

    def report_finished(self):
        """当前批次汇报结束，才允许执行已收集好的下一批；不要求用户确认。"""
        self._report_pending = False
        self._start_next()

    async def _prepare_delete(self, op):
        name = "inspect_task" if op.name == "delete_task" else "inspect_reminder"
        query = await self.gateway.submit("backup:" + op.id, name, op.arguments)
        data = query.result.get("structured_content", {}) if isinstance(query.result, dict) else {}
        resource = data.get("resource")
        if query.status != "succeeded" or data.get("present") is not True or not isinstance(resource, dict):
            raise ValueError("无法保存删除前的数据，未执行删除")
        if op.name == "delete_reminder" and (resource.get("recurrence") or resource.get("recurringEventId")
                or resource.get("eventType", "default") != "default" or resource.get("attendees")
                or resource.get("conferenceData") or resource.get("hangoutLink")):
            raise ValueError("暂不支持自动删除重复、特殊或含参与者/会议链接的日程，以免无法恢复")
        self.gateway.store.save_deletion_backup(op.id, data)

    async def _flush(self, batch):
        completed: list[Operation] = []
        try:
            for key in batch:
                if key not in self._pending:
                    self._queued.discard(key)
                    continue
                op = self.gateway.store.get(key)
                if op is None:
                    self._pending.pop(key, None)
                    self._queued.discard(key)
                    continue
                try:
                    if op.name in {"delete_task", "delete_reminder"}:
                        await self._prepare_delete(op)
                except Exception as exc:
                    # 只读备份失败意味着删除尚未发出，不能记成结果未知。
                    if self.gateway.approve(key):
                        self.gateway.store.transition(key, "approved", "executing")
                        self.gateway.store.transition(key, "executing", "failed", {"error": str(exc)})
                        failed = self.gateway.store.get(key)
                        if failed is not None:
                            completed.append(failed)
                    self._pending.pop(key, None)
                    self._queued.discard(key)
                    continue
                # 备份期间仍可取消；重新认领状态，取消了就绝不删除。
                self._pending.pop(key, None)
                self._queued.discard(key)
                if self.gateway.approve(key):
                    op = await self.gateway.execute(key)
                    completed.append(await self._verify(op))
            if completed:
                self._report_pending = True
                self.reports.put_nowait(CommitReport(tuple(completed), "\n".join(
                    f"{op.name} [{op.status}]：{self._report_text(op)}" for op in completed)))
        finally:
            self._execution = None
            self._arm()
            self._start_next()

    def _report_text(self, op):
        if op.status == "succeeded" and op.name in {"delete_task", "delete_reminder"}:
            backup = self.gateway.store.deletion_backup(op.id)
            if backup and backup.get("title"):
                return f"已删除{backup['title']}"
        return format_for_model(op)["text"]

    def is_notice_current(self, notice):
        return notice.turn_id == self.current_turn and notice.generation == self._generation and not self._speaking

    def _owned(self):
        owned = {}
        for key in self._created_sources:
            op = self.gateway.store.get(key)
            if op is None or op.status != "succeeded":
                continue
            if op.name == "add_items":
                try:
                    for item in created_tasks(op):
                        owned[("task", item.list_id, item.task_id)] = item.title
                except ValueError:
                    continue
            elif isinstance(op.result, dict):
                data = op.result.get("structured_content", {})
                if isinstance(data.get("event_id"), str) and data["event_id"]:
                    owned[("reminder", "", data["event_id"])] = op.arguments["title"].strip()
        return owned

    def has_unambiguous_recent_creation(self) -> bool:
        """明确的‘刚才那个不要了’快捷入口，不猜测多个目标。"""
        if not self._created_sources or not self._mutations or self._mutations[-1] != self._created_sources[-1]:
            return False
        source = self.gateway.store.get(self._created_sources[-1])
        if source is None or source.status != "succeeded":
            return False
        if source.name == "add_items":
            try:
                return len(created_tasks(source)) == 1
            except ValueError:
                return False
        data = source.result.get("structured_content", {}) if isinstance(source.result, dict) else {}
        return source.name == "create_reminder" and bool(data.get("event_id"))

    async def undo_recent(self, title=""):
        turn = self.current_turn
        if not turn:
            return {"is_error": True, "text": "没有新的用户输入"}
        source = None
        if not title and self._mutations:
            latest = self.gateway.store.get(self._mutations[-1])
            if latest is not None and latest.name not in {"add_items", "create_reminder"}:
                return {"is_error": True, "text": "刚才不是创建清单条目或提醒，请说明要撤销哪个新增内容"}
        for key in reversed(self._created_sources):
            op = self.gateway.store.get(key)
            if op is None:
                continue
            labels = op.arguments.get("items", [op.arguments.get("title", "")])
            if not title or any(label.strip() == title.strip() for label in labels):
                source = op
                break
        if source is None:
            return {"is_error": True, "text": "本次会话没有找到对应的新增内容；不能撤销以前已有的东西"}
        if source.status in {"pending", "approved"}:
            return {"is_error": True, "text": "这些内容尚未执行，请用 cancel_pending 取消等待操作"}
        if source.status == "executing":
            return {"is_error": True, "text": "创建仍在进行，请等结果确定后再撤销"}
        source = await self._verify(source)
        if turn != self.current_turn:
            return {"is_error": True, "text": "用户已继续说话，尚未执行撤销"}
        if source.status != "succeeded":
            return {"is_error": True, "text": "创建结果未确认，不能猜测 ID 删除，请先核实"}
        owned = self._owned()
        if source.name == "add_items":
            keys = [("task", item.list_id, item.task_id) for item in created_tasks(source)]
        else:
            data = source.result.get("structured_content", {}) if isinstance(source.result, dict) else {}
            keys = [("reminder", "", data.get("event_id", ""))]
        keys = [key for key in keys if key in owned and (not title or owned[key].strip() == title.strip())]
        if not keys:
            return {"is_error": True, "text": "缺少可用的精确创建 ID，未删除任何内容"}
        if not title and len(keys) > 1:
            return {"is_error": True, "text": "刚才新增了多个内容，请说明要撤销哪一个：" + "、".join(owned[key] for key in keys)}
        outputs = [await self._delete(key, owned[key]) for key in keys]
        return {"is_error": any(item["is_error"] for item in outputs), "text": "；".join(item["text"] for item in outputs),
                "status": "pending" if all(item.get("status") == "pending" for item in outputs) else "mixed",
                "operations": outputs}

    async def delete_item(self, kind, item_id, list_id=""):
        if kind not in {"task", "reminder"} or not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("删除目标无效")
        if kind == "task" and (not isinstance(list_id, str) or not list_id.strip()):
            raise ValueError("删除清单条目必须提供精确 list_id")
        if kind == "reminder" and list_id not in {"", "primary"}:
            raise ValueError("目前只支持主日历")
        if not self.current_turn:
            return {"is_error": True, "text": "没有新的用户输入，未执行删除"}
        key = (kind, list_id if kind == "task" else "", item_id)
        return await self._delete(key, self._owned().get(key, "指定内容"))

    async def _delete(self, key, title):
        previous_id = self._deletions.get(key)
        if previous_id is not None:
            previous = self.gateway.store.get(previous_id)
            if previous is not None and previous.status not in {"failed", "cancelled"}:
                return self._output(previous)
        name = "delete_task" if key[0] == "task" else "delete_reminder"
        args = {"list_id": key[1], "task_id": key[2]} if key[0] == "task" else {"event_id": key[2]}
        output = await self.invoke(self.current_turn, name, args)
        if output.get("operation_id"):
            self._deletions[key] = output["operation_id"]
            self._deleted_sources.append(output["operation_id"])
        return output

    async def restore_recent(self, title=""):
        if not self.current_turn:
            return {"is_error": True, "text": "没有新的用户输入"}
        for key in reversed(self._deleted_sources):
            deleted = self.gateway.store.get(key)
            backup = self.gateway.store.deletion_backup(key)
            if deleted is None or not backup or (title and backup.get("title", "").strip() != title.strip()):
                continue
            if deleted.status != "succeeded":
                return {"is_error": True, "text": "删除尚未确认成功，请先核实，不能盲目恢复"}
            prior = self.gateway.store.get(self._restorations.get(key, ""))
            if prior is not None and prior.status not in {"failed", "cancelled"}:
                return self._output(prior)
            args: dict[str, Any] = {"resource": backup["resource"]}
            if deleted.name == "delete_task":
                args["list_id"] = deleted.arguments["list_id"]
            output = await self.invoke(self.current_turn,
                "restore_task" if deleted.name == "delete_task" else "restore_reminder", args)
            self._restorations[key] = output["operation_id"]
            return output
        return {"is_error": True, "text": "本次会话没有对应的已删除内容可恢复"}

    def reconnect_prompt(self) -> str | None:
        if (self._paused_for_reconnect and self._pending
                and not self._speaking and self.current_turn is None):
            return f"刚才断线了，还有 {len(self._pending)} 项操作没执行，要继续吗？"
        return None

    def on_reconnect(self):
        self._invalidate()
        self.current_turn = None
        self._speech_id = None
        self._speaking = False
        self._stopped_at = None
        self._previous_input = None
        self._ready = False
        self._paused_for_reconnect = bool(self._pending)

    async def run(self):
        self._runner = asyncio.current_task()
        try:
            await asyncio.Future()
        finally:
            self._runner = None

    async def close(self):
        self._closed = True
        self._invalidate()
        if self._runner is not None and self._runner is not asyncio.current_task():
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for key in tuple(self._pending):
            self.gateway.cancel(key)
        self._pending.clear()
