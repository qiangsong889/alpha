"""计划的确认规则；由调用方提供播放和用户事件，不依赖 WebSocket 或音频。"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
import time
from typing import Any

from core.plan import Plan
from integrations.operations import Operation
from integrations.tool_gateway import ToolGateway


@dataclass(frozen=True)
class CoordinatorResult:
    # ok 表示控制请求被接受，不能据此声称每一项工具都成功。
    ok: bool
    reason: str
    operations: tuple[Operation, ...] = ()
    version: int | None = None
    snapshot: dict[str, Any] | None = None


@dataclass
class _UserTurn:
    id: str
    received_at: float
    sequence: int
    used: bool = False
    answered: bool = False


class PlanCoordinator:
    def __init__(self, gateway: ToolGateway, clock: Callable[[], float] = time.monotonic,
                 *, timezone_name: str = "America/Los_Angeles"):
        self.gateway, self.clock = gateway, clock
        self.plan: Plan | None = None
        self._timezone_name = timezone_name
        self._sequence = 0
        self._turn: _UserTurn | None = None
        self._seen_turns: dict[str, _UserTurn] = {}
        self._played_at: float | None = None
        self._played_version: int | None = None
        self._played_sequence: int | None = None
        self._expired = False
        # 修改可以并行处理不同编号；计数仅阻止 collect、复述完成和确认中途插入。
        self._editing = 0

    def _result(self, ok: bool, reason: str,
                operations: tuple[Operation, ...] = ()) -> CoordinatorResult:
        # 控制工具成功或失败都携带当前快照，模型才能看到最新版本与固定项编号。
        return CoordinatorResult(ok, reason, operations,
                                 self.plan.version if self.plan is not None else None,
                                 self.snapshot())

    def collect(self, op: Operation) -> CoordinatorResult:
        """busy 时没有加入计划；调用方必须取消刚登记的 op，并如实告诉模型。"""
        if self._editing or (self.plan is not None and self.plan.status == "executing"):
            return self._result(False, "busy")
        current = self.gateway.store.get(op.id)
        if current != op:
            return self._result(False, "invalid_operation")
        plan = self.plan
        if plan is None or plan.status in {"done", "cancelled"}:
            plan = Plan(timezone_name=self._timezone_name)
        try:
            added = plan.add(op)
        except (ValueError, TypeError):
            return self._result(False, "invalid_arguments")
        if not added:
            return self._result(False, "duplicate_operation")
        self.plan = plan
        self._invalidate_recap()
        return self._result(True, "collected")

    def _identity_error(self, plan_id: str) -> str | None:
        if self.plan is None:
            return "no_plan"
        if plan_id != self.plan.id:
            return "plan_mismatch"
        return None

    def _version_error(self, plan_id: str, version: int) -> str | None:
        error = self._identity_error(plan_id)
        if error:
            return error
        assert self.plan is not None
        return "version_mismatch" if type(version) is not int or version != self.plan.version else None

    def _invalidate_recap(self) -> None:
        self._played_at = self._played_version = self._played_sequence = None
        self._expired = False

    def recap_played(self, plan_id: str, version: int, interrupted: bool) -> CoordinatorResult:
        error = self._version_error(plan_id, version)
        if error:
            return self._result(False, error)
        assert self.plan is not None
        if self._editing:
            return self._result(False, "busy")
        if self.plan.status not in {"collecting", "awaiting_confirmation"}:
            return self._result(False, "invalid_state")
        if not self.plan.items:
            return self._result(False, "empty_plan")
        if interrupted:
            self.plan.status = "collecting"
            self._invalidate_recap()
            return self._result(False, "not_played")
        # 同一个完成通知重复到达，不能延长有效期或抹掉已经收到的用户话语。
        if self._played_version == version and self.plan.status == "awaiting_confirmation":
            return self._result(True, "already_played")
        self._sequence += 1
        self._played_sequence = self._sequence
        self._played_at = self.clock()
        self._played_version = version
        self._expired = False
        self.plan.status = "awaiting_confirmation"
        return self._result(True, "played")

    def user_turn(self, turn_id: str) -> CoordinatorResult:
        """调用方只传有意义的话；语义是否同意仍由模型判断，这里不匹配‘对/好’。"""
        if not isinstance(turn_id, str) or not turn_id.strip():
            return self._result(False, "invalid_turn")
        if turn_id in self._seen_turns:
            reason = "turn_used" if self._seen_turns[turn_id].used else "duplicate_turn"
            return self._result(False, reason)
        self.expire()
        self._sequence += 1
        self._turn = _UserTurn(turn_id, self.clock(), self._sequence)
        self._seen_turns[turn_id] = self._turn
        return self._result(True, "turn_received")

    def turn_answered(self, turn_id: str) -> None:
        turn = self._seen_turns.get(turn_id)
        if turn is not None:
            turn.answered = True

    def expire(self) -> bool:
        if (self.plan is not None and self.plan.status == "awaiting_confirmation"
                and self._played_at is not None and self.clock() - self._played_at >= 120):
            self.plan.status = "collecting"
            self._invalidate_recap()
            self._expired = True
            return True
        return False

    def _control_error(self, plan_id: str, *, require_played: bool) -> str | None:
        error = self._identity_error(plan_id)
        if error:
            return error
        assert self.plan is not None
        if require_played and self._editing:
            return "busy"
        self.expire()
        if require_played and self._expired:
            return "expired"
        # 保留消耗记录，确认执行完后同一句话再次确认仍明确返回 turn_used。
        if self._turn is not None and self._turn.used:
            return "turn_used"
        if self.plan.status not in {"collecting", "awaiting_confirmation"}:
            return "invalid_state"
        if require_played:
            if (self.plan.status != "awaiting_confirmation" or self._played_version != self.plan.version
                    or self._played_at is None or self._played_sequence is None):
                return "not_played"
        if self._turn is None or self._turn.answered:
            return "no_user_turn"
        if require_played:
            assert self._played_at is not None and self._played_sequence is not None
            # 相同的 clock 数值也可以有先后事件；sequence 不依赖时钟分辨率。
            if (self._turn.sequence <= self._played_sequence
                    or self._turn.received_at < self._played_at):
                return "no_user_turn"
        return None

    def _operations(self) -> tuple[Operation, ...]:
        assert self.plan is not None
        return tuple(deepcopy(op) for item in self.plan.items
                     if (op := self.gateway.store.get(item.op_id)) is not None)

    async def confirm(self, plan_id: str, version: int,
                      duplicates_intended: bool = False) -> CoordinatorResult:
        error = self._version_error(plan_id, version) or self._control_error(plan_id, require_played=True)
        if error:
            return self._result(False, error)
        assert self.plan is not None and self._turn is not None
        if not self.plan.items:
            return self._result(False, "empty_plan")
        if type(duplicates_intended) is not bool:
            return self._result(False, "invalid_arguments")
        if self.plan.has_duplicates and not duplicates_intended:
            return self._result(False, "quantity_ambiguous")
        if len(self._operations()) != len(self.plan.items):
            return self._result(False, "operation_missing")
        # 检查、消耗输入、登记 executing 之间没有 await，第二次确认不能插进来。
        self._turn.used = True
        self.plan.status = "executing"
        try:
            for item in self.plan.items:
                if self.gateway.approve(item.op_id):
                    await self.gateway.execute(item.op_id)
                # approve=False 不重放；报告会读取 cancelled/executing 等真实状态。
        finally:
            # 任务取消时，gateway 将已发出的请求记为 unknown；尚未执行的项明确取消。
            for item in self.plan.items:
                op = self.gateway.store.get(item.op_id)
                if op is not None and op.status in {"pending", "approved"}:
                    self.gateway.cancel(op.id)
            # done 只表示流程结束，不表示每项成功。
            self.plan.status = "done"
        return self._result(True, "completed", self._operations())

    def _item_index(self, item_id: int) -> int | None:
        assert self.plan is not None
        if type(item_id) is not int:
            return None
        return next((i for i, item in enumerate(self.plan.items) if item.item_id == item_id), None)

    async def update(self, plan_id: str, item_id: int,
                     arguments: dict[str, Any]) -> CoordinatorResult:
        """按固定编号修改；同一句话可修改多处，不能据此获得确认资格。"""
        error = self._control_error(plan_id, require_played=False)
        if error:
            return self._result(False, error)
        assert self.plan is not None and self._turn is not None
        index = self._item_index(item_id)
        if index is None:
            return self._result(False, "operation_unavailable")
        old = self.gateway.store.get(self.plan.items[index].op_id)
        if old is None or old.status not in {"pending", "approved"}:
            return self._result(False, "operation_unavailable")
        if not isinstance(arguments, dict):
            return self._result(False, "invalid_arguments")
        try:
            candidate = replace(old, status="pending", arguments=deepcopy(arguments))
            deepcopy(self.plan).replace(index, candidate)
        except (ValueError, TypeError):
            return self._result(False, "invalid_arguments")
        turn = self._turn
        self._editing += 1
        try:
            try:
                new = await self.gateway.submit(turn.id, old.name, candidate.arguments)
            except Exception:
                return self._result(False, "registration_failed")
            if self._turn is not turn or turn.answered or turn.used:
                self.gateway.cancel(new.id)
                return self._result(False, "turn_changed")
            error = self._control_error(plan_id, require_played=False)
            if error:
                self.gateway.cancel(new.id)
                return self._result(False, error)
            # 别的修改可能删了前面的项目；按固定编号重新找位置，不比较整个计划的版本。
            index = self._item_index(item_id)
            if index is None or self.plan.items[index].op_id != old.id:
                self.gateway.cancel(new.id)
                return self._result(False, "operation_unavailable")
            if not self.gateway.cancel(old.id):
                self.gateway.cancel(new.id)
                return self._result(False, "operation_unavailable")
            self.plan.replace(index, new)
            # 修改不消耗这句话；复述作废已经足以阻止把修改当成确认。
            self._invalidate_recap()
            return self._result(True, "updated", (deepcopy(new),))
        finally:
            self._editing -= 1

    def remove(self, plan_id: str, item_id: int) -> CoordinatorResult:
        error = self._control_error(plan_id, require_played=False)
        if error:
            return self._result(False, error)
        assert self.plan is not None
        index = self._item_index(item_id)
        if index is None:
            return self._result(False, "operation_unavailable")
        op_id = self.plan.items[index].op_id
        if not self.gateway.cancel(op_id):
            return self._result(False, "operation_unavailable")
        self.plan.remove(index)
        self._invalidate_recap()
        return self._result(True, "removed")

    def cancel_plan(self, plan_id: str) -> CoordinatorResult:
        error = self._control_error(plan_id, require_played=False)
        if error:
            return self._result(False, error)
        assert self.plan is not None
        operations = self._operations()
        if len(operations) != len(self.plan.items) or any(
            op.status not in {"pending", "approved", "cancelled"} for op in operations
        ):
            return self._result(False, "operation_unavailable")
        for op in operations:
            if op.status != "cancelled" and not self.gateway.cancel(op.id):
                self.plan.status = "collecting"
                self._invalidate_recap()
                return self._result(False, "cancel_failed", self._operations())
        self.plan.status = "cancelled"
        self._invalidate_recap()
        return self._result(True, "cancelled", self._operations())

    def on_reconnect(self) -> None:
        """保留内容和执行进度，清空确认资格；语音层还须丢弃旧连接的播放回调。"""
        self._turn = None
        self._invalidate_recap()
        if self.plan is not None and self.plan.status == "awaiting_confirmation":
            self.plan.status = "collecting"

    def snapshot(self) -> dict[str, Any] | None:
        if self.plan is None:
            return None
        items = []
        for item in self.plan.items:
            op = self.gateway.store.get(item.op_id)
            items.append({
                "item_id": item.item_id, "op_id": item.op_id, "name": item.name, "arguments": item.arguments,
                "status": op.status if op is not None else None,
                "result": deepcopy(op.result) if op is not None else None,
            })
        return {"id": self.plan.id, "version": self.plan.version,
                "status": self.plan.status, "items": items}
