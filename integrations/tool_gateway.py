"""所有工具实际执行前的状态边界；审批资格由调用它的协调层决定。"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from integrations.operations import Operation, OperationStatus, OperationStore
from integrations.tool_policy import get_policy


class ToolNotSent(RuntimeError):
    """executor 确定请求尚未发出时使用；发出后超时不能使用这个异常。"""


@dataclass(frozen=True)
class ExecutorResult:
    """executor 的统一返回值。result 必须能保存成 JSON，ok 必须是 bool。"""

    ok: bool
    result: object | None = None


Executor = Callable[[str, dict[str, Any]], Awaitable[ExecutorResult]]


class ToolGateway:
    """同步方法修改审批状态，异步方法执行工具；实例与 store 在同一事件循环线程使用。"""

    def __init__(self, store: OperationStore, executor: Executor):
        self.store = store
        self.executor = executor

    async def submit(self, turn_id: str, name: str, arguments: dict[str, Any]) -> Operation:
        """创建新操作；immediate 自动批准并执行，deferred/confirm 保留 pending。"""
        policy = get_policy(name)
        if policy is None:
            raise ValueError(f"工具不在策略白名单中：{name}")
        operation = self.store.create(turn_id, name, arguments)
        if policy.commit == "immediate":
            self.approve(operation.id)
            return await self.execute(operation.id)
        # 同参数的新请求仍然是新操作；是否为用户有意重复，由后续计划层决定。
        return operation

    def approve(self, op_id: str) -> bool:
        """只批准 pending 操作；用户确认、计划版本和播放资格由后续协调层核对。"""
        operation = self.store.get(op_id)
        if operation is None or get_policy(operation.name) is None:
            return False
        return self.store.transition(op_id, "pending", "approved")

    def cancel(self, op_id: str) -> bool:
        """只取消 pending/approved；已经开始执行的请求不能假装已被撤销。"""
        operation = self.store.get(op_id)
        if operation is None or operation.status not in {"pending", "approved"}:
            return False
        return self.store.transition(op_id, operation.status, "cancelled")

    async def execute(self, op_id: str) -> Operation:
        """只执行 approved；重复或状态不符时返回当前记录，不重放工具。"""
        operation = self._get(op_id)
        if get_policy(operation.name) is None:
            raise ValueError(f"工具不在策略白名单中：{operation.name}")
        # 从读取到条件登记执行中，没有 await；数据库中的 WHERE status 再保证只认领一次。
        # 同一 ID 的另一个 execute 会看到 executing/终态，不会再次调用 executor。
        if not self.store.transition(op_id, "approved", "executing"):
            return self._get(op_id)

        try:
            response = await self.executor(operation.name, operation.arguments)
            if not isinstance(response, ExecutorResult) or type(response.ok) is not bool:
                raise TypeError("executor 必须返回 ExecutorResult，且 ok 必须是 bool")
            status: OperationStatus = "succeeded" if response.ok else "failed"
            # 把结果保存也放在 try 里：远端成功但结果无法编码时，不能把它当作确定失败。
            self.store.transition(op_id, "executing", status, response.result)
        except ToolNotSent as exc:
            self.store.transition(op_id, "executing", "failed", self._error(exc))
        except Exception as exc:
            self.store.transition(op_id, "executing", "unknown", self._error(exc))
        except BaseException as exc:
            # 包括任务取消：远端可能已经收到请求，记 unknown，再保留取消/退出的传播行为。
            self.store.transition(op_id, "executing", "unknown", self._error(exc))
            raise
        return self._get(op_id)

    def _get(self, op_id: str) -> Operation:
        operation = self.store.get(op_id)
        if operation is None:
            raise KeyError(f"操作不存在：{op_id}")
        return operation

    @staticmethod
    def _error(exc: BaseException) -> dict[str, str]:
        return {"error": f"{type(exc).__name__}: {exc}"}
