"""SQLite 操作记录：记录事实和状态，不负责计划、工具执行或自动重试。"""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Literal, cast
from uuid import uuid4


OperationStatus = Literal[
    "pending", "approved", "cancelled", "executing", "succeeded", "failed", "unknown",
]

_TRANSITIONS: dict[OperationStatus, frozenset[OperationStatus]] = {
    "pending": frozenset({"approved", "cancelled"}),
    "approved": frozenset({"executing", "cancelled"}),
    "executing": frozenset({"succeeded", "failed", "unknown"}),
}


@dataclass(frozen=True)
class Operation:
    id: str
    turn_id: str
    name: str
    arguments: dict[str, Any]
    status: OperationStatus
    result: object | None
    created_at: float
    updated_at: float


def _arguments_json(arguments: dict[str, Any]) -> str:
    # JSON 对象的键顺序不影响含义，递归排序对象键；列表顺序、重复元素和字符串内容保留。
    # 参数应来自 JSON，拒绝 NaN/Infinity，避免保存无法可靠比较的非标准 JSON 值。
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class OperationStore:
    """一个实例持有一个同步 SQLite 连接，应在创建它的线程使用，结束时调用 close。"""

    def __init__(self, db_path: str | Path):
        self._db = sqlite3.connect(str(db_path))
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'pending', 'approved', 'cancelled', 'executing',
                        'succeeded', 'failed', 'unknown'
                    )),
                    result TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS operations_recent
                ON operations (name, status, updated_at)
            """)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Operation:
        return Operation(
            id=row["id"], turn_id=row["turn_id"], name=row["name"],
            arguments=json.loads(row["arguments"]),
            status=cast(OperationStatus, row["status"]),
            result=json.loads(row["result"]) if row["result"] is not None else None,
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def create(self, turn_id: str, name: str, arguments: dict[str, Any]) -> Operation:
        """保存一个新的 pending 操作；时间使用可持久化的 Unix 时间戳（秒）。"""
        op_id, now = uuid4().hex, time.time()
        encoded = _arguments_json(arguments)
        with self._db:
            self._db.execute("""
                INSERT INTO operations
                    (id, turn_id, name, arguments, status, result, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', NULL, ?, ?)
            """, (op_id, turn_id, name, encoded, now, now))
        # 反序列化一份快照，之后修改调用方的参数字典，不会改变记录的含义。
        return Operation(op_id, turn_id, name, json.loads(encoded), "pending", None, now, now)

    def transition(
        self, op_id: str, from_status: OperationStatus, to_status: OperationStatus,
        result: object | None = None,
    ) -> bool:
        """条件更新状态；非法或不匹配时返回 False。result=None 表示保留旧结果，不表示清空。"""
        if to_status not in _TRANSITIONS.get(from_status, frozenset()):
            return False
        encoded_result = json.dumps(result, ensure_ascii=False, allow_nan=False) if result is not None else None
        with self._db:
            # 检查与状态修改合在一条 SQL 里，避免两个调用都先读到 approved，再各执行一次。
            cursor = self._db.execute("""
                UPDATE operations SET status = ?, result = COALESCE(?, result), updated_at = ?
                WHERE id = ? AND status = ?
            """, (to_status, encoded_result, time.time(), op_id, from_status))
            return cursor.rowcount == 1

    def get(self, op_id: str) -> Operation | None:
        row = self._db.execute("SELECT * FROM operations WHERE id = ?", (op_id,)).fetchone()
        return self._from_row(row) if row is not None else None

    def recent_succeeded(
        self, name: str, arguments: dict[str, Any], within_s: float = 600,
    ) -> Operation | None:
        """找时间窗口内最近成功的相同操作；只是查询记录，不自动拦截或重放操作。"""
        if within_s < 0:
            raise ValueError("within_s 不能为负数")
        row = self._db.execute("""
            SELECT * FROM operations
            WHERE name = ? AND arguments = ? AND status = 'succeeded' AND updated_at >= ?
            ORDER BY updated_at DESC, rowid DESC LIMIT 1
        """, (name, _arguments_json(arguments), time.time() - within_s)).fetchone()
        return self._from_row(row) if row is not None else None

    def recover_on_startup(self) -> list[Operation]:
        """应用启动时显式调用：遗留 executing 改为 unknown，返回更新后的记录，不自动重试。"""
        now = time.time()
        with self._db:
            # 先拿写锁，再读取并修改同一批记录；返回值与本次真正修改的记录保持一致。
            self._db.execute("BEGIN IMMEDIATE")
            rows = self._db.execute("SELECT * FROM operations WHERE status = 'executing' ORDER BY rowid").fetchall()
            self._db.execute("""
                UPDATE operations SET status = 'unknown', updated_at = ? WHERE status = 'executing'
            """, (now,))
            return [replace(self._from_row(row), status="unknown", updated_at=now) for row in rows]

    def close(self) -> None:
        self._db.close()

    def discard_unsubmitted(self) -> list[Operation]:
        """程序重启不恢复旧计划；明确取消旧进程遗留的未发送操作。"""
        with self._db:
            rows = self._db.execute("SELECT * FROM operations WHERE status IN ('pending','approved')").fetchall()
            self._db.execute("UPDATE operations SET status='cancelled', updated_at=? WHERE status IN ('pending','approved')", (time.time(),))
        return [op for row in rows if (op := self.get(row["id"])) is not None]

    def unresolved(self) -> list[Operation]:
        rows = self._db.execute("SELECT * FROM operations WHERE status='unknown' ORDER BY updated_at DESC LIMIT 20").fetchall()
        return [self._from_row(row) for row in rows]

    def record_verification(self, op_id: str, result: object, *, succeeded: bool = False) -> Operation:
        """仅由可信查询结果调用；没有查到资源不能证明原请求未执行。"""
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        with self._db:
            self._db.execute("UPDATE operations SET status=?, result=?, updated_at=? WHERE id=? AND status='unknown'",
                             ("succeeded" if succeeded else "unknown", encoded, time.time(), op_id))
        op = self.get(op_id)
        if op is None:
            raise KeyError(op_id)
        return op
