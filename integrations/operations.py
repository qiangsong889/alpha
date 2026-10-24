"""SQLite 操作记录：记录事实和状态，不负责计划、工具执行或自动重试。"""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sqlite3
import time
from typing import IO, Any, Literal, cast
from uuid import uuid4
from copy import deepcopy

from integrations.tool_arguments import comparison_arguments


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
        self._db_path = str(db_path)
        self._process_lock: IO[str] | None = None
        self._tool_schemas: dict[str, dict[str, Any]] = {}
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
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS deletion_backups (
                    op_id TEXT PRIMARY KEY REFERENCES operations(id),
                    snapshot TEXT NOT NULL
                )
            """)

    def save_deletion_backup(self, op_id: str, snapshot: dict[str, Any]) -> None:
        """在发出删除请求之前持久化原数据；不随工具结果覆盖。"""
        if self.get(op_id) is None:
            raise KeyError(op_id)
        encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO deletion_backups VALUES (?, ?)", (op_id, encoded))

    def deletion_backup(self, op_id: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT snapshot FROM deletion_backups WHERE op_id=?", (op_id,)).fetchone()
        return json.loads(row[0]) if row else None

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

    def between(self, since: float | None = None, until: float | None = None) -> list[Operation]:
        """按创建时间取操作记录（对话记忆用），从新到旧；类型和关键词由调用方筛选。"""
        where, params = [], []
        if since is not None:
            where.append("created_at >= ?")
            params.append(since)
        if until is not None:
            where.append("created_at < ?")
            params.append(until)
        rows = self._db.execute(
            f"SELECT * FROM operations {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY created_at DESC",
            params).fetchall()
        return [self._from_row(row) for row in rows]

    def earliest(self) -> float | None:
        """最早一条操作记录的时间；没有记录返回 None。"""
        return self._db.execute("SELECT MIN(created_at) FROM operations").fetchone()[0]

    def for_turns(self, turn_ids: list[str]) -> dict[str, list[Operation]]:
        """这些轮次各自执行过的操作，按执行顺序。"""
        if not turn_ids:
            return {}
        placeholders = ", ".join("?" for _ in turn_ids)
        grouped: dict[str, list[Operation]] = {}
        for row in self._db.execute(
                f"SELECT * FROM operations WHERE turn_id IN ({placeholders}) ORDER BY created_at", turn_ids):
            grouped.setdefault(row["turn_id"], []).append(self._from_row(row))
        return grouped

    def set_tool_schemas(self, schemas: dict[str, dict[str, Any]]) -> None:
        """使用当前加载的目录比较新旧记录，不修改历史参数或真实调用。"""
        self._tool_schemas = deepcopy(schemas)

    def _recent_matching(self, name: str, arguments: dict[str, Any], status: OperationStatus,
                         within_s: float) -> Operation | None:
        if within_s < 0:
            raise ValueError("within_s 不能为负数")
        schema = self._tool_schemas.get(name, {})
        expected = _arguments_json(comparison_arguments(arguments, schema))
        # 不能先按原始 JSON 过滤，否则省略默认值的旧记录仍然查不到。
        rows = self._db.execute("""
            SELECT * FROM operations WHERE name=? AND status=? AND updated_at>=?
            ORDER BY updated_at DESC, rowid DESC
        """, (name, status, time.time() - within_s))
        for row in rows:
            op = self._from_row(row)
            if _arguments_json(comparison_arguments(op.arguments, schema)) == expected:
                return op
        return None

    def recent_succeeded(
        self, name: str, arguments: dict[str, Any], within_s: float = 600,
    ) -> Operation | None:
        """找时间窗口内最近成功的相同操作；只是查询记录，不自动拦截或重放操作。"""
        return self._recent_matching(name, arguments, "succeeded", within_s)

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

    def recent_unknown(self, name: str, arguments: dict[str, Any], within_s: float = 600) -> Operation | None:
        """有界的重提保护；过期不等于执行失败，原 unknown 记录仍保留。"""
        return self._recent_matching(name, arguments, "unknown", within_s)

    def acquire_process_lock(self) -> None:
        """生产入口在恢复前独占操作库，防止把另一进程的 executing 当成崩溃遗留。"""
        if self._db_path == ":memory:" or self._process_lock is not None:
            return
        import fcntl

        lock_file = open(self._db_path + ".lock", "a")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            raise RuntimeError("操作库正在被另一个 Alpha/Qwen 入口使用，请先关闭那个进程") from None
        self._process_lock = lock_file

    def close(self) -> None:
        try:
            self._db.close()
        finally:
            if self._process_lock is not None:
                self._process_lock.close()
                self._process_lock = None

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
