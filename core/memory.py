"""长期记忆和来源记录；与操作记录共用文件，使用独立表。"""

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time
from typing import Literal, cast

MemoryKind = Literal["preference", "fact"]
MAX_MEMORIES = 50
MAX_CONTENT_CHARS = 2000
# 标签后只允许空白和必需的分隔符；Bearer 单独要求至少 16 位 token，减少普通词语误伤。
# 格式检测是第二道防线，无法识别所有无标签秘密；模型仍必须遵守不保存秘密的规则。
_SECRET = re.compile(
    r"(?:密码|口令|验证码|私钥|密钥|访问令牌|API[ _-]?KEY|access[ _-]?token|refresh[ _-]?token|"
    r"password|passwd|secret|authorization)\s*(?:是|为|[:：=])\s*\S+"
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"
    r"|\bsk-[A-Za-z0-9_-]{8,}|\bAIza[A-Za-z0-9_-]{20,}"
    r"|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----", re.IGNORECASE,
)


def validate_memory_content(content: str) -> str:
    """无需数据库的内容校验；提取器和持久存储共用秘密格式检测。"""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("记忆内容不能为空")
    content = content.strip()
    if _SECRET.search(content):
        raise ValueError("不能保存密码、密钥、验证码等秘密信息")
    return content


@dataclass(frozen=True)
class Memory:
    id: int
    kind: MemoryKind
    content: str
    created_at: float
    updated_at: float
    source_turn: str
    source: Literal["explicit", "auto"] = "explicit"


class MemoryStore:
    """在创建连接的线程使用；调用方负责 close，SQLite 事务内检查容量。"""

    def __init__(self, db_path: str | Path):
        self._db = sqlite3.connect(str(db_path))
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL CHECK(kind IN ('preference', 'fact')),
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    source_turn TEXT NOT NULL,
                    UNIQUE(kind, content)
                )
            """)
            columns = {row['name'] for row in self._db.execute("PRAGMA table_info(memories)")}
            if 'source' not in columns:
                self._db.execute("ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'explicit' CHECK(source IN ('explicit','auto'))")
            self._db.execute("CREATE TABLE IF NOT EXISTS memory_revision (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL)")
            self._db.execute("INSERT OR IGNORE INTO memory_revision VALUES(1, 0)")
            for event in ('INSERT', 'UPDATE', 'DELETE'):
                self._db.execute(f"""CREATE TRIGGER IF NOT EXISTS memory_revision_{event.lower()}
                    AFTER {event} ON memories BEGIN
                    UPDATE memory_revision SET version=version+1 WHERE id=1;
                    END""")

    def version(self) -> int:
        return self._db.execute("SELECT version FROM memory_revision WHERE id=1").fetchone()[0]

    def snapshot(self) -> tuple[int, list[Memory]]:
        with self._db:
            self._db.execute("BEGIN")
            return self.version(), self.list()

    @staticmethod
    def _content(content: str) -> str:
        return validate_memory_content(content)

    def _capacity(self, content: str, *, replacing: int | None = None) -> None:
        rows = self._db.execute("SELECT id, content FROM memories").fetchall()
        kept = [row for row in rows if row['id'] != replacing]
        if len(kept) >= MAX_MEMORIES or sum(len(row['content']) for row in kept) + len(content) > MAX_CONTENT_CHARS:
            raise ValueError("记忆已满，请先删掉一些")

    def _add(self, content, kind, source_turn, source):
        content = self._content(content)
        if kind not in ('preference', 'fact') or source not in ('explicit', 'auto'):
            raise ValueError("kind 或 source 无效")
        existing = self._db.execute(
            "SELECT id FROM memories WHERE kind=? AND content=?", (kind, content)).fetchone()
        if existing is not None:
            return existing['id']
        self._capacity(content)
        now = time.time()
        cursor = self._db.execute(
            "INSERT INTO memories(kind, content, created_at, updated_at, source_turn, source) VALUES(?,?,?,?,?,?)",
            (kind, content, now, now, source_turn, source))
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def add(self, content: str, kind: MemoryKind, source_turn: str, *,
            source: Literal["explicit", "auto"] = "explicit") -> int:
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            return self._add(content, kind, source_turn, source)

    def _update(self, id, content, source, source_turn):
        content = self._content(content)
        if source not in ('explicit', 'auto'):
            raise ValueError("source 无效")
        if self._db.execute("SELECT id FROM memories WHERE id=?", (id,)).fetchone() is None:
            raise ValueError("记忆不存在")
        self._capacity(content, replacing=id)
        try:
            self._db.execute("UPDATE memories SET content=?, updated_at=?, source=?, source_turn=COALESCE(?,source_turn) WHERE id=?",
                             (content, time.time(), source, source_turn, id))
        except sqlite3.IntegrityError as exc:
            raise ValueError("相同记忆已存在，请删除重复项") from exc

    def update(self, id: int, content: str, *, source: Literal["explicit", "auto"] = "explicit",
               source_turn: str | None = None) -> None:
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            self._update(id, content, source, source_turn)

    def apply_auto(self, expected_version: int, changes: list[dict], source_turn: str) -> list[Memory] | None:
        """版本检查与整批写入在同一事务完成；None 表示快照已过时，异常则整批回滚。"""
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            if self.version() != expected_version:
                return None
            changed = []
            for item in changes:
                before = self.version()
                if item['operation'] == 'add':
                    ident = self._add(item['content'], item['kind'], source_turn, 'auto')
                elif item['operation'] == 'update':
                    ident = item['memory_id']
                    target = next((row for row in self.list() if row.id == ident), None)
                    if target is None or target.kind != item['kind']:
                        raise ValueError("自动记忆更新目标已失效")
                    self._update(ident, item['content'], 'auto', source_turn)
                else:
                    raise ValueError("自动记忆操作无效")
                if self.version() != before:
                    changed.append(next(row for row in self.list() if row.id == ident))
            return changed

    def delete(self, id: int) -> None:
        with self._db:
            if self._db.execute("DELETE FROM memories WHERE id=?", (id,)).rowcount != 1:
                raise ValueError("记忆不存在")

    def list(self) -> list[Memory]:
        return [Memory(row['id'], cast(MemoryKind, row['kind']), row['content'], row['created_at'],
                       row['updated_at'], row['source_turn'], row['source'])
                for row in self._db.execute("SELECT * FROM memories ORDER BY id")]

    def close(self) -> None:
        self._db.close()
