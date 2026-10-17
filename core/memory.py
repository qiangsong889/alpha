"""用户明确要求保存的长期记忆；与操作记录共用文件，使用独立表。"""

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

    @staticmethod
    def _content(content: str) -> str:
        return validate_memory_content(content)

    def _capacity(self, content: str, *, replacing: int | None = None) -> None:
        rows = self._db.execute("SELECT id, content FROM memories").fetchall()
        kept = [row for row in rows if row['id'] != replacing]
        if len(kept) >= MAX_MEMORIES or sum(len(row['content']) for row in kept) + len(content) > MAX_CONTENT_CHARS:
            raise ValueError("记忆已满，请先删掉一些")

    def add(self, content: str, kind: MemoryKind, source_turn: str) -> int:
        content = self._content(content)
        if not isinstance(kind, str) or kind not in {"preference", "fact"}:
            raise ValueError("kind 必须是 preference 或 fact")
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            existing = self._db.execute(
                "SELECT id FROM memories WHERE kind=? AND content=?", (kind, content)).fetchone()
            if existing is not None:
                return existing['id']
            self._capacity(content)
            now = time.time()
            cursor = self._db.execute(
                "INSERT INTO memories(kind, content, created_at, updated_at, source_turn) VALUES(?,?,?,?,?)",
                (kind, content, now, now, source_turn))
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def update(self, id: int, content: str) -> None:
        content = self._content(content)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            if self._db.execute("SELECT id FROM memories WHERE id=?", (id,)).fetchone() is None:
                raise ValueError("记忆不存在")
            self._capacity(content, replacing=id)
            try:
                self._db.execute("UPDATE memories SET content=?, updated_at=? WHERE id=?", (content, time.time(), id))
            except sqlite3.IntegrityError as exc:
                raise ValueError("相同记忆已存在，请删除重复项") from exc

    def delete(self, id: int) -> None:
        with self._db:
            if self._db.execute("DELETE FROM memories WHERE id=?", (id,)).rowcount != 1:
                raise ValueError("记忆不存在")

    def list(self) -> list[Memory]:
        return [Memory(row['id'], cast(MemoryKind, row['kind']), row['content'], row['created_at'],
                       row['updated_at'], row['source_turn'])
                for row in self._db.execute("SELECT * FROM memories ORDER BY id")]

    def close(self) -> None:
        self._db.close()
