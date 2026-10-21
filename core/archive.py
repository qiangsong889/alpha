"""对话存档：每一轮说了什么、回答了什么、实际播放得怎样。

只存对话本身。工具调用的细节在 operations.db、费用在 costs.db，三者通过同一个 turn_id 关联，
不在这里重复保存。存档写入失败不能影响回答，调用方负责捕获异常。
"""

from pathlib import Path
import sqlite3
import time

TURN_STATUSES = frozenset({'ok', 'error'})
DELIVERIES = frozenset({'played', 'interrupted', 'not_played'})


class ArchiveStore:
    """在创建连接的线程使用；调用方负责 close。"""

    def __init__(self, db_path: str | Path):
        path = str(db_path)
        if path != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, timeout=5)
        self._db.row_factory = sqlite3.Row
        if path != ':memory:':
            self._db.execute('PRAGMA journal_mode=WAL')
        with self._db:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at REAL NOT NULL,
                    last_turn_at REAL NOT NULL,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    source TEXT
                )
            """)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    created_at REAL NOT NULL,
                    user_text TEXT NOT NULL,
                    answer_text TEXT,
                    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
                    error TEXT,
                    delivery TEXT NOT NULL DEFAULT 'not_reported'
                        CHECK (delivery IN ('not_reported', 'played', 'interrupted', 'not_played')),
                    delivered_at REAL,
                    spoken_text TEXT,
                    model TEXT,
                    brain_ms INTEGER
                )
            """)
            self._db.execute('CREATE INDEX IF NOT EXISTS turns_session ON turns (session_id, created_at)')

    def record_turn(self, *, turn_id: str, session_id: str, user_text: str, answer_text: str | None,
                    status: str, model: str | None, brain_ms: int | None, error: str | None = None,
                    source: str | None = None, created_at: float | None = None) -> None:
        if status not in TURN_STATUSES:
            raise ValueError(f'未知的轮次状态：{status}')
        now = time.time() if created_at is None else created_at
        with self._db:
            self._db.execute("""
                INSERT INTO sessions (session_id, started_at, last_turn_at, turn_count, source)
                VALUES (?, ?, ?, 0, ?)
                ON CONFLICT(session_id) DO NOTHING
            """, (session_id, now, now, source))
            # 同一 turn_id 只记一次；重复写入（例如重试保存）不会把计数加两遍。
            inserted = self._db.execute("""
                INSERT OR IGNORE INTO turns
                    (turn_id, session_id, created_at, user_text, answer_text, status, error, model, brain_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (turn_id, session_id, now, user_text, answer_text, status, error, model, brain_ms)).rowcount
            if inserted:
                self._db.execute("""
                    UPDATE sessions SET turn_count = turn_count + 1,
                        last_turn_at = MAX(last_turn_at, ?) WHERE session_id = ?
                """, (now, session_id))

    def mark_delivery(self, turn_id: str, delivery: str, spoken_text: str | None = None) -> bool:
        """语音层报告这一轮的回答实际播放得怎样；找不到这一轮时返回 False。"""
        if delivery not in DELIVERIES:
            raise ValueError(f'未知的播放状态：{delivery}')
        with self._db:
            return self._db.execute("""
                UPDATE turns SET delivery = ?, delivered_at = ?, spoken_text = ?
                WHERE turn_id = ?
            """, (delivery, time.time(), spoken_text, turn_id)).rowcount == 1

    def turn(self, turn_id: str) -> dict | None:
        row = self._db.execute('SELECT * FROM turns WHERE turn_id = ?', (turn_id,)).fetchone()
        return dict(row) if row is not None else None

    def session(self, session_id: str) -> dict | None:
        row = self._db.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,)).fetchone()
        return dict(row) if row is not None else None

    def close(self) -> None:
        self._db.close()
