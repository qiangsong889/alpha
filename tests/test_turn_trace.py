"""session_id → turn_id → 操作 / 费用 / 存档 的关联；不请求真实模型或工具。"""

import asyncio
from contextlib import asynccontextmanager
import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

import brain.agent as module
from brain import server
from brain.context import ConversationContext
from core import costs
from core.archive import ArchiveStore
from core.memory import MemoryStore
from core.trace import CURRENT_SESSION, CURRENT_TURN
from integrations.operations import OperationStore
from integrations.tool_gateway import ExecutorResult
from tests.test_brain import FakeWorker
from tests.test_context import history


def reply(content, stop_reason="end_turn"):
    # 带完整用量，账本能算出价格；归属测试关心的是 turn_id 而不是金额。
    return NS(content=content, stop_reason=stop_reason, model="claude-haiku-5-5", usage=NS(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0,
        model_dump=lambda mode="json": {"input_tokens": 10, "output_tokens": 5,
                                        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}))


class ArchiveStoreTests(unittest.TestCase):
    def setUp(self):
        self.archive = ArchiveStore(":memory:")
        self.addCleanup(self.archive.close)

    def record(self, turn_id="t1", session_id="s1", **extra):
        values = dict(turn_id=turn_id, session_id=session_id, user_text="开灯", answer_text="好了",
                      status="ok", model="claude-haiku-5-5", brain_ms=900)
        values.update(extra)
        self.archive.record_turn(**values)

    def test_turn_creates_session_and_counts_each_turn_once(self):
        self.record("t1", created_at=100)
        self.record("t2", created_at=130)
        self.record("t2", created_at=999)  # 同一轮重复写入不会重复计数
        session = self.archive.session("s1")
        self.assertEqual(session["turn_count"], 2)
        self.assertEqual((session["started_at"], session["last_turn_at"]), (100, 130))
        self.assertEqual(self.archive.turn("t1")["delivery"], "not_reported")

    def test_delivery_is_recorded_and_validated(self):
        self.record("t1")
        self.assertTrue(self.archive.mark_delivery("t1", "interrupted"))
        self.assertEqual(self.archive.turn("t1")["delivery"], "interrupted")
        self.assertFalse(self.archive.mark_delivery("missing", "played"))
        with self.assertRaises(ValueError):
            self.archive.mark_delivery("t1", "heard")
        with self.assertRaises(ValueError):
            self.record("t3", status="maybe")


class CostAttributionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, costs, "_ledger", None)

    def test_old_ledger_gains_columns_and_keeps_old_rows_unattributed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "costs.db"
            db = sqlite3.connect(path)
            db.execute('''CREATE TABLE cost_events (id TEXT PRIMARY KEY, component TEXT NOT NULL,
                model TEXT NOT NULL, region TEXT NOT NULL, usage TEXT NOT NULL, usd TEXT NOT NULL,
                reason TEXT NOT NULL, updated_at REAL NOT NULL)''')
            db.execute("INSERT INTO cost_events VALUES ('old','Claude','m','singapore','{}','0','',1)")
            db.commit()
            db.close()
            ledger = costs.configure_costs(path)
            row = ledger.db.execute("SELECT session_id, turn_id FROM cost_events WHERE id='old'").fetchone()
            self.assertEqual(row, (None, None))
            ledger.close()
            costs._ledger = None

    def test_context_attribution_explicit_override_and_first_write_wins(self):
        ledger = costs.configure_costs(":memory:")
        session, turn = CURRENT_SESSION.set("s1"), CURRENT_TURN.set("t1")
        try:
            ident = costs.record_cost("Claude", "claude-haiku-5-5", reason="request_started_no_final_usage")
            CURRENT_TURN.set("t2")  # 最终用量在别的上下文写入，也不能改掉最初的归属
            costs.record_cost("Claude", "claude-haiku-5-5", {"input_tokens": 1, "output_tokens": 1},
                              event_id=ident)
            costs.record_cost("TTS", "qwen3-tts-flash-realtime", {"characters": 3},
                              event_id="tts", session_id=None, turn_id="voice-turn")
        finally:
            CURRENT_TURN.reset(turn)
            CURRENT_SESSION.reset(session)
        rows = dict(ledger.db.execute("SELECT id, turn_id FROM cost_events").fetchall())
        self.assertEqual(rows[ident], "t1")
        self.assertEqual(rows["tts"], "voice-turn")


class BrainTraceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = NS(messages=NS(create=AsyncMock()))
        self.executor = AsyncMock(return_value=ExecutorResult(ok=True, result="已打开"))

        @asynccontextmanager
        async def http_context(**kwargs):
            yield NS()

        @asynccontextmanager
        async def claude_context():
            yield self.client

        patches = [
            patch.dict(module.os.environ, {
                "HA_URL": "http://fake", "HA_TOKEN": "fake", "ANTHROPIC_API_KEY": "fake",
                "ANTHROPIC_MODEL": "claude-haiku-5-5", "REMINDER_TIMEZONE": "America/Los_Angeles",
                "ALPHA_AUTO_MEMORY": "0", "ALPHA_CONTEXT_IDLE_MIN": "30",
                "ALPHA_CONTEXT_COMPACT_TOKENS": "15000", "ALPHA_CONTEXT_HARD_TOKENS": "40000",
            }),
            patch.object(module, "load_dotenv"),
            patch.object(module, "OperationStore", side_effect=lambda path: OperationStore(":memory:")),
            patch.object(module, "MemoryStore", side_effect=lambda path: MemoryStore(":memory:")),
            patch.object(module, "make_executor", return_value=self.executor),
            patch.object(module.httpx2, "AsyncClient", http_context),
            patch.object(module, "AsyncAnthropic", claude_context),
            patch.object(module, "PersonalWorker", FakeWorker),
            patch.object(module, "load_personal_catalog",
                         AsyncMock(return_value=[{"name": "list_items", "input_schema": {"type": "object"}}])),
            patch.object(module, "load_ha_catalog", AsyncMock(return_value=(
                [{"name": "intent__HassTurnOn", "input_schema": {"type": "object"}}],
                [{"type": "text", "text": "Alpha + Assist"}]))),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.ledger = costs.configure_costs(":memory:")
        self.addCleanup(setattr, costs, "_ledger", None)
        self.brain = module.AlphaBrain(archive_path=":memory:")
        self.addAsyncCleanup(self.brain.close)
        await self.brain.start()

    def cost_rows(self):
        return self.ledger.db.execute(
            "SELECT component, session_id, turn_id FROM cost_events ORDER BY rowid").fetchall()

    async def test_one_turn_links_claude_costs_operation_and_archive(self):
        call = NS(type="tool_use", name="intent__HassTurnOn", input={"name": "客厅灯"}, id="c1")
        self.client.messages.create.side_effect = [reply([call], "tool_use"), reply([NS(type="text", text="好")])]
        await self.brain.ask("开灯", turn_id="a" * 32)

        session_id = self.brain.session_id
        turn = self.brain.archive.turn("a" * 32)
        self.assertEqual((turn["session_id"], turn["status"], turn["user_text"]), (session_id, "ok", "开灯"))
        self.assertTrue(turn["answer_text"])
        claude = [row for row in self.cost_rows() if row[0] == "Claude"]
        self.assertEqual(len(claude), 2)
        self.assertTrue(all(row[1:] == (session_id, "a" * 32) for row in claude))
        ops = self.brain.store._db.execute("SELECT name, turn_id FROM operations").fetchall()
        self.assertEqual([tuple(row) for row in ops], [("intent__HassTurnOn", "a" * 32)])
        # 上下文变量在一轮结束后恢复，下一轮之外的代码不会继承它。
        self.assertIsNone(CURRENT_TURN.get())
        self.assertIsNone(CURRENT_SESSION.get())

    async def test_generated_turn_id_and_failed_turn_are_archived_without_message_text(self):
        self.client.messages.create.side_effect = RuntimeError("secret detail")
        with self.assertRaises(RuntimeError):
            await self.brain.ask("查一下")
        rows = self.brain.archive._db.execute("SELECT turn_id, status, error, answer_text FROM turns").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["turn_id"]), 32)
        self.assertEqual((rows[0]["status"], rows[0]["error"], rows[0]["answer_text"]),
                         ("error", "RuntimeError", None))

    async def test_archive_failure_does_not_change_the_answer(self):
        self.client.messages.create.side_effect = [reply([NS(type="text", text="你好")])]
        self.brain.archive.record_turn = lambda **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("locked"))
        with self.assertLogs("alpha.brain", level="WARNING") as logs:
            self.assertEqual(await self.brain.ask("你好"), "你好")
        self.assertIn("[存档] 写入失败", " ".join(logs.output))

    async def test_idle_gap_starts_a_new_session_for_archive_and_costs(self):
        now = [1000.0]
        self.brain.conversation.clock = lambda: now[0]
        self.client.messages.create.side_effect = [reply([NS(type="text", text="一")]),
                                                   reply([NS(type="text", text="二")])]
        await self.brain.ask("第一句", turn_id="1" * 32)
        first_session = self.brain.session_id
        now[0] += 31 * 60
        await self.brain.ask("第二句", turn_id="2" * 32)
        self.assertNotEqual(self.brain.session_id, first_session)
        self.assertEqual(self.brain.archive.turn("1" * 32)["session_id"], first_session)
        self.assertEqual(self.brain.archive.turn("2" * 32)["session_id"], self.brain.session_id)
        sessions = {row[2]: row[1] for row in self.cost_rows()}
        self.assertEqual(sessions["1" * 32], first_session)
        self.assertEqual(sessions["2" * 32], self.brain.session_id)

    async def test_delivery_updates_archive_only(self):
        self.client.messages.create.side_effect = [reply([NS(type="text", text="好的")])]
        await self.brain.ask("播报", turn_id="d" * 32)
        history = list(self.brain.messages)
        self.assertTrue(await self.brain.mark_delivery("d" * 32, "interrupted"))
        self.assertEqual(self.brain.archive.turn("d" * 32)["delivery"], "interrupted")
        self.assertEqual(self.brain.messages, history)
        self.assertFalse(await self.brain.mark_delivery("e" * 32, "played"))

    async def test_archive_is_off_unless_an_entry_point_enables_it(self):
        plain = module.AlphaBrain()
        self.addAsyncCleanup(plain.close)
        await plain.start()
        self.assertIsNone(plain.archive)


class CompactionAttributionTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_compaction_belongs_to_session_not_last_turn(self):
        context = ConversationContext(soft_tokens=100, hard_tokens=20000)
        self.addAsyncCleanup(context.close)
        brain = NS(messages=history(), model="claude-haiku-5-5", _lock=asyncio.Lock(),
                   client=NS(messages=NS(count_tokens=AsyncMock(return_value=NS(input_tokens=9000)))))
        seen = []

        async def summarize(*args):
            seen.append((CURRENT_SESSION.get(), CURRENT_TURN.get()))
            return '{"user_facts":["旧目标"]}'
        context.summarize = summarize
        session, turn = CURRENT_SESSION.set("s1"), CURRENT_TURN.set("last-turn")
        try:
            context.after_turn(brain)  # 任务在这一轮的上下文里创建
        finally:
            CURRENT_TURN.reset(turn)
            CURRENT_SESSION.reset(session)
        await context.task
        self.assertEqual(seen, [("s1", None)])


class ServerTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_event_only_for_request_id_clients_and_same_id_reaches_brain(self):
        seen = []

        async def ask(text, *, on_text, on_signal, search_announced=False, turn_id=None):
            seen.append(turn_id)
            await on_text("好")

        async def collect(request):
            response = await server.ask_stream(request)
            return [json.loads(line) async for line in response.body_iterator]

        with patch.object(server, "brain", NS(ask=ask, last_timings=[])):
            with_id = await collect(server.AskRequest(text="开灯", request_id="batch1"))
            without = await collect(server.AskRequest(text="开灯"))
        self.assertEqual(with_id[0], {"type": "turn", "request_id": "batch1", "turn_id": seen[0]})
        self.assertTrue(all(event["type"] != "turn" for event in without))
        self.assertEqual([len(turn_id) for turn_id in seen], [32, 32])

    async def test_delivered_validates_and_forwards(self):
        mark = AsyncMock(return_value=True)
        with patch.object(server, "brain", NS(mark_delivery=mark)):
            result = await server.delivered(server.DeliveredRequest(turn_id="f" * 32, delivery="played"))
        self.assertEqual(result, {"ok": True})
        mark.assert_awaited_once_with("f" * 32, "played")
        with self.assertRaises(ValueError):
            server.DeliveredRequest(turn_id="not-a-turn", delivery="played")
        with self.assertRaises(ValueError):
            server.DeliveredRequest(turn_id="f" * 32, delivery="heard")


if __name__ == "__main__":
    unittest.main()
