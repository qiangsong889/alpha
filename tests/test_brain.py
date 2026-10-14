"""AlphaBrain 生命周期及行为回归测试，不访问真实服务。"""

import asyncio
import io
import sys
from contextlib import asynccontextmanager, redirect_stdout
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult, TextContent
import brain.agent as module
from apps import cli


def reply(content, stop_reason="end_turn"):
    return NS(content=content, stop_reason=stop_reason, usage=NS(
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    ))


class FakeWorker:
    def __init__(self, server):
        self.stopped = False
        self.call = AsyncMock(return_value=CallToolResult(
            content=[TextContent(type="text", text="任务列表")],
        ))

    async def run(self):
        try:
            await asyncio.Future()
        finally:
            self.stopped = True


class BrainTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.http_closed = False
        self.claude_closed = False
        self.client = NS(messages=NS(create=AsyncMock()))

        @asynccontextmanager
        async def http_context(**kwargs):
            try:
                yield NS()
            finally:
                self.http_closed = True

        @asynccontextmanager
        async def claude_context():
            try:
                yield self.client
            finally:
                self.claude_closed = True

        self.personal = [{"name": "list_items", "input_schema": {"type": "object"}}]
        self.ha = ([{"name": "intent__HassTurnOn", "input_schema": {"type": "object"}}],
                   [{"type": "text", "text": "Alpha + Assist"}])
        patches = [
            patch.dict(module.os.environ, {
                "HA_URL": "http://fake", "HA_TOKEN": "fake",
                "ANTHROPIC_API_KEY": "fake", "REMINDER_TIMEZONE": "America/Los_Angeles",
            }),
            patch.object(module, "load_dotenv"),
            patch.object(module.httpx2, "AsyncClient", http_context),
            patch.object(module, "AsyncAnthropic", claude_context),
            patch.object(module, "PersonalWorker", FakeWorker),
            patch.object(module, "load_personal_catalog", AsyncMock(return_value=self.personal)),
            patch.object(module, "load_ha_catalog", AsyncMock(return_value=self.ha)),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.brain = module.AlphaBrain()
        self.addAsyncCleanup(self.brain.close)

    async def test_history_time_prefix_and_idempotent_lifecycle(self):
        await self.brain.start()
        worker = self.brain.worker
        await asyncio.sleep(0)
        await self.brain.start()
        self.assertIs(self.brain.worker, worker)
        self.client.messages.create.side_effect = [
            reply([NS(type="text", text="你好")]),
            reply([NS(type="text", text="记得")]),
        ]
        with self.assertLogs("alpha.brain", level="INFO") as logs:
            self.assertEqual(await self.brain.ask("你好"), "你好")
            self.assertEqual(await self.brain.ask("记住了吗"), "记得")
        self.assertIn("cache_read=0", " ".join(logs.output))
        self.assertIn("时区：America/Los_Angeles", self.brain.messages[0]["content"])
        self.assertTrue(self.brain.messages[0]["content"].endswith("\n你好"))
        self.assertEqual(len(self.client.messages.create.call_args.kwargs["messages"]), 3)
        self.assertEqual(len(self.brain.messages), 4)
        await self.brain.close()
        await self.brain.close()
        self.assertTrue(worker.stopped)
        self.assertTrue(self.http_closed and self.claude_closed)

    async def test_device_shortcut_preserves_tool_history(self):
        await self.brain.start()
        call = NS(type="tool_use", name="intent__HassTurnOn", input={"name": "客厅灯"}, id="c1")
        self.client.messages.create.return_value = reply([call], "tool_use")
        result = CallToolResult(content=[TextContent(type="text", text='{"response_type":"action_done"}')])
        with patch.object(module, "call_ha_tool", AsyncMock(return_value=result)) as ha:
            self.assertEqual(await self.brain.ask("开灯"), "已打开客厅灯。")
        ha.assert_awaited_once()
        self.assertEqual(self.client.messages.create.await_count, 1)
        self.assertEqual(self.brain.messages[2]["content"][0]["tool_use_id"], "c1")
        self.assertEqual(len(self.brain.messages), 4)

    async def test_personal_tool_error_returns_to_claude(self):
        await self.brain.start()
        self.brain.worker.call.side_effect = ConnectionError("断开")
        call = NS(type="tool_use", name="list_items", input={"list_name": "My Tasks"}, id="c1")
        self.client.messages.create.side_effect = [reply([call], "tool_use"), reply([NS(type="text", text="查询失败")])]
        self.assertEqual(await self.brain.ask("查看任务"), "查询失败")
        result = self.brain.messages[2]["content"][0]
        self.assertTrue(result["is_error"])
        self.assertIn("断开", result["content"])
        self.assertEqual([label for label, _ in self.brain.last_timings], ["Claude 第1次", "工具", "Claude 第2次"])

    async def test_failed_discovery_retries_before_ask(self):
        module.load_ha_catalog.side_effect = [None, self.ha]
        await self.brain.start()
        self.assertFalse(self.brain.ha_available)
        self.assertIn("家居设备暂时不可用", self.brain.system[0]["text"])
        self.client.messages.create.return_value = reply([NS(type="text", text="好了")])
        await self.brain.ask("再试一下")
        self.assertTrue(self.brain.ha_available)
        self.assertEqual(module.load_ha_catalog.await_count, 2)

    async def test_api_failure_does_not_save_incomplete_turn(self):
        await self.brain.start()
        self.client.messages.create.side_effect = RuntimeError("API 失败")
        with self.assertRaisesRegex(RuntimeError, "API 失败"):
            await self.brain.ask("你好")
        self.assertEqual(self.brain.messages, [])

    async def test_duplicate_tools_abort_start_and_cleanup(self):
        module.load_ha_catalog.return_value = (self.personal, self.ha[1])
        with self.assertRaisesRegex(RuntimeError, "工具重名"):
            await self.brain.start()
        self.assertTrue(self.http_closed)
        self.assertIsNone(self.brain.worker_task)
        with self.assertRaisesRegex(RuntimeError, "先调用"):
            await self.brain.ask("你好")

    async def test_text_cli_only_passes_request_and_closes(self):
        fake = NS(start=AsyncMock(), ask=AsyncMock(return_value="回答"),
                  close=AsyncMock(), last_timings=[])
        output = io.StringIO()
        with patch.object(cli, "AlphaBrain", return_value=fake), \
                patch("builtins.input", side_effect=["你好", "exit"]), redirect_stdout(output):
            await cli.main()
        fake.ask.assert_awaited_once_with("你好")
        fake.close.assert_awaited_once()
        self.assertIn("Alpha：回答", output.getvalue())

    async def test_voice_cli_retains_record_transcribe_speak(self):
        fake = NS(start=AsyncMock(), ask=AsyncMock(return_value="已关闭"),
                  close=AsyncMock(), last_timings=[])
        audio = NS(size=16000)
        listen = NS(_get_model=lambda: None,
                    record=unittest.mock.Mock(side_effect=[audio, EOFError()]),
                    transcribe=unittest.mock.Mock(return_value="关灯"))
        with patch.object(cli, "AlphaBrain", return_value=fake), \
                patch.dict(sys.modules, {"voice.listen": listen}), \
                patch.object(cli, "speak", AsyncMock()) as speak, \
                redirect_stdout(io.StringIO()):
            await cli.main(voice=True)
        fake.ask.assert_awaited_once_with("关灯")
        speak.assert_awaited_once_with("已关闭")
        fake.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
