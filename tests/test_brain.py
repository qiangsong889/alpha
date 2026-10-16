"""AlphaBrain 生命周期及行为回归测试，不访问真实服务。"""

import asyncio
import io
import json
import sys
from contextlib import asynccontextmanager, redirect_stdout
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from mcp.types import CallToolResult, TextContent
import brain.agent as module
from apps import cli
from integrations.operations import OperationStore
from integrations.tool_gateway import ExecutorResult


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
    def final_stream(self, answer="已完成。"):
        call = NS(type="tool_use", name=module.VOICE_REPLY, input={"answer": answer}, id="final")
        return self.fake_stream(reply([call], "tool_use"),
                                self.speech_events(json.dumps({"answer": answer}, ensure_ascii=False)))

    def fake_stream(self, response, events=(), final_gate=None, final_waiting=None):
        class Stream:
            async def __aiter__(self):
                for event in events:
                    yield event

            async def get_final_message(self):
                if final_waiting is not None:
                    final_waiting.set()
                if final_gate is not None:
                    await final_gate.wait()
                return response

        @asynccontextmanager
        async def manager():
            yield Stream()
        return manager()

    def speech_events(self, *parts):
        return [NS(type="content_block_start", index=0,
                   content_block=NS(type="tool_use", name=module.VOICE_REPLY)), *[
            NS(type="content_block_delta", index=0,
               delta=NS(type="input_json_delta", partial_json=part)) for part in parts
        ]]

    async def test_speech_waits_until_full_response_has_been_checked(self):
        await self.brain.start()
        final = reply([NS(type="tool_use", name=module.VOICE_REPLY,
                          input={"answer": "你好。还需要什么？"}, id="voice")], "tool_use")
        gate, arrived = asyncio.Event(), asyncio.Event()
        parts = []

        async def emit(text):
            parts.append(text)

        self.client.messages.stream = Mock(return_value=self.fake_stream(final,
            self.speech_events('{"answer":"你好。', '还需要什么？"}'), gate, arrived))
        task = asyncio.create_task(self.brain.ask("你好", on_text=emit))
        try:
            await asyncio.wait_for(arrived.wait(), 1)
            self.assertFalse(task.done())
            self.assertEqual(parts, [])
            gate.set()
            self.assertEqual(await task, "你好。还需要什么？")
        finally:
            gate.set()
            await task
        self.assertEqual("".join(parts), "你好。还需要什么？")
        self.executor.assert_not_awaited()
        self.assertEqual(self.brain.messages[-1]["content"], "你好。还需要什么？")
        self.assertNotIn("disable_parallel_tool_use", self.client.messages.stream.call_args.kwargs["tool_choice"])

    async def test_one_response_can_request_multiple_business_tools(self):
        self.add_catalog_tools("add_items", "create_reminder")
        await self.brain.start()
        expected = [("intent__HassTurnOn", {"name": "客厅灯"}),
                    ("add_items", {"list_name": "购物清单", "items": ["牛奶"]}),
                    ("create_reminder", {"title": "开会", "local_time": "2026-10-16T08:00:00"})]
        calls = [NS(type="tool_use", name=name, input=args, id=f"tool-{i}")
                 for i, (name, args) in enumerate(expected)]
        self.client.messages.stream = Mock(side_effect=[
            self.fake_stream(reply(calls, "tool_use")), self.final_stream("事情都办好了。"),
        ])
        async def emit(text):
            self.assertEqual(self.executor.await_count, 3)
        await self.brain.ask("开灯，加牛奶，明早八点提醒开会", on_text=emit)
        self.assertEqual([(c.args[0], c.args[1]) for c in self.executor.await_args_list], expected)
        self.assertEqual(self.client.messages.stream.call_count, 2)
        self.assertEqual([r["tool_use_id"] for r in self.brain.messages[2]["content"]],
                         ["tool-0", "tool-1", "tool-2"])

    async def test_early_summary_with_later_business_call_is_not_spoken(self):
        await self.brain.start()
        early = NS(type="tool_use", name=module.VOICE_REPLY, input={"answer": "已经全做完了。"}, id="early")
        tool = NS(type="tool_use", name="list_items", input={"list_name": "购物清单"}, id="query")
        self.client.messages.stream = Mock(side_effect=[
            self.fake_stream(reply([early, tool], "tool_use"), self.speech_events('{"answer":"已经全做完了。"}')),
            self.final_stream("清单里有牛奶。"),
        ])
        emit = AsyncMock()
        await self.brain.ask("购物清单里有什么", on_text=emit)
        self.executor.assert_awaited_once()
        emit.assert_awaited_once_with("清单里有牛奶。")
        results = self.brain.messages[2]["content"]
        self.assertEqual([r["tool_use_id"] for r in results], ["early", "query"])
        self.assertTrue(results[0]["is_error"])
        self.assertFalse(results[1]["is_error"])

    async def test_stream_does_not_read_tool_preamble_and_executes_before_answer(self):
        await self.brain.start()
        tool = NS(type="tool_use", name="list_items", input={"list_name": "购物清单"}, id="list")
        final = NS(type="tool_use", name=module.VOICE_REPLY, input={"answer": "清单里有牛奶。"}, id="voice")
        self.client.messages.stream = Mock(side_effect=[
            self.fake_stream(reply([tool], "tool_use"), [
                NS(type="content_block_delta", index=0,
                   delta=NS(type="text_delta", text="我这就查一下")),
            ]),
            self.fake_stream(reply([final], "tool_use"), self.speech_events('{"answer":"清单里有牛奶。"}')),
        ])
        parts = []

        async def emit(text):
            self.executor.assert_awaited_once()
            parts.append(text)

        await self.brain.ask("清单里有什么", on_text=emit)
        self.assertEqual("".join(parts), "清单里有牛奶。")
        self.assertEqual(self.client.messages.stream.call_count, 2)

    async def test_stream_device_waits_for_explicit_turn_completion(self):
        await self.brain.start()
        tool = NS(type="tool_use", name="intent__HassTurnOn", input={"name": "客厅灯"}, id="on")
        self.client.messages.stream = Mock(side_effect=[
            self.fake_stream(reply([tool], "tool_use")), self.final_stream("已打开客厅灯。"),
        ])
        emit = AsyncMock()
        self.assertEqual(await self.brain.ask("开灯", on_text=emit), "已打开客厅灯。")
        emit.assert_awaited_once_with("已打开客厅灯。")
        self.assertEqual(self.client.messages.stream.call_count, 2)

    async def invoke_tool(self, name, arguments, request="执行", *, call_id="c1"):
        call = NS(type="tool_use", name=name, input=arguments, id=call_id)
        self.client.messages.create.side_effect = [
            reply([call], "tool_use"), reply([NS(type="text", text="已核实结果")]),
        ]
        await self.brain.ask(request)
        return self.brain.messages[-2]["content"][0]

    def add_catalog_tools(self, *names):
        self.personal.extend({"name": name, "input_schema": {
            "type": "object", "properties": {}, "additionalProperties": False,
        }} for name in names)

    async def asyncSetUp(self):
        self.http_closed = False
        self.claude_closed = False
        self.client = NS(messages=NS(create=AsyncMock()))
        self.executor = AsyncMock(return_value=ExecutorResult(ok=True, result="任务列表"))

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
            patch.object(module, "OperationStore", side_effect=lambda path: OperationStore(":memory:")),
            patch.object(module, "make_executor", return_value=self.executor),
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

    async def test_device_summary_preserves_tool_history(self):
        await self.brain.start()
        call = NS(type="tool_use", name="intent__HassTurnOn", input={"name": "客厅灯"}, id="c1")
        self.client.messages.create.side_effect = [reply([call], "tool_use"), reply([NS(type="text", text="完成")])]
        self.assertEqual(await self.brain.ask("开灯"), "已打开客厅灯。")
        self.executor.assert_awaited_once_with("intent__HassTurnOn", {"name": "客厅灯"})
        self.assertEqual(self.client.messages.create.await_count, 2)
        self.assertEqual(self.brain.messages[2]["content"][0]["tool_use_id"], "c1")
        self.assertEqual(len(self.brain.messages), 4)

    async def test_personal_tool_error_returns_to_claude(self):
        await self.brain.start()
        self.executor.side_effect = ConnectionError("断开")
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

    async def test_unclassified_loaded_tool_is_rejected(self):
        await self.brain.start()
        self.brain.tools.append({"name": "unclassified", "input_schema": {"type": "object"}})
        result = await self.invoke_tool("unclassified", {})
        self.assertTrue(result["is_error"])
        self.executor.assert_not_awaited()

    async def test_policy_tool_not_loaded_is_rejected(self):
        await self.brain.start()
        result = await self.invoke_tool("create_reminder", {"title": "开会"})
        self.assertTrue(result["is_error"])
        self.executor.assert_not_awaited()

    async def test_duplicate_reminder_is_recorded_but_not_executed(self):
        self.add_catalog_tools("create_reminder")
        await self.brain.start()
        args = {"title": "开会", "local_time": "2026-10-16T08:00:00"}
        first = await self.invoke_tool("create_reminder", args)
        second = await self.invoke_tool("create_reminder", dict(reversed(list(args.items()))))
        self.assertFalse(first["is_error"])
        self.assertIn("本次没有执行", second["content"])
        self.assertIn("confirm_duplicate", second["content"])
        self.assertEqual(self.executor.await_count, 1)
        # 拦截也必须留下独立记录，而不是借用之前成功的那一条。
        rows = self.brain.store._db.execute("SELECT status, turn_id FROM operations ORDER BY rowid").fetchall()
        self.assertEqual([row["status"] for row in rows], ["succeeded", "cancelled"])
        self.assertNotEqual(rows[0]["turn_id"], rows[1]["turn_id"])

    async def test_explicit_duplicate_control_is_removed_before_executor(self):
        self.add_catalog_tools("create_reminder")
        await self.brain.start()
        args = {"title": "开会", "local_time": "2026-10-16T08:00:00"}
        await self.invoke_tool("create_reminder", args)
        await self.invoke_tool("create_reminder", {**args, "confirm_duplicate": True})
        self.assertEqual(self.executor.await_count, 2)
        self.executor.assert_awaited_with("create_reminder", args)
        schema = next(tool for tool in self.brain.tools if tool["name"] == "create_reminder")["input_schema"]
        self.assertEqual(schema["properties"]["confirm_duplicate"]["type"], "boolean")
        self.assertNotIn("confirm_duplicate", self.personal[-1]["input_schema"]["properties"])

    async def test_string_true_does_not_bypass_duplicate_check(self):
        self.add_catalog_tools("add_items")
        await self.brain.start()
        args = {"list_name": "购物清单", "items": ["牛奶"]}
        await self.invoke_tool("add_items", args)
        result = await self.invoke_tool("add_items", {**args, "confirm_duplicate": "true"})
        self.assertIn("本次没有执行", result["content"])
        self.assertEqual(self.executor.await_count, 1)

    async def test_timeout_is_unknown_and_cannot_be_bypassed(self):
        self.add_catalog_tools("create_reminder")
        await self.brain.start()
        self.executor.side_effect = TimeoutError("等待响应超时")
        args = {"title": "开会", "local_time": "2026-10-16T08:00:00"}
        result = await self.invoke_tool("create_reminder", args)
        self.assertTrue(result["is_error"])
        self.assertIn("不要重试", result["content"])
        self.assertEqual(self.brain.store.recent_unknown("create_reminder", args).status, "unknown")
        result = await self.invoke_tool("create_reminder", {**args, "confirm_duplicate": True})
        self.assertTrue(result["is_error"])
        self.assertIn("本次没有执行", result["content"])
        self.assertEqual(self.executor.await_count, 1)

    async def test_off_on_off_all_execute_without_duplicate_suppression(self):
        self.ha[0].append({"name": "intent__HassTurnOff", "input_schema": {"type": "object"}})
        await self.brain.start()
        for name in ["intent__HassTurnOff", "intent__HassTurnOn", "intent__HassTurnOff"]:
            self.client.messages.create.side_effect = [reply([
                NS(type="tool_use", name=name, input={"name": "客厅灯"}, id="device"),
            ], "tool_use"), reply([NS(type="text", text="完成")])]
            self.assertIn("客厅灯", await self.brain.ask("操作灯"))
        self.assertEqual(self.executor.await_count, 3)
        self.assertEqual(self.client.messages.create.await_count, 6)

    async def test_old_ha_unknown_does_not_block_new_user_command(self):
        await self.brain.start()
        self.executor.side_effect = [TimeoutError("HA 重启"), ExecutorResult(ok=True, result="成功")]
        args = {"name": "客厅灯"}
        await self.invoke_tool("intent__HassTurnOn", args)
        result = await self.invoke_tool("intent__HassTurnOn", args)
        self.assertFalse(result["is_error"])
        self.assertEqual(self.executor.await_count, 2)
        # 不把以前的未知执行擅自改成成功。
        self.assertEqual(self.brain.store.recent_unknown("intent__HassTurnOn", args).status, "unknown")

    async def test_ha_unknown_still_blocks_automatic_retry_in_same_turn(self):
        await self.brain.start()
        self.executor.side_effect = TimeoutError()
        tool = NS(type="tool_use", name="intent__HassTurnOn", input={"name": "客厅灯"}, id="ha")
        self.client.messages.create.side_effect = [
            reply([tool], "tool_use"), reply([tool], "tool_use"), reply([NS(type="text", text="请查询确认")]),
        ]
        await self.brain.ask("开灯")
        self.assertEqual(self.executor.await_count, 1)

    async def test_add_items_unknown_expires_and_does_not_get_refreshed_by_blocked_attempts(self):
        self.add_catalog_tools("add_items")
        await self.brain.start()
        args = {"list_name": "购物清单", "items": ["牛奶"]}
        self.executor.side_effect = TimeoutError()
        await self.invoke_tool("add_items", args)
        original = self.brain.store.recent_unknown("add_items", args)
        await self.invoke_tool("add_items", args)
        self.assertEqual(self.brain.store.get(original.id).updated_at, original.updated_at)
        self.executor.side_effect = None
        with patch("integrations.operations.time.time", return_value=original.updated_at + 601):
            result = await self.invoke_tool("add_items", args)
        self.assertFalse(result["is_error"])
        self.assertEqual(self.executor.await_count, 2)

    async def test_ha_success_signals_only_after_turn_completion(self):
        await self.brain.start()
        tool = NS(type="tool_use", name="intent__HassTurnOn", input={"name": "客厅灯"}, id="ha")
        self.client.messages.stream = Mock(side_effect=[
            self.fake_stream(reply([tool], "tool_use")), self.final_stream(),
        ])
        emit, signal = AsyncMock(), AsyncMock()
        await self.brain.ask("开灯", on_text=emit, on_signal=signal)
        emit.assert_not_called()
        signal.assert_awaited_once_with("ha_done")
        self.assertEqual(self.client.messages.stream.call_count, 2)

    async def test_ha_query_also_uses_completion_tone(self):
        self.ha[0].append({"name": "homeassistant__GetLiveContext", "input_schema": {"type": "object"}})
        await self.brain.start()
        tool = NS(type="tool_use", name="homeassistant__GetLiveContext", input={}, id="ha")
        final = NS(type="tool_use", name=module.VOICE_REPLY, input={"answer": "客厅灯开着。"}, id="voice")
        self.client.messages.stream = Mock(side_effect=[
            self.fake_stream(reply([tool], "tool_use")),
            self.fake_stream(reply([final], "tool_use"), self.speech_events('{"answer":"客厅灯开着。"}')),
        ])
        emit, signal = AsyncMock(), AsyncMock()
        await self.brain.ask("查一下状态", on_text=emit, on_signal=signal)
        emit.assert_not_called()
        signal.assert_awaited_once_with("ha_done")

    async def test_ha_query_then_action_does_not_stop_after_query(self):
        self.ha[0].append({"name": "homeassistant__GetLiveContext", "input_schema": {"type": "object"}})
        await self.brain.start()
        query = NS(type="tool_use", name="homeassistant__GetLiveContext", input={}, id="query")
        action = NS(type="tool_use", name="intent__HassTurnOn", input={"name": "客厅灯"}, id="on")
        self.client.messages.stream = Mock(side_effect=[
            self.fake_stream(reply([query], "tool_use")), self.fake_stream(reply([action], "tool_use")),
            self.final_stream(),
        ])
        emit, signal = AsyncMock(), AsyncMock()
        await self.brain.ask("先查询再开灯", on_text=emit, on_signal=signal)
        self.assertEqual(self.executor.await_count, 2)
        emit.assert_not_called()
        signal.assert_awaited_once_with("ha_done")

    async def test_ha_failure_does_not_play_success_tone(self):
        await self.brain.start()
        self.executor.return_value = ExecutorResult(ok=False, result="失败")
        tool = NS(type="tool_use", name="intent__HassTurnOn", input={"name": "客厅灯"}, id="ha")
        final = NS(type="tool_use", name=module.VOICE_REPLY, input={"answer": "开灯失败。"}, id="voice")
        self.client.messages.stream = Mock(side_effect=[
            self.fake_stream(reply([tool], "tool_use")),
            self.fake_stream(reply([final], "tool_use")),
        ])
        emit, signal = AsyncMock(), AsyncMock()
        await self.brain.ask("开灯", on_text=emit, on_signal=signal)
        signal.assert_not_called()
        emit.assert_awaited_once_with("开灯失败。")

    async def test_three_device_tasks_execute_before_single_completion_tone(self):
        self.ha[0].append({"name": "intent__HassTurnOff", "input_schema": {"type": "object"}})
        await self.brain.start()
        expected = [("intent__HassTurnOff", {"name": "客厅灯"}),
                    ("intent__HassTurnOff", {"name": "厨房灯"}),
                    ("intent__HassTurnOn", {"name": "卧室灯"})]
        self.client.messages.stream = Mock(side_effect=[
            *[self.fake_stream(reply([NS(type="tool_use", name=name, input=args, id=f"c{i}")], "tool_use"))
              for i, (name, args) in enumerate(expected)], self.final_stream(),
        ])
        async def completed(name):
            self.assertEqual(name, "ha_done")
            self.assertEqual(self.executor.await_count, 3)

        emit, signal = AsyncMock(), AsyncMock(side_effect=completed)
        await self.brain.ask("关客厅灯和厨房灯，再开卧室灯", on_text=emit, on_signal=signal)
        self.assertEqual([(c.args[0], c.args[1]) for c in self.executor.await_args_list], expected)
        emit.assert_not_called()
        signal.assert_awaited_once()
        self.assertEqual(self.client.messages.stream.call_count, 4)

    async def test_device_then_personal_tasks_all_execute_in_same_turn(self):
        self.add_catalog_tools("add_items", "create_reminder")
        await self.brain.start()
        expected = [("intent__HassTurnOn", {"name": "客厅灯"}),
                    ("add_items", {"list_name": "购物清单", "items": ["牛奶"]}),
                    ("create_reminder", {"title": "开会", "local_time": "2026-10-16T08:00:00"})]
        self.client.messages.stream = Mock(side_effect=[
            *[self.fake_stream(reply([NS(type="tool_use", name=name, input=args, id=f"c{i}")], "tool_use"))
              for i, (name, args) in enumerate(expected)], self.final_stream("牛奶已加入，提醒已设好。"),
        ])
        emit = AsyncMock()
        await self.brain.ask("开客厅灯，加牛奶，明早八点提醒开会", on_text=emit)
        self.assertEqual([(c.args[0], c.args[1]) for c in self.executor.await_args_list], expected)
        rows = self.brain.store._db.execute("SELECT turn_id,status FROM operations").fetchall()
        self.assertEqual(len({row["turn_id"] for row in rows}), 1)
        self.assertTrue(all(row["status"] == "succeeded" for row in rows))
        self.assertEqual(self.client.messages.stream.call_count, 4)

    async def test_text_mode_continues_after_device_to_personal_tool(self):
        self.add_catalog_tools("add_items")
        await self.brain.start()
        self.client.messages.create.side_effect = [
            reply([NS(type="tool_use", name="intent__HassTurnOn", input={"name": "客厅灯"}, id="ha")], "tool_use"),
            reply([NS(type="tool_use", name="add_items", input={"list_name": "购物清单", "items": ["牛奶"]}, id="task")], "tool_use"),
            reply([NS(type="text", text="灯已打开，牛奶已加入。")]),
        ]
        self.assertEqual(await self.brain.ask("开灯并加牛奶"), "灯已打开，牛奶已加入。")
        self.assertEqual(self.executor.await_count, 2)

    async def test_voice_style_applies_to_system_and_device_shortcut(self):
        self.ha[0].append({"name": "light__HassLightSet", "input_schema": {"type": "object"}})
        self.brain.style = "voice"
        await self.brain.start()
        self.assertIn(module.VOICE_RULES, self.brain.system[0]["text"])
        self.client.messages.create.side_effect = [reply([
            NS(type="tool_use", name="light__HassLightSet",
               input={"name": "客厅灯", "brightness": 30}, id="light"),
        ], "tool_use"), reply([NS(type="text", text="完成")])]
        self.assertEqual(await self.brain.ask("调到百分之三十"), "已把客厅灯调到百分之三十。")
        self.assertEqual(self.client.messages.create.await_count, 2)

    async def test_text_style_does_not_add_voice_rules(self):
        await self.brain.start()
        self.assertNotIn(module.VOICE_RULES, self.brain.system[0]["text"])

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
