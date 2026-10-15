"""当前生产语音流程：不需要 finish_collection、计划或播放确认。"""

import asyncio
import base64
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from core.simple_coordinator import SimpleCoordinator
from integrations.operations import OperationStore
from integrations.tool_gateway import ExecutorResult, ToolGateway
from voice.audio import StreamingPlayer, SpeechActivity
from voice.simple_receiver import receiver_type
from voice.qwen_realtime import QwenReceiver, CostTracker, claims_completion, session_config


class SimpleReceiverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.remote = AsyncMock(return_value=ExecutorResult(True, "成功"))
        self.gateway = ToolGateway(self.store, self.remote)
        self.flow = SimpleCoordinator(self.gateway, window_s=.02, incomplete_s=.06)
        self.addAsyncCleanup(self.flow.close)
        self.ws = Mock(send=AsyncMock())
        self.player = StreamingPlayer()
        self.player.ready = True
        self.receiver = receiver_type(QwenReceiver, claims_completion=claims_completion)(
            self.ws, self.player, SpeechActivity(16000), CostTracker("singapore"), "session",
            gateway=self.gateway, workflow=self.flow,
            tools=[{"name": name, "input_schema": {"type": "object"}}
                   for name in ("add_items", "list_items", "intent__HassTurnOff")])
        self.addAsyncCleanup(self.receiver.close_playback_tasks)

    def sent(self):
        return [json.loads(call.args[0]) for call in self.ws.send.call_args_list]

    async def user(self, text, key, tasks):
        for event in (
            {"type": "input_audio_buffer.speech_started", "item_id": key, "audio_start_ms": 0},
            {"type": "input_audio_buffer.speech_stopped", "item_id": key, "audio_end_ms": 1000},
            {"type": "input_audio_buffer.committed", "item_id": key},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": key, "transcript": text},
        ):
            await self.receiver.handle(event, tasks)

    async def response(self, key, tasks, calls=(), text="", audio=False):
        await self.receiver.handle({"type": "response.created", "response": {"id": key}}, tasks)
        for index, (name, args) in enumerate(calls):
            await self.receiver.handle({"type": "response.function_call_arguments.done", "response_id": key,
                "call_id": f"{key}-{index}", "name": name, "arguments": json.dumps(args)}, tasks)
        if audio:
            await self.receiver.handle({"type": "response.audio.delta", "response_id": key, "item_id": key + "-audio",
                "delta": base64.b64encode(b"\x00\x20" * 24).decode()}, tasks)
        output = [{"type": "message", "id": key + "-audio", "content": [
            {"type": "audio" if audio else "text", "transcript": text}]}] if text else []
        await self.receiver.handle({"type": "response.done", "response": {
            "id": key, "status": "completed", "output": output}}, tasks)
        for _ in range(8):
            await asyncio.sleep(0)

    def outputs(self):
        return [(event["item"]["call_id"], json.loads(event["item"]["output"])) for event in self.sent()
            if event["type"] == "conversation.item.create" and event["item"]["type"] == "function_call_output"]

    async def test_write_executes_without_finish_collection_or_ack_completion(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("加牛奶", "u1", tasks)
            await self.response("add", tasks, [("add_items", {"list_name": "购物清单", "items": ["牛奶"]})])
            await asyncio.sleep(.07)
            self.remote.assert_awaited_once()
            call_id, output = self.outputs()[0]
            self.assertEqual(call_id, "add-0")
            self.assertEqual(output["status"], "pending")
            self.assertIn("尚未执行", output["text"])
            self.assertEqual(self.receiver.recaps, {})

    async def test_light_executes_before_response_done(self):
        entered = asyncio.Event()
        async def remote(*args):
            entered.set()
            return ExecutorResult(True, "成功")
        self.remote.side_effect = remote
        self.flow.window_s = 100
        async with asyncio.TaskGroup() as tasks:
            await self.user("关灯", "u1", tasks)
            await self.receiver.handle({"type": "response.created", "response": {"id": "light"}}, tasks)
            await self.receiver.handle({"type": "response.function_call_arguments.done", "response_id": "light",
                "call_id": "light-call", "name": "intent__HassTurnOff", "arguments": '{"name":"客厅灯"}'}, tasks)
            await asyncio.wait_for(entered.wait(), 1)
            self.assertNotIn("light", self.receiver.finished_responses)
            self.assertFalse(self.flow.has_pending)
            await self.receiver.handle({"type": "response.done", "response": {"id": "light", "status": "completed"}}, tasks)

    async def test_cancel_pending_tool_cancels_waiting_write(self):
        self.flow.window_s = .15
        self.flow.incomplete_s = .2
        async with asyncio.TaskGroup() as tasks:
            await self.user("加牛奶", "u1", tasks)
            await self.response("add", tasks, [("add_items", {"list_name": "购物清单", "items": ["牛奶"]})])
            await self.response("ack", tasks, text="记下了")
            await self.user("算了不加了", "u2", tasks)
            await self.response("cancel", tasks, [("cancel_pending", {})])
            await asyncio.sleep(.2)
            self.remote.assert_not_awaited()
            self.assertIn("已取消", self.outputs()[-1][1]["text"])

    async def test_confirmation_uses_new_input_without_waiting_for_question_audio(self):
        self.remote.return_value = ExecutorResult(True, {"structured_content": {"present": True, "title": "牛奶"}})
        async with asyncio.TaskGroup() as tasks:
            await self.user("删掉原来的牛奶", "u1", tasks)
            await self.response("request", tasks, [("delete_item", {"kind": "task", "list_id": "list", "item_id": "old"})])
            self.assertTrue(self.outputs()[-1][1]["needs_confirmation"])
            await self.response("question", tasks, text="要删除牛奶吗？", audio=True)
            self.assertTrue(self.player.buffers)  # 不调用声卡回调，没有“播放完”的信号。
            await self.user("对，删掉", "u2", tasks)
            self.remote.return_value = ExecutorResult(True, "删除成功")
            await self.response("confirm", tasks, [("delete_item", {
                "kind": "task", "list_id": "list", "item_id": "old", "confirmed": True})])
            self.remote.assert_awaited_with("delete_task", {"list_id": "list", "task_id": "old"})
            self.assertIn("已经删掉了牛奶", self.outputs()[-1][1]["text"])

    async def test_read_results_include_precise_ids_for_delete_target(self):
        self.remote.return_value = ExecutorResult(True, {"text": "购物清单：牛奶", "structured_content": {
            "list_id": "list", "items": [{"task_id": "old", "title": "牛奶"}]}})
        async with asyncio.TaskGroup() as tasks:
            await self.user("购物清单有什么", "u1", tasks)
            await self.response("read", tasks, [("list_items", {"list_name": "购物清单"})])
            self.assertEqual(self.outputs()[-1][1]["data"]["items"][0]["task_id"], "old")

    async def test_old_controls_are_not_exposed_or_executable(self):
        config = session_config("Tina", [])
        self.assertEqual({tool["function"]["name"] for tool in config["tools"]},
                         {"cancel_pending", "undo_recent", "delete_item"})
        async with asyncio.TaskGroup() as tasks:
            await self.user("加牛奶", "u1", tasks)
            await self.response("old-tool", tasks, [("finish_collection", {"complete": True})])
            self.assertTrue(self.outputs()[-1][1]["is_error"])
            self.remote.assert_not_awaited()

    async def test_incomplete_input_asks_through_qwen_without_tts_or_recap(self):
        async with asyncio.TaskGroup() as tasks:
            with patch("asyncio.create_subprocess_exec", AsyncMock()) as subprocess:
                await self.user("还有……", "u1", tasks)
                await self.response("wait", tasks)
                await asyncio.sleep(.08)
                await self.receiver.maybe_respond()
                subprocess.assert_not_awaited()
                self.assertIsNone(self.receiver.pending_recap)
                self.assertTrue(any("你还想补充什么" in str(event) for event in self.sent()))

    async def test_reconnect_prompts_once_without_executing_pending_work(self):
        self.flow.user_turn("old")
        for title in ("牛奶", "鸡蛋"):
            await self.flow.invoke("old", "add_items", {"list_name": "购物清单", "items": [title]})
        # 断线可能发生在说话中；旧连接的 speaking 状态不能阻塞新连接提示。
        self.flow.speech_started("interrupted")
        self.flow.on_reconnect()
        await self.receiver.maybe_respond()
        self.assertTrue(any("还有 2 项操作没执行，要继续吗" in str(event) for event in self.sent()))
        self.assertEqual(sum(event["type"] == "response.create" for event in self.sent()), 1)
        async with asyncio.TaskGroup() as tasks:
            await self.response("reconnected", tasks, text="刚才断线了，还有两项操作没执行，要继续吗？")
            await self.receiver.maybe_respond()
            await asyncio.sleep(.05)
            self.remote.assert_not_awaited()
            self.assertEqual(sum(event["type"] == "response.create" for event in self.sent()), 1)
            await self.user("继续执行", "continue", tasks)
            await self.response("continue-response", tasks)
            await asyncio.sleep(.05)
            self.assertEqual(self.remote.await_count, 2)

    async def test_reconnect_without_pending_work_does_not_prompt(self):
        self.flow.on_reconnect()
        await self.receiver.maybe_respond()
        self.assertFalse(any(event["type"] == "response.create" for event in self.sent()))
