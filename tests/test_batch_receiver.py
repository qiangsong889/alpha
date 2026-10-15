"""新生产语音接线：假 WebSocket、直接声卡回调、真实业务对象。"""

import asyncio
import base64
import json
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace

import numpy as np

from core.task_coordinator import TaskCoordinator
from integrations.operations import OperationStore
from integrations.tool_gateway import ToolGateway, ExecutorResult
from voice.audio import StreamingPlayer, SpeechActivity, SpeechCandidate
from voice.batch_receiver import receiver_type
from voice.qwen_realtime import QwenReceiver, CostTracker, claims_completion


class BatchReceiverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.remote = AsyncMock(return_value=ExecutorResult(True, "成功"))
        self.gateway = ToolGateway(self.store, self.remote)
        self.flow = TaskCoordinator(self.gateway, window_s=.01, incomplete_s=.04)
        self.addAsyncCleanup(self.flow.close)
        asyncio.create_task(self.flow.run())
        self.ws = Mock(send=AsyncMock())
        self.player = StreamingPlayer()
        self.player.ready = True
        self.player.latency = 0
        tools = [{"name": name, "input_schema": {"type": "object"}} for name in
                 ("add_items", "intent__HassTurnOff", "create_reminder", "list_items")]
        self.receiver = receiver_type(QwenReceiver, claims_completion=claims_completion)(
            self.ws, self.player, SpeechActivity(16000), CostTracker("singapore"), "session",
            gateway=self.gateway, workflow=self.flow, tools=tools)
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

    async def response(self, response_id, tasks, calls=(), *, text="", audio=False):
        r = self.receiver
        await r.handle({"type": "response.created", "response": {"id": response_id}}, tasks)
        for index, (name, args) in enumerate(calls):
            await r.handle({"type": "response.function_call_arguments.done", "response_id": response_id,
                            "call_id": f"{response_id}-{index}", "name": name,
                            "arguments": json.dumps(args)}, tasks)
        if audio:
            await r.handle({"type": "response.audio.delta", "response_id": response_id,
                            "item_id": response_id + "-audio", "delta": base64.b64encode(b"\x00\x20" * 24).decode()}, tasks)
        output = [{"type": "message", "id": response_id + "-audio", "content": [
            {"type": "audio" if audio else "text", "transcript": text}]}] if text else []
        await r.handle({"type": "response.done", "response": {
            "id": response_id, "status": "completed", "output": output}}, tasks)
        for _ in range(8):
            await asyncio.sleep(0)

    async def prepare(self, tasks, items):
        await self.user("加入购物清单", "u1", tasks)
        await self.response("tools", tasks, [
            ("add_items", {"list_name": "购物清单", "items": items}),
            ("finish_collection", {"complete": True}),
        ])
        await self.response("ack", tasks, text="好的，记下了。", audio=True)

    async def test_single_write_is_registered_then_window_executes_once_with_correct_call_id(self):
        async with asyncio.TaskGroup() as tasks:
            await self.prepare(tasks, ["牛奶"])
            self.assertFalse(self.player.buffers)  # 收集确认语不机械插话。
            await asyncio.sleep(.07)
            self.remote.assert_awaited_once()
            results = [item["item"] for item in self.sent()
                       if item["type"] == "conversation.item.create" and item["item"]["type"] == "function_call_output"]
            self.assertEqual(results[0]["call_id"], "tools-0")
            self.assertIn("尚未执行", results[0]["output"])

    async def test_three_actions_recap_playback_then_confirm_executes_one_api_call(self):
        async with asyncio.TaskGroup() as tasks:
            await self.prepare(tasks, ["牛奶", "鸡蛋", "酱油"])
            await asyncio.sleep(.06)
            await self.receiver.maybe_respond()
            recap = self.receiver.pending_recap
            self.assertIsNotNone(recap)
            self.remote.assert_not_awaited()
            await self.response("recap", tasks, text=recap.text, audio=True)
            self.player.callback(np.zeros((24, 1), dtype=np.float32), 24, None, None)
            await asyncio.sleep(.01)
            self.assertEqual(self.flow.plan.status, "awaiting_confirmation")
            await self.user("对，执行吧", "u2", tasks)
            await self.response("confirm", tasks, [
                ("confirm_plan", {"plan_id": recap.plan_id, "version": recap.version}),
                ("finish_collection", {"complete": True}),
            ])
            await self.response("queued", tasks, text="好的")
            await asyncio.sleep(.05)
            self.remote.assert_awaited_once()

    async def test_recap_tool_call_rejected_and_does_not_execute(self):
        async with asyncio.TaskGroup() as tasks:
            await self.prepare(tasks, ["牛奶", "鸡蛋", "酱油"])
            await asyncio.sleep(.05)
            await self.receiver.maybe_respond()
            recap = self.receiver.pending_recap
            await self.response("bad-recap", tasks, [
                ("confirm_plan", {"plan_id": recap.plan_id, "version": recap.version}),
            ], text=recap.text)
            self.remote.assert_not_awaited()
            outputs = [json.loads(event["item"]["output"]) for event in self.sent()
                       if event["type"] == "conversation.item.create" and event["item"]["type"] == "function_call_output"]
            self.assertTrue(any(item.get("reason") == "recap_only" for item in outputs))

    async def test_old_response_cannot_register_write_for_new_user_turn(self):
        async with asyncio.TaskGroup() as tasks:
            self.flow.speech_started("session:new")
            self.flow.user_turn("session:new")
            self.receiver.response_turns["old"] = "old"
            self.receiver.tool_pending["old"] = 1
            self.receiver.turn_pending["old"] = 1
            await self.receiver.execute_tool({"response_id": "old", "call_id": "old-call", "name": "add_items",
                                              "arguments": '{"list_name":"购物清单","items":["牛奶"]}'})
            self.assertEqual(self.flow.chain.all_tasks, ())
            self.remote.assert_not_awaited()

    async def test_schema_error_does_not_leave_pending_record(self):
        self.receiver.schemas["add_items"] = {"type": "object", "required": ["list_name", "items"]}
        async with asyncio.TaskGroup() as tasks:
            await self.user("加牛奶", "u1", tasks)
            await self.response("bad", tasks, [("add_items", {"items": ["牛奶"]})])
            self.assertEqual(self.flow.chain.all_tasks, ())
            self.remote.assert_not_awaited()

    async def test_two_recap_mismatches_use_local_tts_instead_of_looping(self):
        self.flow.speech_started("u1")
        self.flow.user_turn("u1")
        self.flow.add("u1", "add_items", {"list_name": "购物清单", "items": ["牛奶", "鸡蛋", "酱油"]})
        text = self.flow.plan.recap(datetime.now(self.receiver.timezone))
        self.receiver._recap_transcripts = {"first": "说错了", "second": "又说错了"}
        for key in ("first", "second"):
            self.receiver.recaps[key] = SimpleNamespace(plan_id=self.flow.batch_id, version=self.flow.plan.version,
                                                        text=text, items=[], invalid=False, checked=False)
        with patch.object(self.receiver, "local_recap", AsyncMock()) as local:
            await self.receiver.wait_recap_played("first")
            await self.receiver.wait_recap_played("second")
            local.assert_awaited_once()
        self.remote.assert_not_awaited()

    async def test_local_tts_success_qualifies_playback_without_opening_audio_device(self):
        self.flow.speech_started("u1")
        self.flow.user_turn("u1")
        self.flow.add("u1", "add_items", {"list_name": "购物清单", "items": ["牛奶", "鸡蛋", "酱油"]})
        recap = SimpleNamespace(plan_id=self.flow.batch_id, version=self.flow.plan.version, text="复述", checked=False)
        process = Mock(returncode=0, wait=AsyncMock(return_value=0))
        with patch("voice.batch_receiver.shutil.which", return_value="/usr/bin/say"), \
                patch("voice.batch_receiver.asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            await self.receiver.local_recap(recap)
        self.assertEqual(self.flow.plan.status, "awaiting_confirmation")
        self.assertTrue(recap.checked)

    async def test_mismatched_recap_does_not_qualify_confirmation(self):
        self.flow.speech_started("u1")
        self.flow.user_turn("u1")
        self.flow.add("u1", "add_items", {"list_name": "购物清单", "items": ["牛奶", "鸡蛋", "酱油"]})
        recap = SimpleNamespace(plan_id=self.flow.batch_id, version=self.flow.plan.version, text="正确复述", items=[], invalid=False, checked=False)
        self.receiver.recaps["wrong"] = recap
        self.receiver._recap_transcripts = {"wrong": "完全不同"}
        await self.receiver.wait_recap_played("wrong")
        self.flow.user_turn("u2")
        self.assertEqual((await self.flow.confirm(recap.plan_id, recap.version)).reason, "not_played")

    async def test_false_completion_without_tools_is_not_played_and_requests_correction(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("明天提醒我开会", "u1", tasks)
            await self.response("unsupported", tasks, text="已经设好了。", audio=True)
            self.assertFalse(self.player.buffers)
            self.assertEqual(self.receiver.correction_counts.get("u1"), 1)
            self.remote.assert_not_awaited()

    async def test_cancelled_local_tts_stops_subprocess_and_never_qualifies(self):
        self.flow.speech_started("u1")
        self.flow.user_turn("u1")
        self.flow.add("u1", "add_items", {"list_name": "购物清单", "items": ["牛奶", "鸡蛋", "酱油"]})
        recap = SimpleNamespace(plan_id=self.flow.batch_id, version=self.flow.plan.version, text="复述", checked=False)
        entered, released = asyncio.Event(), asyncio.Event()
        process = Mock(returncode=None)
        async def wait():
            entered.set()
            await released.wait()
            return process.returncode
        def terminate():
            process.returncode = -15
            released.set()
        process.wait = wait
        process.terminate.side_effect = terminate
        with patch("voice.batch_receiver.shutil.which", return_value="/usr/bin/say"), \
                patch("voice.batch_receiver.asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            task = asyncio.create_task(self.receiver.local_recap(recap))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        process.terminate.assert_called_once()
        self.assertNotEqual(self.flow.plan.status, "awaiting_confirmation")
