"""Offline regression checks: no microphone, network, or paid model calls."""

import asyncio
import base64
import json
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from mcp.types import CallToolResult, TextContent

from voice.audio import SpeechActivity, SpeechCandidate, StreamingPlayer, is_meaningful
import voice.qwen_realtime as qwen


class AudioTests(unittest.TestCase):
    def test_next_item_preserves_unplayed_preamble_and_frame_offset(self):
        player = StreamingPlayer()
        player.ready = True
        player.append(np.full(600, 16384, dtype="<i2").tobytes(), "preamble", 0)
        first = np.zeros((240, 1), dtype=np.float32)
        player.callback(first, 240, None, None)
        player.append(np.full(480, 8192, dtype="<i2").tobytes(), "result", 0)
        self.assertEqual(player.offset, 240)
        self.assertEqual(player.item_id, "preamble")
        remaining = np.zeros((840, 1), dtype=np.float32)
        player.callback(remaining, 840, None, None)
        np.testing.assert_allclose(remaining[:360], .5)
        np.testing.assert_allclose(remaining[360:], .25)
        self.assertEqual(player.item_id, "result")
        self.assertEqual(player.played_frames, 480)
        self.assertFalse(player.buffers)
        self.assertFalse(player._buffer_items)

    def test_actual_interruption_discards_all_queued_items(self):
        player = StreamingPlayer()
        player.ready = True
        player.append(b"\x00\x40" * 600, "preamble", 0)
        player.append(b"\x00\x20" * 600, "result", 0)
        player.pause()
        self.assertEqual(player.clear(), ("preamble", 0, 0))
        self.assertFalse(player.buffers)
        self.assertFalse(player._buffer_items)
        output = np.ones((240, 1), dtype=np.float32)
        player.callback(output, 240, None, None)
        np.testing.assert_array_equal(output, 0)
        player.append(b"\x00\x20" * 240, "new", 0)
        player.callback(output, 240, None, None)
        np.testing.assert_allclose(output, .25)

    def test_clear_before_next_item_starts_reports_zero_heard_for_that_item(self):
        player = StreamingPlayer()
        player.ready = True
        player.append(b"\x00\x40" * 240, "preamble", 0)
        player.callback(np.zeros((240, 1), dtype=np.float32), 240, None, None)
        player.append(b"\x00\x20" * 240, "result", 1)
        self.assertEqual(player.clear(), ("result", 1, 0))

    def test_pause_preserves_audio_and_resume_continues(self):
        player = StreamingPlayer()
        player.ready = True
        player.append(np.full(2400, 16384, dtype="<i2").tobytes(), "answer", 0)
        output = np.zeros((240, 1), dtype=np.float32)
        player.duck(True)
        player.callback(output, 240, None, None)
        np.testing.assert_allclose(output, .1)
        player.pause()
        player.callback(output, 240, None, None)
        self.assertEqual(player.played_frames, 240)
        self.assertEqual(len(player.buffers), 1)
        np.testing.assert_array_equal(output, 0)
        player.resume()
        player.callback(output, 240, None, None)
        np.testing.assert_allclose(output, .5)
        self.assertEqual(player.clear(), ("answer", 0, 20))
        self.assertFalse(player.buffers)

    def test_vad_both_rates_and_short_gap(self):
        for rate in (16000, 24000):
            activity = SpeechActivity(rate)
            activity.vad = Mock()
            activity.vad.is_speech.return_value = True
            frame = np.zeros(rate // 50, dtype="<i2").tobytes()
            for _ in range(20):
                activity.observe(frame)
            self.assertEqual(activity.voiced_ms, 400)
            self.assertEqual(len(activity.vad.is_speech.call_args.args[0]), 640)
            activity.vad.is_speech.return_value = False
            for _ in range(4):
                activity.observe(frame)
            self.assertEqual(activity.voiced_ms, 400)
            activity.observe(frame)
            self.assertEqual(activity.voiced_ms, 0)

    def test_meaningful(self):
        for noise in ("", "。", "咳", "嗯，呃", "cough"):
            self.assertFalse(is_meaningful(noise))
        for text in ("好", "是", "等一下", "关客厅灯"):
            self.assertTrue(is_meaningful(text))


class QwenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ws = Mock()
        self.ws.send = AsyncMock()
        self.player = StreamingPlayer()
        self.player.ready = True
        self.activity = SpeechActivity(16000)
        self.receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                          qwen.CostTracker("singapore"), "session")

    def sent(self):
        return [json.loads(call.args[0]) for call in self.ws.send.call_args_list]

    async def emit(self, kind, **fields):
        await self.receiver.handle({"type": kind, **fields}, None)

    async def test_tool_result_audio_does_not_cut_off_earlier_response_audio(self):
        for response_id, item_id, pcm in (("first", "preamble", b"\x00\x40" * 240),
                                           ("second", "result", b"\x00\x20" * 240)):
            self.receiver.response_pending = True
            await self.emit("response.created", response={"id": response_id})
            await self.emit("response.audio.delta", response_id=response_id, item_id=item_id,
                            delta=base64.b64encode(pcm).decode())
            await self.emit("response.done", response={"id": response_id, "status": "completed"})
        output = np.zeros((480, 1), dtype=np.float32)
        self.player.callback(output, 480, None, None)
        np.testing.assert_allclose(output[:240], .5)
        np.testing.assert_allclose(output[240:], .25)

    async def test_noise_resumes_and_deletes_without_cancelling(self):
        target = SpeechCandidate("noise", 0, committed=True)
        self.receiver.candidate = target
        self.receiver.current_response = "old"
        self.player.append(b"\x00\x40" * 2400, "answer", 0)
        self.player.duck(True)
        self.activity.voiced_ms = 400
        self.receiver.pause_candidate(target)
        self.assertTrue(self.player.paused)
        await self.emit("conversation.item.input_audio_transcription.completed",
                        item_id="noise", transcript="咳")
        self.assertFalse(self.player.paused)
        self.assertEqual(self.player.gain, 1)
        self.assertTrue(self.player.buffers)
        self.assertEqual([event["type"] for event in self.sent()], ["conversation.item.delete"])

    async def test_meaningful_cancels_truncates_then_waits_for_old_done(self):
        self.receiver.candidate = SpeechCandidate("question", 0, committed=True, stopped=True)
        self.receiver.current_response = "old"
        self.player.append(b"\x00\x40" * 2400, "answer", 0)
        self.player.callback(np.zeros((480, 1), dtype=np.float32), 480, None, None)
        await self.emit("conversation.item.input_audio_transcription.completed",
                        item_id="question", transcript="等一下")
        events = self.sent()
        self.assertEqual([event["type"] for event in events],
                         ["response.cancel", "conversation.item.truncate"])
        self.assertNotIn("response_id", events[0])
        self.assertEqual(events[1]["audio_end_ms"], 20)
        await self.emit("response.audio.delta", response_id="old", item_id="answer",
                        delta=base64.b64encode(b"\x00\x40" * 240).decode())
        self.assertFalse(self.player.buffers)
        await self.emit("response.done", response={"id": "old", "status": "cancelled"})
        self.assertEqual(self.sent()[-1]["type"], "response.create")
        await self.receiver.maybe_respond()
        self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)

    async def test_transcript_before_commit_does_not_answer_early(self):
        self.receiver.candidate = SpeechCandidate("question", 0, stopped=True)
        await self.emit("conversation.item.input_audio_transcription.completed",
                        item_id="question", transcript="你好")
        self.assertEqual(self.sent(), [])
        await self.emit("input_audio_buffer.committed", item_id="question")
        self.assertEqual(self.sent()[-1]["type"], "response.create")
        await self.emit("response.created", response={"id": "new"})
        self.assertEqual(len(self.sent()), 2)

    async def test_tool_call_runs_in_background_and_preserves_call_id(self):
        worker = Mock()
        entered, release = asyncio.Event(), asyncio.Event()

        async def call(name, arguments):
            entered.set()
            await release.wait()
            return CallToolResult(content=[TextContent(type="text", text="已完成：牛奶")])

        worker.call = AsyncMock(side_effect=call)
        tool = {"name": "complete_items", "input_schema": {"type": "object"}}
        receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker, [tool])
        event = {"type": "response.function_call_arguments.done", "response_id": "r1",
                 "name": "complete_items", "call_id": "call_milk",
                 "arguments": '{"list_name":"购物清单","titles":["牛奶"]}'}
        async with asyncio.TaskGroup() as tasks:
            await receiver.handle(event, tasks)
            await asyncio.wait_for(entered.wait(), 1)
            # worker 尚未完成时，接收循环仍能处理其他事件。
            await receiver.handle({"type": "response.done", "response": {
                "id": "r1", "status": "completed", "output": [{
                    "type": "function_call", "name": event["name"], "call_id": event["call_id"],
                    "arguments": event["arguments"],
                }],
            }}, tasks)
            self.assertEqual(self.sent(), [])
            release.set()
        worker.call.assert_awaited_once_with("complete_items", {"list_name": "购物清单", "titles": ["牛奶"]})
        events = self.sent()
        result = events[0]["item"]
        self.assertEqual(result["type"], "function_call_output")
        self.assertEqual(result["call_id"], "call_milk")
        self.assertEqual(json.loads(result["output"]), {"is_error": False, "text": "已完成：牛奶"})
        self.assertEqual(events[-1]["type"], "response.create")
        clock = events[-2]["item"]
        self.assertEqual(clock["role"], "user")
        self.assertIn("[当前时间：", clock["content"][0]["text"])
        self.assertIn("America/Los_Angeles", clock["content"][0]["text"])

    async def test_tool_errors_are_returned_without_executing_forbidden_calls(self):
        for name, arguments, message in (("delete_all", "{}", "白名单"),
                                         ("list_items", "[]", "JSON 对象"),
                                         ("list_items", "{", "JSONDecodeError")):
            worker = Mock(call=AsyncMock())
            self.ws.send.reset_mock()
            receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                         self.receiver.costs, "session", worker,
                                         [{"name": "list_items", "input_schema": {}}])
            async with asyncio.TaskGroup() as tasks:
                await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                       "name": name, "call_id": "call", "arguments": arguments}, tasks)
            worker.call.assert_not_awaited()
            output = json.loads(self.sent()[0]["item"]["output"])
            self.assertTrue(output["is_error"])
            self.assertIn(message, output["text"])

    async def test_tool_mcp_error_is_sent_to_model(self):
        worker = Mock(call=AsyncMock(return_value=CallToolResult(
            is_error=True, content=[TextContent(type="text", text="找不到清单")],
        )))
        receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker,
                                     [{"name": "list_items", "input_schema": {}}])
        async with asyncio.TaskGroup() as tasks:
            await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                   "name": "list_items", "call_id": "call", "arguments": "{}"}, tasks)
        self.assertEqual(json.loads(self.sent()[0]["item"]["output"]),
                         {"is_error": True, "text": "找不到清单"})

    async def test_ha_calls_route_to_http_instead_of_personal_worker(self):
        worker = Mock(call=AsyncMock())
        http_client = Mock()
        tools = [{"name": name, "input_schema": {}} for name in qwen.HA_DEVICE_TOOLS]
        receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker, tools,
                                     ha_url="http://ha/api/mcp", http_client=http_client)
        result = CallToolResult(content=[TextContent(type="text", text="操作成功")])
        with patch.object(qwen, "call_ha_tool", AsyncMock(return_value=result)) as call_ha:
            async with asyncio.TaskGroup() as tasks:
                for name in sorted(qwen.HA_DEVICE_TOOLS):
                    await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                           "name": name, "call_id": name, "arguments": '{"name":"客厅灯"}'}, tasks)
            for name in qwen.HA_DEVICE_TOOLS:
                call_ha.assert_any_await(name, {"name": "客厅灯"}, url="http://ha/api/mcp", http_client=http_client)
            self.assertEqual(call_ha.await_count, 4)
        worker.call.assert_not_awaited()
        self.assertEqual({e["item"]["call_id"] for e in self.sent()}, qwen.HA_DEVICE_TOOLS)

    async def test_ha_error_and_undiscovered_tool_are_returned_to_model(self):
        tool = {"name": "intent__HassTurnOn", "input_schema": {}}
        for tools in ([tool], []):
            self.ws.send.reset_mock()
            receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                         self.receiver.costs, "session", tools=tools,
                                         ha_url="http://ha/api/mcp", http_client=Mock())
            with patch.object(qwen, "call_ha_tool", AsyncMock(side_effect=OSError("HA 正在重启"))) as call_ha:
                async with asyncio.TaskGroup() as tasks:
                    await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                           "name": tool["name"], "call_id": "ha-call", "arguments": "{}"}, tasks)
                if tools:
                    call_ha.assert_awaited_once()
                else:
                    call_ha.assert_not_awaited()
            output = self.sent()[0]["item"]
            self.assertEqual(output["call_id"], "ha-call")
            result = json.loads(output["output"])
            self.assertTrue(result["is_error"])
            self.assertIn("HA 正在重启" if tools else "白名单", result["text"])

    async def test_mixed_ha_and_personal_tools_continue_once(self):
        result = CallToolResult(content=[TextContent(type="text", text="完成")])
        worker = Mock(call=AsyncMock(return_value=result))
        tools = [{"name": name, "input_schema": {}} for name in ("list_items", "intent__HassTurnOff")]
        receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker, tools,
                                     ha_url="http://ha/api/mcp", http_client=Mock())
        with patch.object(qwen, "call_ha_tool", AsyncMock(return_value=result)) as call_ha:
            async with asyncio.TaskGroup() as tasks:
                for tool in tools:
                    await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                           "name": tool["name"], "call_id": tool["name"], "arguments": "{}"}, tasks)
                await receiver.handle({"type": "response.done", "response": {
                    "id": "r", "status": "completed",
                }}, tasks)
            call_ha.assert_awaited_once()
        worker.call.assert_awaited_once_with("list_items", {})
        self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)

    async def test_successful_device_actions_drain_silently_and_next_turn_still_works(self):
        names = ["intent__HassTurnOn", "intent__HassTurnOff", "light__HassLightSet"]
        result = CallToolResult(content=[TextContent(type="text", text=json.dumps({
            "response": {"response_type": "action_done", "data": {}},
        }))])
        for tools_finish_first in (True, False):
            self.ws.send.reset_mock()
            receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                         self.receiver.costs, "session",
                                         tools=[{"name": name, "input_schema": {}} for name in names],
                                         ha_url="http://ha/api/mcp", http_client=Mock())
            with patch.object(qwen, "call_ha_tool", AsyncMock(return_value=result)):
                async with asyncio.TaskGroup() as tasks:
                    for name in names:
                        await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                               "name": name, "call_id": name, "arguments": "{}"}, tasks)
                    if tools_finish_first:
                        await asyncio.sleep(0)
                    await receiver.handle({"type": "response.done", "response": {
                        "id": "r", "status": "completed",
                    }}, tasks)
            outputs = [event["item"] for event in self.sent() if event["type"] == "conversation.item.create"]
            self.assertEqual(len(outputs), 3)
            self.assertTrue(all(item["type"] == "function_call_output" for item in outputs))
            self.assertTrue(receiver.response_pending)
            self.assertTrue(receiver.silent_response_pending)
            self.assertEqual(self.sent()[-1]["type"], "response.create")
            await receiver.handle({"type": "response.created", "response": {"id": "silent"}}, None)
            await receiver.handle({"type": "response.audio.delta", "response_id": "silent", "item_id": "answer",
                                   "delta": base64.b64encode(b"\x00\x40" * 240).decode()}, None)
            self.assertFalse(self.player.buffers)
            await receiver.handle({"type": "response.done", "response": {
                "id": "silent", "status": "completed",
            }}, None)
            self.assertFalse(receiver.response_pending)
            # 静默结束只跳过本轮确认，不影响下一句指令。
            receiver.candidate = SpeechCandidate("next", 100, committed=True, meaningful=True)
            await receiver.maybe_respond()
            self.assertEqual(self.sent()[-1]["type"], "response.create")

    async def test_twelve_consecutive_silent_actions_complete_each_followup(self):
        name = "intent__HassTurnOff"
        result = CallToolResult(content=[], structured_content={"response_type": "action_done"})
        receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", tools=[{"name": name, "input_schema": {}}],
                                     ha_url="http://ha/api/mcp", http_client=Mock())
        with patch.object(qwen, "call_ha_tool", AsyncMock(return_value=result)) as call_ha:
            for index in range(12):
                response_id = f"action-{index}"
                async with asyncio.TaskGroup() as tasks:
                    await receiver.handle({"type": "response.function_call_arguments.done", "response_id": response_id,
                                           "name": name, "call_id": f"call-{index}", "arguments": "{}"}, tasks)
                    await receiver.handle({"type": "response.done", "response": {
                        "id": response_id, "status": "completed",
                    }}, tasks)
                await receiver.handle({"type": "response.created", "response": {"id": f"silent-{index}"}}, None)
                await receiver.handle({"type": "response.done", "response": {
                    "id": f"silent-{index}", "status": "completed",
                }}, None)
                self.assertFalse(receiver.response_pending)
                self.assertIsNone(receiver.current_response)
            self.assertEqual(call_ha.await_count, 12)
        self.assertEqual(sum(event["type"] == "response.create" for event in self.sent()), 12)
        self.assertFalse(self.player.buffers)

    async def test_silent_followup_does_not_repeat_device_write(self):
        receiver = self.receiver
        receiver.response_pending = True
        receiver.silent_response_pending = True
        await self.emit("response.created", response={"id": "silent"})
        event = {"type": "response.function_call_arguments.done", "response_id": "silent",
                 "name": "intent__HassTurnOff", "call_id": "duplicate", "arguments": "{}"}
        with patch.object(qwen, "call_ha_tool", AsyncMock()) as call_ha:
            async with asyncio.TaskGroup() as tasks:
                await receiver.handle(event, tasks)
            self.assertNotIn("duplicate", receiver.tool_calls)
            with self.assertRaises(qwen.SessionEnded):
                await self.emit("response.done", response={"id": "silent", "status": "completed", "output": [{
                    "type": "function_call", "name": event["name"], "call_id": event["call_id"], "arguments": "{}",
                }]})
            call_ha.assert_not_awaited()

    async def test_failed_uncertain_and_query_results_keep_model_response(self):
        cases = [
            ("intent__HassTurnOff", CallToolResult(is_error=True, content=[TextContent(type="text", text="失败")])),
            ("intent__HassTurnOff", CallToolResult(content=[TextContent(type="text", text="操作成功")])),
            ("intent__HassTurnOff", CallToolResult(content=[], structured_content={
                "response_type": "action_done", "data": {"failed": ["light.living_room"]},
            })),
            ("homeassistant__GetLiveContext", CallToolResult(content=[TextContent(type="text", text="客厅灯已关")])),
        ]
        for name, result in cases:
            self.ws.send.reset_mock()
            receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                         self.receiver.costs, "session",
                                         tools=[{"name": name, "input_schema": {}}],
                                         ha_url="http://ha/api/mcp", http_client=Mock())
            with patch.object(qwen, "call_ha_tool", AsyncMock(return_value=result)):
                async with asyncio.TaskGroup() as tasks:
                    await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                           "name": name, "call_id": "call", "arguments": "{}"}, tasks)
                    await receiver.handle({"type": "response.done", "response": {
                        "id": "r", "status": "completed",
                    }}, tasks)
            self.assertEqual(self.sent()[-1]["type"], "response.create")

    async def test_parallel_tools_create_one_continuation(self):
        worker = Mock(call=AsyncMock(return_value=CallToolResult(
            content=[TextContent(type="text", text="完成")],
        )))
        receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker,
                                     [{"name": "list_items", "input_schema": {}}])
        async with asyncio.TaskGroup() as tasks:
            for call_id in ("first", "second"):
                await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                       "name": "list_items", "call_id": call_id, "arguments": "{}"}, tasks)
            await receiver.handle({"type": "response.done", "response": {
                "id": "r", "status": "completed",
            }}, tasks)
        outputs = [e["item"] for e in self.sent() if e.get("item", {}).get("type") == "function_call_output"]
        self.assertEqual({item["call_id"] for item in outputs}, {"first", "second"})
        self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)

    async def test_worker_exception_is_returned_with_call_id(self):
        worker = Mock(call=AsyncMock(side_effect=TimeoutError("服务超时")))
        receiver = qwen.QwenReceiver(self.ws, self.player, self.activity,
                                     self.receiver.costs, "session", worker,
                                     [{"name": "add_items", "input_schema": {}}])
        async with asyncio.TaskGroup() as tasks:
            await receiver.handle({"type": "response.function_call_arguments.done", "response_id": "r",
                                   "name": "add_items", "call_id": "failed", "arguments": "{}"}, tasks)
        result = self.sent()[0]["item"]
        self.assertEqual(result["call_id"], "failed")
        output = json.loads(result["output"])
        self.assertTrue(output["is_error"])
        self.assertIn("服务超时", output["text"])

    async def test_new_meaningful_turn_suppresses_old_tool_continuation(self):
        old, new = SpeechCandidate("old", 0), SpeechCandidate("new", 100)
        receiver = self.receiver
        receiver.candidate = new
        receiver.last_meaningful_candidate = new
        receiver.tool_candidates["old-response"] = old
        receiver.tool_pending["old-response"] = 0
        receiver.finished_responses["old-response"] = "completed"
        await receiver.maybe_respond()
        self.assertEqual(self.sent(), [])

    async def test_pending_answer_cancelled_after_new_meaningful_input(self):
        self.receiver.candidate = SpeechCandidate("first", 0, committed=True, meaningful=True)
        await self.receiver.maybe_respond()
        self.receiver.candidate = SpeechCandidate("second", 100, committed=True, stopped=True)
        await self.emit("conversation.item.input_audio_transcription.completed",
                        item_id="second", transcript="换个问题")
        await self.emit("response.created", response={"id": "first_answer"})
        self.assertEqual(self.sent()[-1]["type"], "response.cancel")

    async def test_unrequested_response_is_reported(self):
        with self.assertRaises(qwen.QwenProtocolError):
            await self.emit("response.created", response={"id": "automatic"})

    async def test_reconnect_reapplies_session_configuration(self):
        sockets = []
        tools = [{"name": "intent__HassTurnOn", "input_schema": {}}]
        http_client = Mock()

        class Socket:
            def __init__(self):
                self.send = AsyncMock()
                self.recv = AsyncMock(side_effect=[
                    json.dumps({"type": "session.created", "session": {"id": str(len(sockets))}}),
                    json.dumps({"type": "session.updated", "session": qwen.session_config("Tina", tools, "客厅灯")}),
                ])

            async def __aenter__(self):
                sockets.append(self)
                return self

            async def __aexit__(self, *args):
                return False

        async def receive(receiver):
            self.assertIs(receiver.http_client, http_client)
            self.assertEqual(receiver.ha_url, "http://ha/api/mcp")
            self.assertIn("intent__HassTurnOn", receiver.tool_names)
            if len(sockets) == 1:
                qwen.raise_api_error({"message": "Turn queue overflow: capacity=8"})
            second_session.set()
            await asyncio.Future()

        second_session = asyncio.Event()
        original_sleep = asyncio.sleep

        async def quick_retry(seconds):
            await original_sleep(0 if seconds == 1 else seconds)

        with patch.object(qwen, "connect", side_effect=lambda *a, **kw: Socket()), \
                patch.object(qwen, "send_microphone", side_effect=lambda *a: asyncio.Future()), \
                patch.object(qwen.QwenReceiver, "run", receive), \
                patch.object(qwen.asyncio, "sleep", quick_retry):
            task = asyncio.create_task(qwen.reconnect_loop(
                "wss://example", "test", self.player, self.receiver.costs, "Tina", "singapore",
                tools=tools, assist_prompt="客厅灯", ha_url="http://ha/api/mcp", http_client=http_client))
            await asyncio.wait_for(second_session.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(sockets), 2)
        for socket in sockets:
            event = json.loads(socket.send.call_args.args[0])
            self.assertEqual(event["type"], "session.update")
            self.assertEqual(event["session"]["tools"][0]["function"]["name"], "intent__HassTurnOn")
            self.assertIn("客厅灯", event["session"]["instructions"])

    async def test_main_loads_both_catalogs_and_closes_clients(self):
        personal = {"name": "list_items", "input_schema": {}}
        ha = {"name": "intent__HassTurnOn", "input_schema": {}}
        prompt = [{"type": "text", "text": "基础规则\nHome Assistant Assist prompt：\n客厅灯"}]
        for configured, available in ((True, True), (True, False), (False, False)):
            worker = Mock(run=AsyncMock())
            environment = {"DASHSCOPE_API_KEY": "test", "QWEN_WORKSPACE_ID": "test"}
            if configured:
                environment.update(HA_URL="http://ha", HA_TOKEN="test")
            catalog = ([ha], prompt) if available else None
            with patch.dict(qwen.os.environ, environment, clear=True), \
                    patch.object(qwen, "load_dotenv"), \
                    patch.object(qwen, "PersonalWorker", return_value=worker), \
                    patch.object(qwen, "stop_worker", AsyncMock()) as stop, \
                    patch.object(qwen, "load_personal_catalog", AsyncMock(return_value=[personal])), \
                    patch.object(qwen, "load_ha_catalog", AsyncMock(return_value=catalog)) as load_ha, \
                    patch.object(qwen.StreamingPlayer, "run", AsyncMock()), \
                    patch.object(qwen, "reconnect_loop", AsyncMock(side_effect=qwen.QwenProtocolError("test stop"))) as reconnect:
                with self.assertRaises(ExceptionGroup):
                    await qwen.main()
                if configured:
                    load_ha.assert_awaited_once()
                else:
                    load_ha.assert_not_awaited()
                args, kwargs = reconnect.await_args
                self.assertEqual([t["name"] for t in args[8]],
                                 ["list_items", "intent__HassTurnOn"] if available else ["list_items"])
                self.assertEqual(kwargs["assist_prompt"], "客厅灯" if available else "")
                self.assertTrue(kwargs["http_client"].is_closed)
                stop.assert_awaited_once()


class CostAndProtocolTests(unittest.TestCase):
    def test_ha_tools_prompt_and_service_specific_fallback(self):
        tools = [{"name": "intent__HassTurnOn", "description": "开灯", "input_schema": {}}]
        config = qwen.session_config("Tina", tools, "客厅灯属于客厅")
        self.assertEqual(config["tools"][0]["function"]["name"], "intent__HassTurnOn")
        self.assertIn("客厅灯属于客厅", config["instructions"])
        self.assertIn("个人服务暂时不可用", config["instructions"])
        self.assertNotIn("家居设备暂时不可用", config["instructions"])
        qwen.verify_session(config, tools)
        with self.assertRaises(qwen.QwenProtocolError):
            qwen.verify_session({**config, "tools": []}, tools)
        personal_only = qwen.session_config("Tina", [{"name": "list_items", "input_schema": {}}])
        self.assertIn("家居设备暂时不可用", personal_only["instructions"])
        self.assertNotIn("个人服务暂时不可用", personal_only["instructions"])

    def test_tools_use_qwen_schema_and_are_checked_on_session_update(self):
        tools = [{"name": "list_items", "description": "列出条目", "input_schema": {
            "type": "object", "properties": {"list_name": {"type": "string"}},
        }}]
        config = qwen.session_config("Tina", tools)
        self.assertEqual(config["tools"][0], {"type": "function", "function": {
            "name": "list_items", "description": "列出条目", "parameters": tools[0]["input_schema"],
        }})
        qwen.verify_session(config, tools)
        with self.assertRaises(qwen.QwenProtocolError):
            qwen.verify_session({**config, "tools": []}, tools)
        self.assertNotIn("[当前时间：", config["instructions"])

    def test_price_detail_and_deduplication(self):
        response = {"id": "reply", "usage": {
            "input_tokens_details": {"text_tokens": 960, "audio_tokens": 168},
            "output_tokens_details": {"text_tokens": 16, "audio_tokens": 60},
        }}
        costs = qwen.CostTracker("singapore")
        self.assertEqual(costs.record("s1", response), Decimal("0.00050044"))
        self.assertIsNone(costs.record("s1", response))
        costs.record("s2", response)
        self.assertEqual(costs.total, Decimal("0.00100088"))
        self.assertIsNone(costs.record("s2", {"id": "missing"}))
        self.assertEqual(costs.unpriced, 1)

    def test_session_validation_and_error_classification(self):
        qwen.verify_session(qwen.session_config("Tina"))
        with self.assertRaises(qwen.QwenProtocolError):
            qwen.verify_session({"turn_detection": {"type": "semantic_vad"}})
        with self.assertRaises(qwen.SessionEnded):
            qwen.raise_api_error({"code": "session_expired"})
        with self.assertRaises(qwen.SessionEnded) as overflow:
            qwen.raise_api_error({"message": "Turn queue overflow: capacity=8"})
        self.assertTrue(qwen.reconnectable(overflow.exception))
        with self.assertRaises(qwen.QwenProtocolError):
            qwen.raise_api_error({"code": "invalid_value", "message": "invalid session parameter"})
        self.assertTrue(qwen.reconnectable(ExceptionGroup("network", [OSError("offline")])) )
        self.assertFalse(qwen.reconnectable(ExceptionGroup("bad config", [qwen.QwenProtocolError("bad")])) )


if __name__ == "__main__":
    unittest.main()
