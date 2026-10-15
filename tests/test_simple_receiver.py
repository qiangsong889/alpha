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
from voice.simple_receiver import receiver_type, shopping_request
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

    async def test_buy_single_item_executes_without_model_tool_selection(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("买牛奶。", "u1", tasks)
            await asyncio.sleep(.06)
            self.remote.assert_awaited_once_with("add_items", {"list_name": "购物清单", "items": ["牛奶"]})
            self.assertFalse(any(e["type"] == "response.create" for e in self.sent()))
            await self.receiver.maybe_respond()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)
            await self.response("report", tasks, text="牛奶加好了。")

    async def test_shopping_query_always_reads_fresh_tool_results(self):
        self.remote.return_value = ExecutorResult(True, {"text": "购物清单：酱油", "structured_content": {
            "list_id": "list", "items": [{"task_id": "soy", "title": "酱油"}]}})
        async with asyncio.TaskGroup() as tasks:
            await self.user("现在购物清单里有什么？", "u1", tasks)
            for _ in range(8):
                await asyncio.sleep(0)
            self.remote.assert_awaited_once_with("list_items", {"list_name": "购物清单"})
            self.assertTrue(any("本次实际返回的数据" in str(e) and "酱油" in str(e) for e in self.sent()))
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)
            await self.response("answer", tasks, text="清单里有酱油。")

    async def test_slow_local_query_does_not_block_speech_events_or_answer_stale_input(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def remote(*args):
            entered.set()
            await release.wait()
            return ExecutorResult(True, "旧查询结果")
        self.remote.side_effect = remote
        async with asyncio.TaskGroup() as tasks:
            await asyncio.wait_for(self.user("购物清单里有什么？", "u1", tasks), .2)
            await asyncio.wait_for(entered.wait(), .2)
            await self.receiver.handle({"type": "input_audio_buffer.speech_started", "item_id": "u2", "audio_start_ms": 1200}, tasks)
            self.assertEqual(self.receiver.candidate.item_id, "u2")
            # 有意义的新输入取代正在查询的旧输入。
            self.flow.record_text("session:u2", "等一下")
            self.flow.user_turn("session:u2")
            release.set()
            for _ in range(8):
                await asyncio.sleep(0)
            self.assertFalse(any(e["type"] == "response.create" for e in self.sent()))
            self.receiver.candidate.stopped = True

    def test_shopping_shortcuts_do_not_guess_ambiguous_or_multiple_actions(self):
        for text in ("卖牛奶。", "买牛奶吗？", "买牛奶，然后不要了", "买牛奶还是酱油", "不要买牛奶", "怎么买牛奶"):
            self.assertIsNone(shopping_request(text), text)

    async def test_unavailable_shopping_tool_uses_regular_model_path(self):
        self.receiver.tool_names.discard("list_items")
        async with asyncio.TaskGroup() as tasks:
            await self.user("购物清单里有什么？", "u1", tasks)
            self.remote.assert_not_awaited()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)

    async def test_promise_without_tool_is_suppressed_and_corrected_once_per_user_turn(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("刚才那个不要了", "u1", tasks)
            with patch.object(self.receiver, "log") as log:
                await self.response("promise", tasks, text="行，那我把刚才加的面包从购物清单里撤掉。", audio=True)
                self.assertTrue(any("只说未做" in call.args[0] for call in log.call_args_list))
            self.assertFalse(self.player.buffers)
            self.remote.assert_not_awaited()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 2)
            self.assertTrue(any("请现在根据用户原始指令调用对应工具" in str(e) for e in self.sent()))
            await self.response("still-promise", tasks, text="我这就把面包删掉。", audio=True)
            await self.receiver.maybe_respond()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 3)
            self.assertFalse(self.player.buffers)
            self.assertTrue(any("这次没有执行成功，请再说一次" in str(e) for e in self.sent()))
            await self.response("failure-notice", tasks, text="这次没有执行成功，请再说一次。", audio=True)
            self.assertTrue(self.player.buffers)
            await self.receiver.maybe_respond()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 3)
            await self.user("把牛奶删掉", "u2", tasks)
            await self.response("new-promise", tasks, text="我这就把牛奶删掉。")
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 5)

    async def test_failure_notice_cannot_execute_tools_or_trigger_another_retry(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("删牛奶", "u1", tasks)
            await self.response("promise", tasks, text="我把牛奶删掉。")
            await self.response("retry", tasks, text="我这就把牛奶删掉。")
            await self.response("notice", tasks, [("add_items", {"list_name": "购物清单", "items": ["错误新增"]})],
                                text="这次没有执行成功，请再说一次。")
            self.remote.assert_not_awaited()
            self.assertFalse(self.flow.has_pending)
            self.assertTrue(self.outputs()[-1][1]["is_error"])
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 3)

    async def test_retry_changes_to_false_completion_still_gets_failure_notice(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("删牛奶", "u1", tasks)
            await self.response("promise", tasks, text="我这就把牛奶删掉。")
            await self.response("false-success", tasks, text="已经删除牛奶了。")
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 3)
            self.assertTrue(any("这次没有执行成功，请再说一次" in str(e) for e in self.sent()))
            self.remote.assert_not_awaited()

    async def test_false_addition_and_its_correction_audio_are_both_withheld(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("把面包加到购物清单", "u1", tasks)
            await self.response("false-add", tasks, text="面包已经加到购物清单了。", audio=True)
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 2)
            self.assertFalse(self.player.buffers)
            await self.response("false-again", tasks, text="面包已经加到购物清单了。", audio=True)
            self.assertFalse(self.player.buffers)
            self.remote.assert_not_awaited()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 3)
            await self.response("failure", tasks, text="这次没有执行成功，请再说一次。", audio=True)
            self.assertTrue(self.player.buffers)

    async def test_unchanged_snapshot_is_not_repeated_but_state_changes_are_sent(self):
        def snapshots():
            return [e for e in self.sent() if e["type"] == "conversation.item.create"
                    and "当前操作记录：" in str(e)]
        async with asyncio.TaskGroup() as tasks:
            await self.user("你好", "u1", tasks)
            await self.response("first", tasks, text="你好。")
            self.assertEqual(len(snapshots()), 1)
            await self.user("还在吗", "u2", tasks)
            await self.response("second", tasks, text="在呢。")
            self.assertEqual(len(snapshots()), 1)
            await self.user("加面包", "u3", tasks)
            await self.response("add", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            await asyncio.sleep(.06)
            await self.receiver.maybe_respond()
            self.assertEqual(len(snapshots()), 2)

    async def test_false_readdition_without_tool_is_withheld_and_corrected(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("卖牛奶", "u1", tasks)
            with patch.object(self.receiver, "log") as log:
                await self.response("false-readd", tasks, text="牛奶已经重新加回购物清单了。", audio=True)
                self.assertTrue(any("疑似虚构" in call.args[0] for call in log.call_args_list))
            self.assertFalse(self.player.buffers)
            self.remote.assert_not_awaited()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 2)
            await self.response("clarify", tasks, text="你是想把牛奶加到购物清单吗？", audio=True)
            self.assertTrue(self.player.buffers)
            self.remote.assert_not_awaited()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 2)

    async def test_single_recent_creation_undo_bypasses_model_confirmation(self):
        self.remote.return_value = ExecutorResult(True, {"structured_content": {
            "version": 1, "status": "succeeded", "list_id": "list", "list_name": "购物清单",
            "created": [{"task_id": "new-bread", "title": "面包"}]}})
        async with asyncio.TaskGroup() as tasks:
            await self.user("加面包", "u1", tasks)
            await self.response("add", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            await asyncio.sleep(.06)
            await self.receiver.maybe_respond()
            await self.response("report", tasks, text="面包加好了")
            async def remote(name, args):
                if name == "inspect_task":
                    return ExecutorResult(True, {"structured_content": {"present": True,
                        "title": "面包", "resource": {"id": "new-bread", "title": "面包"}}})
                return ExecutorResult(True, "已删除面包")
            self.remote.side_effect = remote
            before = sum(e["type"] == "response.create" for e in self.sent())
            await self.user("刚才那个不要了", "u2", tasks)
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), before)
            await asyncio.sleep(.06)
            await self.receiver.maybe_respond()
            self.remote.assert_awaited_with("delete_task", {"list_id": "list", "task_id": "new-bread"})
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), before + 1)

    async def test_promise_correction_can_result_in_actual_tool_execution(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("把面包加到清单", "u1", tasks)
            await self.response("promise", tasks, text="我来把面包加到购物清单。")
            await self.response("corrected-tool", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            await asyncio.sleep(.06)
            self.remote.assert_awaited_once()
            self.assertEqual(self.outputs()[0][1]["status"], "pending")

    async def test_local_undo_report_cannot_repeat_undo_or_execute_other_tools(self):
        self.remote.return_value = ExecutorResult(True, {"structured_content": {
            "version": 1, "status": "succeeded", "list_id": "list", "list_name": "购物清单",
            "created": [{"task_id": "new-bread", "title": "面包"}]}})
        async with asyncio.TaskGroup() as tasks:
            await self.user("加面包", "u1", tasks)
            await self.response("add", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            await asyncio.sleep(.06)
            await self.receiver.maybe_respond()
            await self.response("added-report", tasks, text="面包加好了")
            async def remote(name, args):
                if name == "inspect_task":
                    return ExecutorResult(True, {"structured_content": {"present": True,
                        "title": "面包", "resource": {"id": "new-bread", "title": "面包"}}})
                return ExecutorResult(True, "已删除面包")
            self.remote.side_effect = remote
            await self.user("刚才那个不要了", "u2", tasks)
            await asyncio.sleep(.06)
            await self.receiver.maybe_respond()
            before = self.remote.await_count
            with patch.object(self.flow, "undo_recent", AsyncMock()) as undo:
                await self.response("bad-report", tasks, [("undo_recent", {}),
                    ("intent__HassTurnOff", {"name": "客厅灯"})])
                undo.assert_not_awaited()
            self.assertEqual(self.remote.await_count, before)
            self.assertTrue(self.flow._report_pending)
            self.assertIsNotNone(self.receiver.active_report)
            self.assertTrue(self.receiver.report_requested)
            for call_id, output in self.outputs()[-2:]:
                self.assertTrue(output["report_only"])
                self.assertFalse(output["executed"])
                self.assertIn("已删除面包", output["text"])
                self.assertTrue(call_id.startswith("bad-report"))
            await self.response("correct-report", tasks, text="面包已从购物清单里撤掉了。")
            self.assertFalse(self.flow._report_pending)
            self.assertIsNone(self.receiver.active_report)
            self.assertEqual([c.args[0] for c in self.remote.await_args_list].count("delete_task"), 1)

    async def test_failed_batch_report_returns_failure_facts_without_new_write(self):
        self.remote.return_value = ExecutorResult(False, "添加失败")
        async with asyncio.TaskGroup() as tasks:
            await self.user("加面包", "u1", tasks)
            await self.response("add", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            await asyncio.sleep(.06)
            await self.receiver.maybe_respond()
            await self.response("bad-report", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            self.remote.assert_awaited_once()
            self.assertIn("[failed]", self.outputs()[-1][1]["text"])
            self.assertIn("添加失败", self.outputs()[-1][1]["text"])
            await self.response("correct-report", tasks, text="这次添加失败了。")
            self.assertFalse(self.flow._report_pending)

    async def test_report_tool_in_done_output_is_blocked_even_with_invalid_arguments(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("加面包", "u1", tasks)
            await self.response("add", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            await asyncio.sleep(.06)
            await self.receiver.maybe_respond()
            await self.receiver.handle({"type": "response.created", "response": {"id": "report"}}, tasks)
            await self.receiver.handle({"type": "response.done", "response": {
                "id": "report", "status": "completed", "output": [{"type": "function_call",
                    "name": "undo_recent", "call_id": "report-call", "arguments": "not-json"}]}}, tasks)
            for _ in range(8):
                await asyncio.sleep(0)
            self.remote.assert_awaited_once()
            self.assertEqual(self.outputs()[-1][0], "report-call")
            self.assertTrue(self.outputs()[-1][1]["report_only"])
            self.assertTrue(self.receiver.report_requested)
            await self.response("correct-report", tasks, text="面包加好了。")
            self.assertFalse(self.flow._report_pending)

    async def test_report_tools_are_bounded_and_do_not_deadlock_next_batch(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("加面包", "u1", tasks)
            await self.response("add", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            await asyncio.sleep(.06)
            await self.receiver.maybe_respond()
            await self.flow.invoke("session:u1", "add_items", {"list_name": "购物清单", "items": ["牛奶"]})
            self.flow.turn_finished("session:u1")
            await asyncio.sleep(.04)
            await self.response("bad-report-1", tasks, [("undo_recent", {})])
            self.remote.assert_awaited_once()
            with patch.object(self.receiver, "log") as log:
                await self.response("bad-report-2", tasks, [("undo_recent", {})])
                self.assertTrue(any("汇报降级" in call.args[0] for call in log.call_args_list))
            await asyncio.sleep(.04)
            await self.receiver.maybe_respond()
            self.assertEqual(self.remote.await_count, 2)
            await self.response("next-report", tasks, text="牛奶加好了。")
            self.assertFalse(self.flow._report_pending)

    async def test_normal_question_and_negation_do_not_trigger_promise_correction(self):
        async with asyncio.TaskGroup() as tasks:
            for index, text in enumerate(("我在呢。", "要我把牛奶删掉吗？", "我不会删除已有的内容。", "如果你同意，我会删除牛奶。")):
                await self.user("请解释一下", f"u{index}", tasks)
                before = sum(e["type"] == "response.create" for e in self.sent())
                await self.response(f"r{index}", tasks, text=text)
                self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), before)

    async def test_promise_with_tool_in_same_response_does_not_trigger_correction(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("加面包", "u1", tasks)
            await self.response("tool", tasks,
                [("add_items", {"list_name": "购物清单", "items": ["面包"]})],
                text="我来把面包加到购物清单。")
            self.assertEqual(self.outputs()[0][1]["status"], "pending")
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)

    async def test_promise_in_streamed_transcript_also_triggers_correction(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("删牛奶", "u1", tasks)
            await self.receiver.handle({"type": "response.created", "response": {"id": "streamed"}}, tasks)
            await self.receiver.handle({"type": "response.audio_transcript.done", "response_id": "streamed",
                "item_id": "audio", "transcript": "我这就把牛奶删掉。"}, tasks)
            await self.receiver.handle({"type": "response.done", "response": {
                "id": "streamed", "status": "completed", "output": []}}, tasks)
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 2)

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

    async def test_pending_write_waits_for_execution_before_requesting_one_report(self):
        self.flow.window_s = .1
        self.flow.incomplete_s = .15
        async with asyncio.TaskGroup() as tasks:
            await self.user("加面包", "u1", tasks)
            await self.response("add", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            self.assertEqual(self.outputs()[0][1]["status"], "pending")
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)
            self.remote.assert_not_awaited()
            await asyncio.sleep(.15)
            await self.receiver.maybe_respond()
            self.remote.assert_awaited_once()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 2)
            self.assertTrue(any("实际执行结果" in str(e) and "succeeded" in str(e) for e in self.sent()))
            await self.response("report", tasks, text="面包加好了")
            await self.receiver.maybe_respond()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 2)

    async def test_read_tool_still_requests_model_continuation(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("清单有什么", "u1", tasks)
            await self.response("read", tasks, [("list_items", {"list_name": "购物清单"})])
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 2)

    async def test_rejected_write_still_requests_model_continuation(self):
        async with asyncio.TaskGroup() as tasks:
            await self.user("加面包", "u1", tasks)
            await self.response("invalid", tasks, [("add_items", {"items": []})])
            self.assertTrue(self.outputs()[0][1]["is_error"])
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 2)

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
            await self.user("算了不加了", "u2", tasks)
            await self.response("cancel", tasks, [("cancel_pending", {})])
            await asyncio.sleep(.2)
            self.remote.assert_not_awaited()
            self.assertIn("已取消", self.outputs()[-1][1]["text"])

    async def test_existing_item_deletes_without_confirmation_and_reports_once(self):
        async def remote(name, args):
            if name == "inspect_task":
                return ExecutorResult(True, {"structured_content": {
                    "present": True, "title": "牛奶", "resource": {"id": "old", "title": "牛奶"}}})
            return ExecutorResult(True, "已删除牛奶")
        self.remote.side_effect = remote
        async with asyncio.TaskGroup() as tasks:
            await self.user("删掉原来的牛奶", "u1", tasks)
            await self.response("request", tasks, [("delete_item", {"kind": "task", "list_id": "list", "item_id": "old"})])
            self.assertEqual(self.outputs()[-1][1]["status"], "pending")
            self.assertNotIn("needs_confirmation", self.outputs()[-1][1])
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 1)
            await asyncio.sleep(.07)
            await self.receiver.maybe_respond()
            self.remote.assert_awaited_with("delete_task", {"list_id": "list", "task_id": "old"})
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 2)
            await self.response("report", tasks, text="牛奶删掉了")
            self.assertFalse(self.flow._report_pending)

    async def test_next_batch_waits_until_previous_batch_report_finishes(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def remote(name, args):
            if args.get("items") == ["面包"]:
                entered.set()
                await release.wait()
            return ExecutorResult(True, "已添加" + args["items"][0])
        self.remote.side_effect = remote
        async with asyncio.TaskGroup() as tasks:
            await self.user("加面包", "u1", tasks)
            await self.response("first", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            await asyncio.wait_for(entered.wait(), 1)
            await self.user("再加牛奶", "u2", tasks)
            await self.response("second", tasks, [("add_items", {"list_name": "购物清单", "items": ["牛奶"]})])
            await asyncio.sleep(.04)
            release.set()
            await asyncio.sleep(.04)
            await self.receiver.maybe_respond()
            self.remote.assert_awaited_once()
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 3)
            await self.response("first-report", tasks, text="面包加好了")
            await asyncio.sleep(.04)
            await self.receiver.maybe_respond()
            self.assertEqual(self.remote.await_count, 2)
            self.assertEqual(sum(e["type"] == "response.create" for e in self.sent()), 4)
            await self.response("second-report", tasks, text="牛奶加好了")
            self.assertFalse(self.flow.has_pending)
            self.assertFalse(self.flow._report_pending)

    async def test_interrupted_report_is_retried_then_next_batch_executes(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def remote(name, args):
            if args.get("items") == ["面包"]:
                entered.set()
                await release.wait()
            return ExecutorResult(True, "已添加" + args["items"][0])
        self.remote.side_effect = remote
        async with asyncio.TaskGroup() as tasks:
            await self.user("加面包", "u1", tasks)
            await self.response("first", tasks, [("add_items", {"list_name": "购物清单", "items": ["面包"]})])
            await asyncio.wait_for(entered.wait(), 1)
            await self.user("再加牛奶", "u2", tasks)
            await self.response("second", tasks, [("add_items", {"list_name": "购物清单", "items": ["牛奶"]})])
            await asyncio.sleep(.04)
            release.set()
            await asyncio.sleep(.04)
            await self.receiver.maybe_respond()
            await self.receiver.handle({"type": "response.created", "response": {"id": "interrupted-report"}}, tasks)
            await self.receiver.handle({"type": "response.audio.delta", "response_id": "interrupted-report",
                "item_id": "report-audio", "delta": base64.b64encode(b"\x00\x20" * 24).decode()}, tasks)
            await self.user("等一下，先听我说", "u3", tasks)
            self.assertTrue(any(e["type"] == "response.cancel" for e in self.sent()))
            await self.receiver.handle({"type": "response.done", "response": {
                "id": "interrupted-report", "status": "cancelled", "output": []}}, tasks)
            self.assertTrue(self.flow._report_pending)
            self.assertIsNotNone(self.receiver.active_report)
            self.remote.assert_awaited_once()
            await self.response("interjection", tasks, text="嗯，我在听。")
            await self.receiver.maybe_respond()
            self.assertTrue(self.receiver.report_requested)
            self.assertEqual(sum("只根据以下实际执行结果" in str(e) for e in self.sent()), 2)
            self.remote.assert_awaited_once()
            await self.response("retried-report", tasks, text="面包已经加好了。")
            await asyncio.sleep(.04)
            await self.receiver.maybe_respond()
            self.assertEqual(self.remote.await_count, 2)
            await self.response("second-report", tasks, text="牛奶已经加好了。")
            self.assertFalse(self.flow._report_pending)
            self.assertFalse(self.flow.has_pending)

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
                         {"cancel_pending", "undo_recent", "delete_item", "restore_recent"})
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
