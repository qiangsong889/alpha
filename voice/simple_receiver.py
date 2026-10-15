"""保留原有音频接收，移除计划、复述、版本和备用 TTS 的生产接线。"""

import asyncio
import base64
import json
from typing import Any
import jsonschema  # type: ignore[import-untyped]

from integrations.tool_executor import HA_ACTION_TOOLS
from integrations.tool_policy import get_policy
from voice.simple_tools import CONTROL_TOOLS
from voice.commit_window import CommitReport


def receiver_type(audio_receiver: Any, *, claims_completion):
    class SimpleReceiver(audio_receiver):  # type: ignore[misc, valid-type]
        def __init__(self, *args, workflow, **kwargs):
            super().__init__(*args, coordinator=workflow, commit_window=workflow, **kwargs)
            self.workflow = workflow
            self.active_report: CommitReport | None = None
            self.schemas = {tool["name"]: tool["input_schema"] for tool in kwargs.get("tools") or []}
            self._hold_next = False
            self._held: dict[str, list[dict]] = {}
            self._muted: set[str] = set()
            self._mute_next = False
            self._pending_write_turns: set[str] = set()
            self._reconnect_prompted = False

        async def request_response(self, target, *, silent=False, correction=""):
            self._hold_next = not silent and not correction
            await super().request_response(target, silent=silent, correction=correction)

        async def _handle(self, event, timers):
            kind = event["type"]
            turn = f"{self.session_id}:{event.get('item_id', '')}"
            if kind == "input_audio_buffer.speech_started":
                self.workflow.speech_started(turn)
            elif kind == "input_audio_buffer.speech_stopped":
                self.workflow.speech_stopped(turn)
            elif kind == "conversation.item.input_audio_transcription.completed":
                self.workflow.record_text(turn, event.get("transcript", ""))
            if kind == "response.created":
                key = event["response"]["id"]
                if self._hold_next:
                    self._held[key] = []
                if self._mute_next:
                    self._muted.add(key)
                self._hold_next = self._mute_next = False
            if kind == "response.audio.delta" and event.get("response_id") in self._held:
                self._held[event["response_id"]].append(event)
                return
            if kind == "response.audio.done" and any(
                any(chunk["item_id"] == event["item_id"] for chunk in chunks) for chunks in self._held.values()
            ):
                return
            if kind == "response.done":
                response = event["response"]
                key = response["id"]
                if key in self._held:
                    chunks = self._held.pop(key)
                    turn_key = self.response_turns.get(key, "")
                    text = "".join(part.get("transcript", part.get("text", ""))
                        for item in response.get("output", []) for part in item.get("content", []))
                    text = text or "".join(self.response_text.get(key, {}).values())
                    is_tool = self.tool_pending.get(key, 0) or any(item.get("type") == "function_call" for item in response.get("output", []))
                    pending = turn_key in self._pending_write_turns
                    unsupported = claims_completion(text) and not self.turn_successes.get(turn_key)
                    if (response.get("status") == "completed" and not is_tool and not pending
                            and not unsupported and key not in self.interrupted | self._muted):
                        for chunk in chunks:
                            self.player.append(base64.b64decode(chunk["delta"]), chunk["item_id"], chunk.get("content_index", 0))
                    elif is_tool or pending:
                        self._muted.add(key)
                if key in self._muted:
                    self.checked_claims.add(key)
            if kind == "response.audio.delta" and event.get("response_id") in self._muted:
                return
            await super()._handle(event, timers)

        async def maybe_respond(self):
            if self.current_response is not None or self.response_pending:
                return
            target = self.candidate
            if target is not None and target.meaningful is None:
                return
            if target is not None and target.meaningful is True and target.committed and not target.responded:
                target.responded = True
                await self.request_response(target)
                return
            effective = self.last_meaningful_candidate if target is not None and target.meaningful is False else target
            turn = self.turn_key(effective)
            if self.turn_pending.get(turn, 0):
                return
            for key, count in self.tool_pending.items():
                if count or self.finished_responses.get(key) != "completed" or key in self.tool_continued:
                    continue
                self.tool_continued.add(key)
                if self.tool_candidates.get(key) is not effective:
                    continue
                # 不等模型调用 finish_collection；当前 response 和全部工具结束即放行窗口。
                self.workflow.turn_finished(self.coordinator_turn(effective))
                self._mute_next = turn in self._pending_write_turns
                await self.request_response(effective, silent=self.tool_silent.get(key, False))
                return
            if await self.correct_unsupported_claim():
                return
            await self.maybe_finish_turn()
            if self.active_report is None and not self.workflow.reports.empty():
                self.active_report = self.workflow.reports.get_nowait()
            if self.active_report is not None and self.report_response is None:
                self.report_requested = True
                self.log("[提交结果] " + self.active_report.text)
                await self.request_response(effective, correction=(
                    "只根据以下实际执行结果简短报告，不再调用工具；失败、未知不得说成功：\n" + self.active_report.text))
                return
            if not self._reconnect_prompted and (prompt := self.workflow.reconnect_prompt()):
                self._reconnect_prompted = True
                self.log("[重连待办] " + prompt)
                await self.request_response(effective, correction=(
                    "只询问下面这句话，不调用工具，不执行等待中的操作：" + prompt))
                return
            while not self.workflow.notices.empty():
                notice = self.workflow.notices.get_nowait()
                if self.workflow.is_notice_current(notice):
                    await self.request_response(effective, correction="只问一句，不调用工具：" + notice.text)
                    return

        async def execute_tool(self, event):
            name, key, call_id = event["name"], event["response_id"], event["call_id"]
            turn = self.response_turns[key]
            full_turn = f"{self.session_id}:{turn}"
            silent = False
            try:
                args = json.loads(event.get("arguments", "{}"))
                if not isinstance(args, dict):
                    raise ValueError("工具参数必须是对象")
                if full_turn != self.workflow.current_turn:
                    raise ValueError("用户已继续说话，本次旧请求未执行")
                if name in CONTROL_TOOLS:
                    jsonschema.validate(args, CONTROL_TOOLS[name]["parameters"])
                    if name == "cancel_pending":
                        output = self.workflow.cancel_pending()
                    elif name == "undo_recent":
                        output = await self.workflow.undo_recent(**args)
                    else:
                        output = await self.workflow.delete_item(**args)
                else:
                    if name not in self.tool_names or get_policy(name) is None:
                        raise ValueError("工具未加载或不在白名单")
                    jsonschema.validate(args, self.schemas[name])
                    if "operation_id" in args:
                        raise ValueError("operation_id 由系统分配")
                    output = await self.workflow.invoke(full_turn, name, args)
                    if output.get("status") == "pending":
                        self._pending_write_turns.add(turn)
                    silent = name in HA_ACTION_TOOLS and output.get("status") == "succeeded"
                for item in output.get("operations", [output]):
                    op_id = item.get("operation_id")
                    if op_id:
                        self.turn_operations.setdefault(turn, []).append(op_id)
                        op = self.gateway.store.get(op_id)
                        if op is not None and (policy := get_policy(op.name)) is not None and policy.mutates:
                            if op.status in {"succeeded", "failed", "unknown"}:
                                self.turn_attempts.setdefault(turn, set()).add(op_id)
                            if op.status == "succeeded":
                                self.turn_successes.setdefault(turn, set()).add(op_id)
            except Exception as exc:
                output = {"is_error": True, "text": str(exc)}
            self.log(f"[工具] {name} {event.get('arguments', '')} → {str(output['text'])[:100]}")
            await self.command("conversation.item.create", item={"type": "function_call_output", "call_id": call_id,
                "output": json.dumps(output, ensure_ascii=False)})
            self.tool_silent[key] = self.tool_silent.get(key, True) and silent
            self.tool_pending[key] -= 1
            self.turn_pending[turn] -= 1
            await self.maybe_respond()
            await self.maybe_finish_turn()
            self.report_wakeup.set()

        async def maybe_finish_turn(self):
            if self.current_response is not None or self.response_pending:
                return
            target = self.candidate
            if target is None or target.meaningful is not True or not target.responded:
                return
            turn = self.turn_key(target)
            if self.turn_pending.get(turn, 0):
                return
            completed = [key for key, status in self.finished_responses.items()
                if status == "completed" and self.response_turns.get(key) == turn
                and key not in self.finalized_responses and key != self.report_response]
            if completed:
                self.finalized_responses.update(completed)
                self.workflow.turn_finished(self.coordinator_turn(target))

        async def watch_reports(self):
            while True:
                await self.maybe_respond()
                await asyncio.sleep(.02)

    return SimpleReceiver
