"""保留原有音频接收，移除计划、复述、版本和备用 TTS 的生产接线。"""

import asyncio
import base64
import json
import re
from typing import Any
import jsonschema  # type: ignore[import-untyped]

from integrations.tool_executor import HA_ACTION_TOOLS
from integrations.tool_policy import get_policy
from voice.simple_tools import CONTROL_TOOLS
from voice.commit_window import CommitReport


def promises_action(text: str) -> bool:
    """保守识别执行承诺；不据此直接执行任何操作。"""
    for clause in re.findall(r"[^。！？!?；;\n]+[。！？!?；;\n]?", text):
        if clause.rstrip().endswith(("？", "?")) or any(word in clause for word in (
            "如果", "假如", "比如", "例如", "不能", "不会", "不再", "不删除", "不撤销",
            "不执行", "没有权限", "无法", "需要你", "需要您", "你说", "你刚才说", "“", "「",
        )):
            continue
        if re.search(r"(?:我(?:这就|现在|马上|来|会|帮你|替你|给你|把|将)|这就|马上就|现在就)"
                     r"[^。！？!?；;\n]{0,40}(?:删除|删掉|撤掉|撤销|移除|添加|加到|加进|设置|设好|创建|打开|关掉|关闭|调到|标记|恢复)", clause):
            return True
    return False


def shopping_request(text: str) -> tuple[str, dict] | None:
    """仅匹配明确的单句购物快捷指令；复杂/含糊表达仍交给模型。"""
    compact = "".join(char for char in text if char.isalnum())
    if compact in {"购物清单里有什么", "现在购物清单里有什么", "查看购物清单", "看看购物清单"}:
        return "list_items", {"list_name": "购物清单"}
    match = re.fullmatch(r"买\s*([^。！？!?，,；;\n]{1,30})[。！!]?", text.strip())
    if match:
        title = match[1].strip()
        if title and not any(word in title for word in (
            "吗", "呢", "多少", "怎么", "什么", "还是", "然后", "不要", "不买", "别", "等等", "还有", "和", "与", "、", "以及",
        )):
            return "add_items", {"list_name": "购物清单", "items": [title]}
    return None


def receiver_type(audio_receiver: Any, *, claims_completion):
    class SimpleReceiver(audio_receiver):  # type: ignore[misc, valid-type]
        def __init__(self, *args, workflow, **kwargs):
            super().__init__(*args, coordinator=workflow, commit_window=workflow, **kwargs)
            self.workflow = workflow
            self.active_report: CommitReport | None = None
            self.report_response: str | None = None
            self.report_requested: bool = False
            self.schemas = {tool["name"]: tool["input_schema"] for tool in kwargs.get("tools") or []}
            self._hold_next = False
            self._held: dict[str, list[dict]] = {}
            self._muted: set[str] = set()
            self._mute_next = False
            self._pending_write_turns: set[str] = set()
            self._pending_only: dict[str, bool] = {}
            self._reconnect_prompted = False
            self._promises: dict[str, str] = {}
            self._checked_promises: set[str] = set()
            self._failure_notified: set[str] = set()
            self._failure_next = False
            self._failure_responses: set[str] = set()
            self._exhausted_turns: set[str] = set()
            self._report_responses: dict[str, CommitReport] = {}
            self._report_retries: dict[tuple[str, ...], int] = {}
            self._handled_report_tool_done: set[str] = set()
            self._timers: asyncio.TaskGroup | None = None
            self._local_results: dict[str, dict[str, Any]] = {}

        async def request_response(self, target, *, silent=False, correction=""):
            # 纠正回答仍可能再次虚构，必须像普通回答一样先核实，再播放。
            # 只有事实报告和禁止调用工具的最终状态提示保持流式播放。
            self._hold_next = not silent and not self.report_requested and not self._failure_next
            await super().request_response(target, silent=silent, correction=correction)

        async def _handle(self, event, timers):
            self._timers = timers
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
                if self.report_requested and self.active_report is not None:
                    self._report_responses[key] = self.active_report
                if self._failure_next:
                    self._failure_responses.add(key)
                    self._failure_next = False
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
                text = "".join(part.get("transcript", part.get("text", ""))
                    for item in response.get("output", []) if item.get("type") == "message"
                    for part in item.get("content", []))
                text = text or "".join(self.response_text.get(key, {}).values())
                is_tool = key in self.tool_pending or any(
                    item.get("type") == "function_call" for item in response.get("output", []))
                promised_without_tool = (response.get("status") == "completed" and not is_tool
                    and key != self.report_response and key not in self.interrupted | self.silent_responses
                    and key not in self._failure_responses
                    and promises_action(text))
                if promised_without_tool:
                    self._promises[key] = text
                turn_key = self.response_turns.get(key, "")
                if (response.get("status") == "completed" and not is_tool
                        and key not in self.interrupted | self.silent_responses | self._failure_responses
                        and key != self.report_response and self.correction_counts.get(turn_key, 0) >= 1
                        and (promised_without_tool or (claims_completion(text) and not self.turn_successes.get(turn_key)))):
                    self._exhausted_turns.add(turn_key)
                if key in self._held:
                    chunks = self._held.pop(key)
                    turn_key = self.response_turns.get(key, "")
                    pending = turn_key in self._pending_write_turns
                    unsupported = claims_completion(text) and not self.turn_successes.get(turn_key)
                    if (response.get("status") == "completed" and not is_tool and not pending
                            and not unsupported and not promised_without_tool
                            and key not in self.interrupted | self._muted):
                        for chunk in chunks:
                            self.player.append(base64.b64decode(chunk["delta"]), chunk["item_id"], chunk.get("content_index", 0))
                    elif is_tool or pending:
                        self._muted.add(key)
                if key in self._muted:
                    self.checked_claims.add(key)
                if key in self._failure_responses:
                    self.checked_claims.add(key)
                if (key in self._report_responses and is_tool and response.get("status") == "completed"
                        and key not in self.interrupted and key not in self._handled_report_tool_done):
                    self._handled_report_tool_done.add(key)
                    self.checked_claims.add(key)
                    # 工具调用不是口头汇报完成。阻止基类清掉事实报告/放行下一批。
                    if self.report_response == key:
                        self.report_response = None
                    report = self._report_responses[key]
                    batch = tuple(op.id for op in report.operations)
                    if self._report_retries.get(batch, 0) < 1:
                        self._report_retries[batch] = 1
                        self.log("[汇报纠正] 重复工具未执行，将根据本批实际结果重新汇报")
                    else:
                        self.log("[汇报降级] 模型再次请求工具，不再重试；本批实际结果：" + report.text)
                        self.active_report = None
                        self.workflow.report_finished()
            if kind == "response.audio.delta" and event.get("response_id") in self._muted:
                return
            finished_report = (kind == "response.done"
                and event["response"]["id"] == self.report_response
                and event["response"].get("status") == "completed"
                and event["response"]["id"] not in self.interrupted)
            await super()._handle(event, timers)
            if finished_report:
                self.workflow.report_finished()

        async def maybe_respond(self):
            if self.current_response is not None or self.response_pending:
                return
            target = self.candidate
            if target is not None and target.meaningful is None:
                return
            if target is not None and target.meaningful is True and target.committed and not target.responded:
                target.responded = True
                raw_text = self.workflow.user_text(self.coordinator_turn(target))
                direct = shopping_request(raw_text)
                if direct is not None and direct[0] in self.tool_names and self._timers is not None:
                    turn = self.turn_key(target)
                    self.turn_pending[turn] = self.turn_pending.get(turn, 0) + 1
                    self._timers.create_task(self.call_local_shopping(target, *direct), name="local-shopping")
                    return
                text = "".join(char for char in raw_text if char.isalnum())
                if (text in {"刚才那个不要了", "刚刚那个不要了", "刚才加的不要了", "撤销刚才的操作"}
                        and self.workflow.has_unambiguous_recent_creation()):
                    output = await self.workflow.undo_recent()
                    self.log("[本地撤销] " + output["text"])
                    await self.command("conversation.item.create", item={"type": "message", "role": "system",
                        "content": [{"type": "input_text", "text": "本次用户请求已由代码调用 undo_recent："
                                     + json.dumps(output, ensure_ascii=False)}]})
                    if output.get("status") == "pending":
                        self._pending_write_turns.add(self.turn_key(target))
                        self.workflow.turn_finished(self.coordinator_turn(target))
                    else:
                        await self.request_response(target, correction="只根据以下实际工具结果简短回答，不再执行工具：" + output["text"])
                    return
                await self.request_response(target)
                return
            effective = self.last_meaningful_candidate if target is not None and target.meaningful is False else target
            turn = self.turn_key(effective)
            if self.turn_pending.get(turn, 0):
                return
            if turn in self._local_results:
                output = self._local_results.pop(turn)
                await self.request_response(effective, correction=(
                    "本次用户请求已由代码调用购物工具。只根据以下本次实际返回的数据简短回答；"
                    "不要重复调用工具，不要根据旧对话猜测清单内容：\n"
                    + json.dumps(output, ensure_ascii=False)))
                return
            for key, count in self.tool_pending.items():
                if count or self.finished_responses.get(key) != "completed" or key in self.tool_continued:
                    continue
                self.tool_continued.add(key)
                if key in self._report_responses:
                    continue  # 只走事实报告的重报路径，不作为一次新的业务工具调用。
                if self.tool_candidates.get(key) is not effective:
                    continue
                # 不等模型调用 finish_collection；当前 response 和全部工具结束即放行窗口。
                self.workflow.turn_finished(self.coordinator_turn(effective))
                if self._pending_only.get(key, False):
                    # 这一组只有待执行写入：结果已交回服务端，先不请求口头确认。
                    # 窗口执行后统一报告，避免生成一次“尚未执行”的多余回答。
                    continue
                self._mute_next = turn in self._pending_write_turns
                await self.request_response(effective, silent=self.tool_silent.get(key, False))
                return
            if await self.correct_unsupported_claim():
                return
            if await self.correct_unfulfilled_promise(effective):
                return
            if turn in self._exhausted_turns and turn not in self._failure_notified:
                await self.notify_unfulfilled_request(effective)
                return
            await self.maybe_finish_turn()
            if self.active_report is None and not self.workflow.reports.empty():
                self.active_report = self.workflow.reports.get_nowait()
            if self.active_report is not None and self.report_response is None:
                self.report_requested = True
                self.log("[提交结果] " + self.active_report.text)
                await self.request_response(effective, correction=(
                    "本批操作已经执行并返回结果，你现在只负责口头汇报，不是再次执行用户指令。"
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

        async def call_local_shopping(self, target, name, args):
            turn = self.turn_key(target)
            try:
                try:
                    output = await self.workflow.invoke(self.coordinator_turn(target), name, args)
                except Exception as exc:
                    output = {"is_error": True, "text": str(exc)}
                self.log(f"[工具] {name} {json.dumps(args, ensure_ascii=False)} → {output['text'][:100]}")
                op_id = output.get("operation_id")
                if op_id:
                    self.turn_operations.setdefault(turn, []).append(op_id)
                if self.coordinator_turn(target) != self.workflow.current_turn:
                    return  # 新输入已取代旧查询；操作记录保留，不补发旧回答。
                if output.get("status") == "pending":
                    self._pending_write_turns.add(turn)
                    self.workflow.turn_finished(self.coordinator_turn(target))
                else:
                    self._local_results[turn] = output
            finally:
                self.turn_pending[turn] -= 1
            await self.maybe_respond()
            self.report_wakeup.set()

        async def correct_unfulfilled_promise(self, target):
            turn = self.turn_key(target)
            for key, text in self._promises.items():
                if key in self._checked_promises:
                    continue
                self._checked_promises.add(key)
                if self.response_turns.get(key) != turn:
                    continue  # 新输入取代旧输入，不补发旧请求。
                self.log(f"[⚠️ 只说未做] response_id={key} ｜ {text[:100]}")
                if self.correction_counts.get(turn, 0) >= 1:
                    self.log("[核对] 本句已补发一次，不再自动请求")
                    continue
                self.correction_counts[turn] = 1
                await self.request_response(target, correction=(
                    "工具执行核对：刚才的回答承诺执行操作，但这次回答没有调用任何工具。"
                    "请现在根据用户原始指令调用对应工具，不要再只口头答应。"
                    "先核对当前操作记录：已登记、执行中、已成功或结果未知的操作不得重复提交；"
                    "若目标不明确则询问目标，失败或未知不得声称成功。"))
                return True
            return False

        async def notify_unfulfilled_request(self, target):
            turn = self.turn_key(target)
            self._failure_notified.add(turn)
            states = {op.status for key in self.turn_operations.get(turn, [])
                      if (op := self.gateway.store.get(key)) is not None
                      and (policy := get_policy(op.name)) is not None and policy.mutates}
            if "unknown" in states:
                text = "操作结果还不确定，请先核实，不要重复执行。"
            elif states & {"pending", "approved", "executing"}:
                text = "操作还在等待结果，我不会重复执行。"
            elif "succeeded" in states:
                text = "操作已有成功记录，请以执行结果为准。"
            else:
                text = "这次没有执行成功，请再说一次。"
            self.log("[执行反馈] " + text)
            self._failure_next = True
            await self.request_response(target, correction=(
                "执行失败提示：工具补发已用完，不再请求执行。只播报以下代码核对的实际状态："
                + text + " 不增加其他内容，不承诺操作，不调用任何工具。"))

        def schedule_tool(self, event, timers):
            if event["response_id"] in self._failure_responses:
                if event["call_id"] in self.tool_calls:
                    return
                self.tool_calls.add(event["call_id"])
                self.log("[执行失败] 失败提示中请求的工具未执行，停止重试")
                timers.create_task(self.command("conversation.item.create", item={
                    "type": "function_call_output", "call_id": event["call_id"],
                    "output": json.dumps({"is_error": True, "text": "执行重试已停止，本次未执行工具"}, ensure_ascii=False),
                }))
                return
            return super().schedule_tool(event, timers)

        async def execute_tool(self, event):
            name, key, call_id = event["name"], event["response_id"], event["call_id"]
            turn = self.response_turns[key]
            full_turn = f"{self.session_id}:{turn}"
            silent = False
            output: dict[str, Any]
            try:
                # 汇报阶段不解析/信任新的动作参数，只回送已保存的执行事实。
                args = {} if key in self._report_responses else json.loads(event.get("arguments", "{}"))
                if not isinstance(args, dict):
                    raise ValueError("工具参数必须是对象")
                if key in self._report_responses:
                    report = self._report_responses[key]
                    output = {"is_error": False, "executed": False, "report_only": True,
                              "text": "正在汇报执行结果，本次没有再次调用工具。请按本批真实结果汇报：\n" + report.text,
                              "operations": []}
                    self.log(f"[汇报拦截] {name} 未执行；保留本批实际结果")
                elif full_turn != self.workflow.current_turn:
                    raise ValueError("用户已继续说话，本次旧请求未执行")
                elif name in CONTROL_TOOLS:
                    jsonschema.validate(args, CONTROL_TOOLS[name]["parameters"])
                    if name == "cancel_pending":
                        output = self.workflow.cancel_pending()
                    elif name == "undo_recent":
                        output = await self.workflow.undo_recent(**args)
                    elif name == "restore_recent":
                        output = await self.workflow.restore_recent(**args)
                    else:
                        output = await self.workflow.delete_item(**args)
                else:
                    if name not in self.tool_names or get_policy(name) is None:
                        raise ValueError("工具未加载或不在白名单")
                    jsonschema.validate(args, self.schemas[name])
                    if "operation_id" in args:
                        raise ValueError("operation_id 由系统分配")
                    output = await self.workflow.invoke(full_turn, name, args)
                    silent = name in HA_ACTION_TOOLS and output.get("status") == "succeeded"
                if output.get("status") == "pending":
                    self._pending_write_turns.add(turn)
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
            self._pending_only[key] = (self._pending_only.get(key, True)
                                       and output.get("status") == "pending"
                                       and not output.get("is_error", False))
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
