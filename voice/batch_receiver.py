"""在原有音频事件处理之上接入唯一的新批次业务入口。"""

import asyncio
import json
import re
import base64
import os
import shutil
from typing import Any
import jsonschema  # type: ignore[import-untyped]
from voice.commit_window import CommitReport

from integrations.tool_executor import HA_ACTION_TOOLS, format_for_model
from integrations.tool_policy import get_policy
from voice.batch_tools import CONTROL_TOOLS, call_batch_control
from voice.plan_tools import control_output


def receiver_type(audio_receiver: Any, *, claims_completion):
    # 由 qwen_realtime 传入音频事件类，避免 -m 运行时循环导入另一个模块实例。
    class BatchQwenReceiver(audio_receiver):  # type: ignore[misc, valid-type]
        def __init__(self, *args, workflow, **kwargs):
            super().__init__(*args, commit_window=workflow, coordinator=workflow, **kwargs)
            self.workflow = workflow
            self.schemas = {tool["name"]: tool["input_schema"] for tool in kwargs.get("tools") or []}
            self.muted: set[str] = set()
            self.mute_next = False
            self.recap_needed = False
            self._notice = None
            self._closed_turns: set[str] = set()
            self._recap_failures: dict[tuple[str, int], int] = {}
            self.active_report: CommitReport | None = None
            self._hold_next = False
            self._held: dict[str, list[dict]] = {}
            self._mutated_turns: set[str] = set()
            self._local_player = None
            self._local_starting = False

        async def request_response(self, target, *, silent=False, correction=""):
            self._hold_next = not correction and not silent
            await super().request_response(target, silent=silent, correction=correction)

        async def _handle(self, event, timers):
            kind = event["type"]
            turn = f"{self.session_id}:{event.get('item_id', '')}"
            if kind == "input_audio_buffer.speech_started":
                if self._local_player is not None and self._local_player.returncode is None:
                    self._local_player.terminate()
                self.workflow.speech_started(turn)
            elif kind == "input_audio_buffer.speech_stopped":
                self.workflow.speech_stopped(turn)
            elif kind == "conversation.item.input_audio_transcription.completed":
                self.workflow.record_text(turn, event.get("transcript", ""))
            if kind == "response.created" and self.mute_next:
                self.muted.add(event["response"]["id"])
                self.mute_next = False
            if kind == "response.created" and self._hold_next:
                self._held[event["response"]["id"]] = []
                self._hold_next = False
            if kind == "response.audio.delta" and event.get("response_id") in self._held:
                self._held[event["response_id"]].append(event)
                return
            if kind == "response.audio.done" and any(
                any(chunk["item_id"] == event["item_id"] for chunk in chunks)
                for chunks in self._held.values()
            ):
                return  # append 前不能先 finish，否则播放器会把晚到音频丢掉。
            if kind == "response.done" and event["response"]["id"] in self._held:
                response = event["response"]
                response_id = response["id"]
                chunks = self._held.pop(response_id)
                response_turn = self.response_turns.get(response_id, "")
                mutating = f"{self.session_id}:{response_turn}" in self._mutated_turns
                is_tool = self.tool_pending.get(response_id, 0) or any(
                    item.get("type") == "function_call" for item in response.get("output", []))
                text = "".join(part.get("transcript", part.get("text", ""))
                               for item in response.get("output", []) for part in item.get("content", []))
                text = text or "".join(self.response_text.get(response_id, {}).values())
                evidence = any((op := self.gateway.store.get(key)) is not None and op.status == "succeeded"
                               and (policy := get_policy(op.name)) is not None and policy.mutates
                               for key in self.turn_operations.get(response_turn, []))
                unsupported = claims_completion(text) and not evidence
                if (response.get("status") == "completed" and not mutating and not is_tool
                        and not unsupported and response_id not in self.interrupted | self.muted):
                    for chunk in chunks:
                        self.player.append(base64.b64decode(chunk["delta"]), chunk["item_id"], chunk.get("content_index", 0))
                elif not unsupported or mutating or is_tool:
                    self.muted.add(response_id)
            if kind == "response.done" and event["response"]["id"] in self.muted:
                self.checked_claims.add(event["response"]["id"])
            if kind == "response.audio.delta" and event.get("response_id") in self.muted:
                return
            await super()._handle(event, timers)
            if kind == "response.done" and event["response"]["id"] in self.muted:
                self.checked_claims.add(event["response"]["id"])

        def active_plan(self):
            return bool(self.workflow.plan.items)

        async def maybe_respond(self):
            if self.current_response is not None or self.response_pending:
                return
            target = self.candidate
            if target is not None and target.meaningful is None:
                return
            if target is not None and target.meaningful is True and target.committed and not target.responded:
                target.responded = True
                # 理解任务的回答默认静音；查询结果或普通聊天在工具后续轮次/最终轮次播放。
                self.mute_next = self.workflow.executor._runner is not None and any(
                    batch.status == "executing" for batch in self.workflow.chain.batches)
                await self.request_response(target)
                return
            effective = self.last_meaningful_candidate if target is not None and target.meaningful is False else target
            if self.turn_pending.get(self.turn_key(effective), 0):
                return
            for response_id, count in self.tool_pending.items():
                if (count or self.finished_responses.get(response_id) != "completed"
                        or response_id in self.tool_continued):
                    continue
                self.tool_continued.add(response_id)
                if self.tool_candidates.get(response_id) is not effective:
                    continue  # 新输入已经取代旧问题，不再生成过时的后续回答。
                if response_id in self.recaps:
                    await self.request_response(effective, silent=True)
                    return
                # 收集过程只静音本地音频，不使用协议的 silent drain：模型还可以继续修改任务。
                self.mute_next = bool(self.workflow.plan.items) or any(
                    batch.status == "executing" for batch in self.workflow.chain.batches)
                await self.request_response(effective)
                return
            if await self.correct_unsupported_claim():
                return
            await self.maybe_finish_turn()
            if any(batch.status == "executing" for batch in self.workflow.chain.batches):
                return  # 执行时继续理解新输入，但不主动播报/复述下一批。
            if self._local_player is not None or self._local_starting:
                return
            if self.active_report is None and not self.workflow.reports.empty():
                self.active_report = self.workflow.reports.get_nowait()
            if self.active_report is not None and self.report_response is None:
                if all(op.name in HA_ACTION_TOOLS and op.status == "succeeded"
                       for op in self.active_report.operations):
                    self.log("[静默] 设备操作全部成功")
                    self.active_report = None
                    return
                self.report_requested = True
                self.log("[提交结果] " + self.active_report.text)
                await self.request_response(effective, correction=(
                    "系统执行结果，仅按这些状态报告，不再调用工具，不把排队说成完成：\n" + self.active_report.text))
                return
            if target is not None and target.meaningful is True and not target.responded:
                return
            if self._notice is None and not self.workflow.notices.empty():
                self._notice = self.workflow.notices.get_nowait()
            if self._notice is not None:
                notice, self._notice = self._notice, None
                if notice.turn_id is not None and notice.turn_id != self.workflow.current_turn:
                    return
                if (notice.batch_id, notice.version) != (self.workflow.batch_id, self.workflow.plan.version):
                    return
                if notice.kind == "recap" and self.workflow.plan.items:
                    await self.request_recap(effective)
                elif notice.kind == "clarify":
                    await self.request_response(effective, correction="只问下面这一个问题，不要调用工具：" + notice.text)
                return

        async def execute_tool(self, event):
            name, response_id, call_id = event["name"], event["response_id"], event["call_id"]
            turn = self.response_turns[response_id]
            full_turn = f"{self.session_id}:{turn}"
            try:
                args = json.loads(event.get("arguments", "{}"))
                if not isinstance(args, dict):
                    raise ValueError("工具参数必须是对象")
                if response_id in self.recaps:
                    self.recaps[response_id].invalid = True
                    output = {"is_error": True, "reason": "recap_only", "text": "复述轮禁止工具，本次未执行"}
                elif full_turn != self.workflow.current_turn:
                    output = {"is_error": True, "reason": "stale_turn", "text": "用户已继续说话，本次旧请求未执行，请重新整理最新输入"}
                elif name in CONTROL_TOOLS:
                    if name == "update_plan_item" and type(args.get("item_id")) is int:
                        task = self.workflow.chain.task(args["item_id"])
                        if task.name in self.schemas:
                            jsonschema.validate(args.get("arguments"), self.schemas[task.name])
                    result = await call_batch_control(self.workflow, name, args)
                    output = control_output(result)
                    if result.ok and name in {"update_plan_item", "remove_plan_item", "cancel_task", "cancel_remaining", "cancel_plan", "set_task_dependencies"}:
                        self._mutated_turns.add(full_turn)
                    if result.reason == "confirmation_recorded_not_submitted":
                        output["text"] = "已记录确认，需先整理完本句再提交，尚未执行"
                else:
                    policy = get_policy(name)
                    if name not in self.tool_names or policy is None:
                        raise ValueError("工具未加载或不在白名单")
                    jsonschema.validate(args, self.schemas[name])
                    if "operation_id" in args:
                        raise ValueError("operation_id 由系统分配，不能由模型指定")
                    if policy.mutates:
                        result = self.workflow.add(full_turn, name, args)
                        self._mutated_turns.add(full_turn)
                        output = control_output(result)
                        self.turn_operations.setdefault(turn, []).extend(op.id for op in result.operations)
                    else:
                        op = await self.gateway.submit(full_turn, name, args)
                        output = dict(format_for_model(op))
                # 当前快照带固定任务编号，模型才能继续替换/取消。
                output.update(snapshot=self.workflow.snapshot(), version=self.workflow.plan.version)
            except Exception as exc:
                output = {"is_error": True, "text": str(exc), "snapshot": self.workflow.snapshot()}
            self.log(f"[工具] {name} {event.get('arguments', '')} → {str(output['text'])[:100]}")
            await self.command("conversation.item.create", item={
                "type": "function_call_output", "call_id": call_id,
                "output": json.dumps(output, ensure_ascii=False)})
            self.tool_pending[response_id] -= 1
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
                         and key not in self.tool_pending and key not in self.finalized_responses]
            if not completed:
                return
            self.finalized_responses.update(completed)
            if all(key in self.recaps or key == self.report_response for key in completed):
                return
            full = self.coordinator_turn(target)
            if full not in self._closed_turns:
                self._closed_turns.add(full)
                self.workflow.finish_turn(full)
                self.workflow.turn_answered(full)

        async def wait_recap_played(self, response_id):
            recap = self.recaps[response_id]
            # 复述内容必须吻合；纯标点/空白差异可以忽略，其他改写不授予确认资格。
            text = "".join(self.response_text.get(response_id, {}).values())
            expected = re.sub(r"\W", "", recap.text)
            actual = re.sub(r"\W", "", getattr(self, "_recap_transcripts", {}).get(response_id, text))
            if actual != expected:
                recap.invalid = True
                key = (recap.plan_id, recap.version)
                self._recap_failures[key] = self._recap_failures.get(key, 0) + 1
                self.log("[复述核对] 内容不一致，不开放确认")
                if self._recap_failures[key] < 2:
                    from core.task_coordinator import TaskNotice
                    self.workflow.notices.put_nowait(TaskNotice("recap", *key))
                else:
                    self.log("[复述核对] 连续两次不一致，改用本机 TTS 逐字播报")
                    await self.local_recap(recap)
                    return
            await super().wait_recap_played(response_id)

        async def local_recap(self, recap):
            self.workflow.recap_played(recap.plan_id, recap.version, True)
            command = shutil.which("say")
            if command is None:
                self.log("[复述核对] 本机没有 say，保持未确认，请修改任务后重试")
                self.workflow.recap_played(recap.plan_id, recap.version, True)
                recap.checked = True
                return
            target = self.candidate
            process = None
            self.player.clear()
            self._local_starting = True
            try:
                process = await asyncio.create_subprocess_exec(
                    command, "-v", os.environ.get("RECAP_TTS_VOICE", "Tingting"), recap.text,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                self._local_player = process
                self._local_starting = False
                if self.candidate is not target:
                    process.terminate()
                code = await process.wait()
                played = code == 0 and self.candidate is target
                self.workflow.recap_played(recap.plan_id, recap.version, interrupted=not played)
                self.log(f"[本机复述] played={played}")
            except OSError as exc:
                self.log(f"[本机复述] 未能播放：{exc}")
                self.workflow.recap_played(recap.plan_id, recap.version, True)
            finally:
                if process is not None and process.returncode is None:
                    process.terminate()
                    await process.wait()
                self._local_player = None
                self._local_starting = False
                recap.checked = True

        async def handle(self, event, timers):
            if event["type"] == "response.done":
                response = event["response"]
                if response["id"] in self.recaps:
                    if not hasattr(self, "_recap_transcripts"):
                        self._recap_transcripts = {}
                    self._recap_transcripts[response["id"]] = "".join(
                        part.get("transcript", part.get("text", ""))
                        for item in response.get("output", []) for part in item.get("content", [])
                    ) or "".join(self.response_text.get(response["id"], {}).values())
            await super().handle(event, timers)

        async def watch_reports(self):
            # 报告、窗口决策和新输入共用一个串行发送入口；不抢占正在说话的用户。
            while True:
                await self.maybe_respond()
                await asyncio.sleep(.02)

    return BatchQwenReceiver
