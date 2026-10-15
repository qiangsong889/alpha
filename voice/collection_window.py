"""围绕用户说话计时的收集窗口；暂未接到 Qwen，不执行任何工具。

一个窗口对应一个正在收集的批次，业务层持有它以跨 WebSocket 重连保留。
窗口只发布决策事件。消费方必须先 is_current，再交给确认/执行协调层；
ready 不表示已经执行，confirmation_required 也不表示已经得到用户确认。
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import math
import time
from typing import Literal

from core.task_batch import TaskChain


@dataclass(frozen=True)
class CollectionEvent:
    kind: Literal["ready", "confirmation_required", "clarification_required", "empty"]
    batch_id: str
    version: int
    turn_id: str
    generation: int
    question: str = ""


class CollectionWindow:
    def __init__(
        self, chain: TaskChain, batch_id: str, *, window_s: float = 2.0,
        incomplete_s: float = 5.0, clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        for value in (window_s, incomplete_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("等待时间必须是有限的非负数")
        if incomplete_s < window_s:
            raise ValueError("未说完的等待时间不能短于普通窗口")
        if chain.batch(batch_id).status != "collecting":
            raise ValueError("窗口只能绑定正在收集的新批次")
        self.chain, self.batch_id = chain, batch_id
        self.window_s, self.incomplete_s = window_s, incomplete_s
        self.events: asyncio.Queue[CollectionEvent] = asyncio.Queue()
        self._clock, self._sleep = clock, sleep
        self._timer: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._generation = 0
        self._turn_id: str | None = None
        self._seen_turns: set[str] = set()
        self._speaking = False
        self._stopped_at: float | None = None
        self._previous_input: tuple[str | None, bool, float | None, bool] = (
            None, False, None, False,
        )
        # 上一句有效输入的判断保留到下一次转写，以便忽略噪音后恢复。
        self._decision: tuple[int, bool, str, bool] | None = None
        self._decision_turn_id = ""
        self._interpreted = False
        self._emitted: CollectionEvent | None = None
        self._closed = False

    def _invalidate(self) -> None:
        self._generation += 1
        self._emitted = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def suspend(self) -> None:
        """断线或等待确认时暂停提交，必须获得新的整理结果才可重新计时。"""
        self._invalidate()
        self._interpreted = False

    def speech_started(self, turn_id: str) -> bool:
        """必须在等待转写/模型之前调用；重复或晚到的输入编号不会重新开窗。"""
        if self._closed or not turn_id or turn_id in self._seen_turns:
            return False
        if self.chain.batch(self.batch_id).status not in {
            "collecting", "waiting_for_clarification", "awaiting_confirmation",
        }:
            return False
        self._seen_turns.add(turn_id)
        self._invalidate()
        self._previous_input = (
            self._turn_id, self._speaking, self._stopped_at, self._interpreted,
        )
        self._turn_id = turn_id
        self._speaking, self._interpreted = True, False
        self._stopped_at = None
        return True

    def speech_stopped(self, turn_id: str) -> bool:
        """记录说完时刻，而非模型回答完成时刻；重复通知不能延长窗口。"""
        if self._closed or turn_id != self._turn_id or not self._speaking:
            return False
        self._speaking = False
        self._stopped_at = self._clock()
        self._arm()
        return True

    def interpreted(
        self, turn_id: str, *, version: int, complete: bool,
        question: str = "", execute_now: bool = False,
    ) -> bool:
        """对应转写已整理完成，包括新增/修改/取消和完整性判断。

        version 必须是整理后的当前版本，防止旧的模型结果提交新计划。
        execute_now 表示用户明确说了“执行吧”，只跳过时间，不跳过完整性和确认。
        """
        batch = self.chain.batch(self.batch_id)
        if (self._closed or turn_id != self._turn_id or batch.version != version
                or batch.status not in {"collecting", "waiting_for_clarification", "awaiting_confirmation"}):
            return False
        if type(complete) is not bool or type(execute_now) is not bool:
            raise ValueError("complete 和 execute_now 必须是 bool")
        question = question.strip() if not complete else ""
        decision = (version, complete, question, execute_now)
        if self._interpreted and decision == self._decision:
            return True
        self._invalidate()
        self._decision, self._interpreted = decision, True
        self._decision_turn_id = turn_id
        # 完整性判断变化只影响收集，不代替播放器与确认协调层的授权。
        self.chain.set_collection_status(self.batch_id, "collecting")
        self._arm()
        return True

    def speech_ignored(self, turn_id: str) -> bool:
        """转写为噪音：恢复上一份判断，重新计时；重复结果不反复延长。

        上一次已经追问的未完成输入仍等待回答，不重复发送同一个追问。
        """
        if (self._closed or turn_id != self._turn_id or self._interpreted):
            return False
        self._invalidate()
        # 若上一句还在等待模型整理，噪音不能把它误判成“已经整理完”。
        # 恢复输入编号后，上一句晚到的模型结果仍可正常交付。
        (self._turn_id, self._speaking, self._stopped_at,
         self._interpreted) = self._previous_input
        if self._interpreted:
            self._stopped_at = self._clock()
        if self.chain.batch(self.batch_id).status == "waiting_for_clarification":
            return True
        self._arm()
        return True

    def _arm(self) -> None:
        if (self._closed or self._speaking or not self._interpreted
                or self._stopped_at is None or self._decision is None
                or self._timer is not None or self._emitted is not None):
            return
        version, complete, _, execute_now = self._decision
        batch = self.chain.batch(self.batch_id)
        if batch.version != version or batch.status != "collecting":
            return
        delay = 0.0 if complete and execute_now else (
            self.window_s if complete else self.incomplete_s
        )
        deadline = self._stopped_at + delay
        task = asyncio.create_task(
            self._wait(deadline, self._generation), name="collection-window-timer",
        )
        self._timer = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _wait(self, deadline: float, generation: int) -> None:
        try:
            # 即使模型整理花了很久，也只等用户说完后剩余的时间。
            while self._clock() < deadline:
                await self._sleep(deadline - self._clock())
            if (self._closed or generation != self._generation or self._speaking
                    or not self._interpreted or self._decision is None):
                return
            version, complete, question, _ = self._decision
            batch = self.chain.batch(self.batch_id)
            if batch.version != version or batch.status != "collecting":
                return
            if not complete:
                kind: Literal["ready", "confirmation_required", "clarification_required", "empty"] = "clarification_required"
                question = question or "你还想补充什么？"
                self.chain.set_collection_status(self.batch_id, "waiting_for_clarification")
            elif not self.chain.effective_tasks(self.batch_id):
                kind = "empty"
            elif self.chain.requires_confirmation(self.batch_id):
                kind = "confirmation_required"
            else:
                kind = "ready"
            event = CollectionEvent(kind, self.batch_id, version, self._decision_turn_id,
                                    generation, question)
            # 最后检查与发布之间没有 await，speech_started 不能插入其中。
            self._emitted = event
            self.events.put_nowait(event)
        except asyncio.CancelledError:
            pass
        finally:
            if self._timer is asyncio.current_task():
                self._timer = None

    def is_current(self, event: CollectionEvent) -> bool:
        """消费事件之前复核；已入队的旧事件也可能被新语音/修改作废。

        复核和登记提交之间不要 await；若必须让出事件循环，之后应重新复核。
        """
        batch = self.chain.batch(self.batch_id)
        return (not self._closed and not self._speaking and self._emitted == event
                and event.generation == self._generation and event.version == batch.version
                and batch.status == ("waiting_for_clarification"
                                     if event.kind == "clarification_required" else "collecting"))

    async def close(self) -> None:
        """只停止收集计时；不取消、执行或撤销任务。"""
        self._closed = True
        self._invalidate()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
