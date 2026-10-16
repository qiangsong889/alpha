"""在发送给大脑之前收集连续的用户指令；取消等待不会丢失未提交文字。"""

import asyncio
import logging
import math
import re
import time

logger = logging.getLogger("alpha.voice_pipeline")
EXECUTE = {"执行", "執行", "执行吧", "執行吧", "开始执行", "開始執行", "立即执行", "就这些", "就這些", "没了", "沒了"}
TRAILING = re.compile(r"(?:然后再|然後再|再帮我|再幫我|还有|還有|然后|然後|或者|等一下|等等|先别执行|先別執行|把|给|給|和|跟)$")


def unfinished(text: str) -> bool:
    """只识别明显未完的句尾；这是保守规则，不是完整的语义理解。"""
    text = text.rstrip(" ，。！？、,;；!?…\t\n")
    tail = re.split(r"[，。！？,;；!?]", text)[-1].strip()
    return bool(TRAILING.search(text) or (
        tail.startswith("把") and not re.search(
            r"加|放|买|買|删|刪|关|關|开|開|调|調|设|設|标|標|完成|拿|移|换|換|改|提醒|取消|撤", tail
        )
    ))


class RequestCollection:
    def __init__(self, window_s: float = 2.0, continuation_s: float = 5.0):
        if not math.isfinite(window_s) or window_s < 0:
            raise ValueError("ALPHA_COLLECTION_SECONDS 必须是非负有限数字")
        self.window_s = window_s
        if not math.isfinite(continuation_s) or continuation_s < 0:
            raise ValueError("ALPHA_CONTINUATION_SECONDS 必须是非负有限数字")
        self.continuation_s = continuation_s
        self._prompted = False
        self.prompt: str | None = None
        self.parts: list[str] = []
        self._speaking = False
        self._awaiting_transcript = False
        self._immediate = False
        self._changed = asyncio.Event()
        self._last_activity = time.monotonic()
        self._closed = False

    def speech_started(self) -> None:
        self._speaking = self._awaiting_transcript = True
        self._changed.set()
        if self.parts:
            logger.info("[收集] 检测到继续说话，暂停提交")

    def speech_stopped(self) -> None:
        self._speaking = False
        self._last_activity = time.monotonic()
        self._changed.set()

    def transcript_resolved(self) -> None:
        self._awaiting_transcript = False
        self._last_activity = time.monotonic()
        self._changed.set()

    def add(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        # 仅独立的执行口令或有明确分隔符的句尾口令生效，不截断“提醒我执行”。
        pieces = re.split(r"[，。！？、,;；!?\s]+", text)
        last = next((p for p in reversed(pieces) if p), "")
        immediate = last in EXECUTE
        if immediate:
            index = text.rfind(last)
            text = text[:index].strip(" ，。！？、,;；!?\t\n")
        if text:
            self.parts.append(text)
        self._immediate = immediate
        self._last_activity = time.monotonic()
        self._changed.set()
        incomplete = unfinished("；".join(self.parts))
        window = self.continuation_s if incomplete else self.window_s
        logger.info("[收集] %d 段待提交，%s", len(self.parts),
                    f"句尾未完整，等待 {window:g} 秒继续窗口" if incomplete else
                    "收到结束口令" if immediate else f"等待 {window:g} 秒安静窗口")

    async def take(self) -> str:
        while self.parts and not self._closed:
            self._changed.clear()
            if self._speaking or self._awaiting_transcript:
                # 最新一段还没转写时不能把旧任务先交出去；宁可等待，也不猜测输入已结束。
                await self._changed.wait()
                continue
            incomplete = unfinished("；".join(self.parts))
            if incomplete and self._prompted:
                # 已追问过一次。保留整批任务，等后续输入，不循环追问或自动执行。
                logger.info("[收集] 内容仍未完整，保留任务，暂停提交")
                return ""
            window = self.continuation_s if incomplete else self.window_s
            remaining = 0 if self._immediate and not incomplete else window - (time.monotonic() - self._last_activity)
            if remaining <= 0:
                if incomplete:
                    self._prompted = True
                    self.prompt = "你还没说完，我在听。接下来要做什么？"
                    logger.info("[收集] %g 秒继续窗口结束，追问一次，尚未提交", self.continuation_s)
                    return ""
                result = "；".join(self.parts)
                self.parts.clear()
                self._immediate = False
                self._prompted = False
                self.prompt = None
                logger.info("[收集] 提交完整请求：%s", result)
                return result
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass
        return ""

    def close(self) -> None:
        self._closed = True
        self.parts.clear()
        self.prompt = None
        self._changed.set()
