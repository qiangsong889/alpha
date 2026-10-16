"""将 Jev 的完整性分类接到 LiveKit 1.8.8 的轮次检测协议。

不收集第二份文字、不主动执行工具；后续 STT 片段的合并和静音计时仍由
LiveKit 完成。0/1 是适配器控制值，不是 Jev 的校准概率。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from livekit.agents import LanguageCode, llm

if TYPE_CHECKING:
    from voice_pipeline.search_router import JevSearchRouter

logger = logging.getLogger("alpha.voice_pipeline")


class JevTurnDetector:
    def __init__(self, router: JevSearchRouter, fallback):
        self.router = router
        self.fallback = fallback

    @property
    def model(self) -> str:
        return self.router.model

    @property
    def provider(self) -> str:
        return "jev"

    async def supports_language(self, language: LanguageCode | None) -> bool:
        return True

    async def unlikely_threshold(self, language: LanguageCode | None) -> float:
        return 0.5

    async def predict_end_of_turn(self, chat_ctx: llm.ChatContext,
                                  *, timeout: float | None = None) -> float:
        messages = [{"role": item.role, "content": item.text_content or ""}
                    for item in chat_ctx.items if isinstance(item, llm.ChatMessage)
                    and item.role in {"user", "assistant"}]
        # 框架可能把同一轮的多个最终转写保存为相邻 user 消息。
        # 必须一起判断，不能只拿‘还有鸡蛋’而漏掉之前的‘把牛奶’。
        parts: list[str] = []
        while messages and messages[-1]["role"] == "user":
            parts.insert(0, messages.pop()["content"])
        text = " ".join(parts).strip()
        if not text:
            return 0.0

        async def predict() -> float:
            decision = await self.router.turn_complete(text, messages)
            if decision is not None:
                logger.info("[轮次判断] Jev %s", "完整，使用短静音窗口" if decision
                            else "未完，延长静音窗口")
                return 1.0 if decision else 0.0
            logger.info("[轮次判断] Jev 不确定或不可用，回退本地模型")
            # 原模型阈值不一定是 0.5，先按它自己的阈值解释，再转为控制值。
            language = LanguageCode("zh")
            if not await self.fallback.supports_language(language):
                return 0.0
            probability = await self.fallback.predict_end_of_turn(chat_ctx)
            threshold = await self.fallback.unlikely_threshold(language)
            return float(threshold is not None and probability >= threshold)

        try:
            if timeout is None:
                return await predict()
            async with asyncio.timeout(timeout):
                return await predict()
        except asyncio.CancelledError:
            raise  # 用户继续说话时，LiveKit 取消旧预测；不能提交旧的半句话。
        except Exception:
            logger.warning("[轮次判断] 预测不可用，使用长静音上限")
            return 0.0
