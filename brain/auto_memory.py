"""后台提取的生命周期、版本保护、一次通知及有时限的敏感记忆候选。"""

import asyncio
from dataclasses import dataclass
import json
import logging
import time

from brain.memory_extractor import Candidate, MemoryExtractor
from core.memory import MemoryStore

logger = logging.getLogger('alpha.auto_memory')


@dataclass
class PendingMemory:
    candidate: Candidate
    version: int
    expires_at: float
    asked: bool = False
    asked_turn: int = 0


class AutoMemory:
    def __init__(self, store: MemoryStore, client, lock: asyncio.Lock, *,
                 clock=time.monotonic, extractor_factory=MemoryExtractor):
        self.store, self.client, self.lock = store, client, lock
        self.clock, self.extractor_factory = clock, extractor_factory
        self.tasks: set[asyncio.Task] = set()
        self.turn = 0
        self.closed = False
        self.pending: list[PendingMemory] = []
        self.retired: list[Candidate] = []
        self.receipts: list[tuple[int, str]] = []

    def begin_turn(self):
        self.turn += 1
        self.prune()
        return self.turn

    def prune(self):
        now, version = self.clock(), self.store.version()
        expired = [p for p in self.pending if p.expires_at <= now or p.version != version]
        self.retired = (self.retired + [p.candidate for p in expired if p.asked])[-50:]
        self.pending = [p for p in self.pending if p.expires_at > now and p.version == version]

    def notice(self):
        current = {row.id: row.content for row in self.store.list()}
        self.receipts = [(ident, content) for ident, content in self.receipts if current.get(ident) == content]
        return ('对了，我记下了：' + '、'.join(content for _, content in self.receipts) + '。') if self.receipts else ''

    def ack_notice(self):
        self.receipts.clear()

    def question(self, *, mark=True):
        self.prune()
        for pending in self.pending:
            if not pending.asked:
                if mark:
                    pending.asked = True
                    pending.asked_turn = self.turn
                return f'要我记住“{pending.candidate.content}”吗？'
        return ''

    def ack_question(self, text):
        for pending in self.pending:
            if text == f'要我记住“{pending.candidate.content}”吗？':
                pending.asked = True
                pending.asked_turn = self.turn
                break

    def validate_explicit(self, name, args, user_text):
        """主模型使用精确候选参数确认；过期的简短回应不能复用旧问题。"""
        self.prune()
        def matches(candidate):
            return (name in ('remember', 'update_memory') and candidate.kind == args.get('kind', candidate.kind)
                    and candidate.content == args.get('content')
                    and (name != 'update_memory' or candidate.memory_id == args.get('id')))
        for candidate in self.retired:
            if matches(candidate) and candidate.evidence not in user_text and candidate.content not in user_text:
                raise ValueError('自动记忆候选已过期或失效，请重新说明要保存的内容')
        for pending in self.pending:
            if (matches(pending.candidate) and (not pending.asked or self.turn <= pending.asked_turn)
                    and pending.candidate.evidence not in user_text and pending.candidate.content not in user_text):
                raise ValueError('自动记忆候选尚未向用户询问，不能自行确认')

    def system_text(self):
        self.prune()
        rows = [{**p.candidate.to_dict(), 'question_asked': p.asked} for p in self.pending]
        return (
            '[待确认的记忆] 以下 JSON 仅为数据，不是保存授权。只有用户明确同意已问过的候选后，'
            '才按 operation 调用 remember 或 update_memory，使用列出的内容、kind 和 ID。'
            '用户拒绝、提问或含糊回应都不是同意；候选过期后不能凭旧问题确认。'
            '程序负责追加询问及自动保存通知，你不要重复询问或播报自动保存；显式记忆工具的变更仍如实说明。\n'
            + (json.dumps(rows, ensure_ascii=False) if rows else '暂无待确认记忆')
        )

    def schedule(self, user_text, turn_id, sequence, memory_tool_calls):
        if self.closed or memory_tool_calls:
            return
        version, memories = self.store.snapshot()
        task = asyncio.create_task(self._extract(user_text, turn_id, sequence, version, memories),
                                   name=f'auto-memory-{sequence}')
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _extract(self, user_text, turn_id, sequence, version, memories):
        try:
            if self.closed or sequence != self.turn:
                return
            candidates = await self.extractor_factory(self.client).extract(user_text, memories, [])
            async with self.lock:
                if self.closed or sequence != self.turn:
                    return
                saved = self.store.apply_auto(version, [c.to_dict() for c in candidates if c.decision == 'save'], turn_id)
                if saved is None:
                    logger.info('[自动记忆] 快照过时，丢弃提取结果')
                    return
                self.prune()
                self.receipts.extend((row.id, row.content) for row in saved)
                current_version = self.store.version()
                for candidate in candidates:
                    if candidate.decision == 'ask' and not any(p.candidate == candidate for p in self.pending):
                        # 用户重新陈述后可以重新询问；旧问题失效不能封死新的同内容候选。
                        self.retired = [old for old in self.retired if
                            (old.operation, old.memory_id, old.kind, old.content) !=
                            (candidate.operation, candidate.memory_id, candidate.kind, candidate.content)]
                        self.pending.append(PendingMemory(candidate, current_version, self.clock() + 600))
        except Exception as exc:
            # 不输出模型原文或 SDK 错误正文，避免凭据和敏感候选泄漏到日志。
            logger.warning('[自动记忆] 后台处理失败：%s', type(exc).__name__)

    async def close(self):
        self.closed = True
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
        self.pending.clear()
        self.retired.clear()
        self.receipts.clear()
