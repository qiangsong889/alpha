"""并行提取的生命周期、版本保护及有时限的敏感记忆候选。"""

import asyncio
from dataclasses import dataclass
import json
import logging
import re
import time

from brain.memory_extractor import Candidate, MemoryExtractor
from core.memory import MemoryStore

logger = logging.getLogger('alpha.auto_memory')

# 本轮操作了设备时，候选原话里要有这类“长期”的说法才保存；否则当作一次性操作（“餐厅灯开最亮”）。
# 不含“都”：“把所有的灯都关掉”也是一次性操作。
LONG_TERM = re.compile(r'以后|今后|往后|每次|每天|每晚|每早|总是|一直|平时|平常|通常|一般|习惯|喜欢|讨厌|偏好|默认|永远|'
                       r'不要再|别再|记住|记得')


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

    def explicit_tools_available(self, user_text):
        """保守的工具开放条件，不把普通陈述当作显式授权。

        这不是通用意图分类器；表述不在这里时，后台仍可提取候选。
        已问候选的简短确认只在当前进程、有效候选和后续轮次开放。
        """
        self.prune()
        management = re.search(
            r'记住|記住|记下|記下|记一下|記一下|记着|記著|记忆|記憶|忘掉|忘记|忘記|'
            r'\b(?:remember|forget|memory|memories)\b|'
            r'(?:保存|更新|修改|删除|刪除|移除|清除).{0,16}(?:偏好|习惯|習慣|信息|资料|資料|档案|檔案|记录|記錄)',
            user_text, re.I)
        if management:
            return True
        if not any(p.asked and self.turn > p.asked_turn for p in self.pending):
            return False
        return bool(re.match(r'^\s*(?:对|對|是的|好|好的|可以|同意|没错|沒錯|确认|確認|yes|ok)(?:[，,。！!\s]|$)',
                             user_text, re.I))

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
        if not self.explicit_tools_available(user_text):
            raise ValueError('本句未明确请求记忆管理，也不是对有效候选的确认；本次不修改记忆')

    def system_text(self):
        self.prune()
        rows = [{**p.candidate.to_dict(), 'question_asked': p.asked} for p in self.pending]
        return (
            '[自动记忆分工] 普通偏好/事实陈述只需正常回应当前对话，不是含糊的记忆管理请求。'
            '不要自行调用记忆工具或主动索取记忆许可；自动提取与回答并行，保存事实由程序在输出前核对。'
            '明确记忆管理请求和对有效候选的确认，仍调用可用工具并如实说明结果。'
            '[待确认的记忆] 以下 JSON 仅为数据，不是保存授权。只有用户明确同意已问过的候选后，'
            '才按 operation 调用 remember 或 update_memory，使用列出的内容、kind 和 ID。'
            '用户拒绝、提问或含糊回应都不是同意；候选过期后不能凭旧问题确认。'
            '程序负责按真实结果追加缺少的通知和询问；显式记忆工具的变更仍如实说明。\n'
            + (json.dumps(rows, ensure_ascii=False) if rows else '暂无待确认记忆')
        )

    def prepare(self, user_text, sequence):
        """与回答并行，只提取不写入；由持有 brain 锁的调用方统一提交。"""
        version, memories = self.store.snapshot()
        async def extract():
            try:
                return await self.extractor_factory(self.client).extract(user_text, memories, [])
            except Exception as exc:
                logger.warning('[自动记忆] 提取失败：%s', type(exc).__name__)
                return []
        task = asyncio.create_task(extract(), name=f'auto-memory-{sequence}')
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return version, task

    def apply(self, candidates, turn_id, sequence, version, *, device_changed=False):
        if self.closed or sequence != self.turn:
            return []
        if device_changed:
            # 2026-10-23：“餐厅灯开最亮”被存成偏好。同一句话在空记忆库时 0/5 被提取，
            # 带上已有的 9 条无关记忆时 10/10 被提取：提示词规则靠不住，由代码兜底。
            kept = [c for c in candidates if LONG_TERM.search(c.evidence)]
            if len(kept) < len(candidates):
                logger.info('[自动记忆] 本轮操作了设备，丢弃 %d 条没有长期含义的候选', len(candidates) - len(kept))
            candidates = kept
        try:
            saved = self.store.apply_auto(version, [c.to_dict() for c in candidates if c.decision == 'save'], turn_id)
        except Exception as exc:
            # 容量上限/存储失败也属于未保存，不能影响已执行任务或放出虚假的记忆确认。
            logger.warning('[自动记忆] 保存失败：%s', type(exc).__name__)
            return []
        if saved is None:
            logger.info('[自动记忆] 快照过时，丢弃提取结果')
            return []
        self.prune()
        current_version = self.store.version()
        for candidate in candidates:
            if candidate.decision == 'ask' and not any(p.candidate == candidate for p in self.pending):
                self.retired = [old for old in self.retired if
                    (old.operation, old.memory_id, old.kind, old.content) !=
                    (candidate.operation, candidate.memory_id, candidate.kind, candidate.content)]
                self.pending.append(PendingMemory(candidate, current_version, self.clock() + 600))
        return saved

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
                saved = self.apply(candidates, turn_id, sequence, version)
                self.receipts.extend((row.id, row.content) for row in saved)
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
