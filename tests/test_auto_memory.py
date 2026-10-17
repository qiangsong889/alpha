"""后台提取的版本保护和通知；使用假的提取器，不请求模型。"""
import asyncio
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock

from brain.auto_memory import AutoMemory
from brain.memory_extractor import Candidate
from core.memory import MemoryStore


def candidate(decision='save', content='不吃香菜', **changes):
    return Candidate(decision, changes.get('operation', 'add'), changes.get('kind', 'preference'),
                     content, changes.get('evidence', content), changes.get('memory_id'))


class AutoMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = MemoryStore(':memory:')
        self.addCleanup(self.store.close)
        self.now = 0
        self.extract = AsyncMock(return_value=[candidate()])
        self.auto = AutoMemory(self.store, None, asyncio.Lock(), clock=lambda: self.now,
                               extractor_factory=lambda _: NS(extract=self.extract))
        self.addAsyncCleanup(self.auto.close)

    async def finish(self):
        await asyncio.gather(*list(self.auto.tasks))

    async def test_save_source_and_notice_once(self):
        seq = self.auto.begin_turn()
        self.auto.schedule('我不吃香菜', 'turn', seq, [])
        await self.finish()
        row = self.store.list()[0]
        self.assertEqual((row.source, row.source_turn), ('auto', 'turn'))
        self.assertEqual(self.auto.notice(), '对了，我记下了：不吃香菜。')
        self.auto.ack_notice()
        self.assertEqual(self.auto.notice(), '')

    async def test_changes_during_extraction_discard_saves_and_questions(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def blocked(*args):
            started.set()
            await release.wait()
            return [candidate(), candidate('ask', '对花生过敏', kind='fact')]
        self.extract.side_effect = blocked
        self.auto.schedule('我不吃香菜，对花生过敏', 'turn', self.auto.begin_turn(), [])
        await started.wait()
        ident = self.store.add('空调二十六度', 'preference', 'newer')
        self.store.update(ident, '空调二十四度')
        release.set()
        await self.finish()
        self.assertEqual([r.content for r in self.store.list()], ['空调二十四度'])
        self.assertEqual(self.auto.pending, [])
        self.assertEqual(self.auto.notice(), '')

    async def test_ask_never_saves_and_expires_at_ten_minutes(self):
        self.extract.return_value = [candidate('ask', '对花生过敏', kind='fact')]
        self.auto.schedule('我对花生过敏', 'turn', self.auto.begin_turn(), [])
        await self.finish()
        self.assertEqual(self.store.list(), [])
        self.auto.begin_turn()
        self.assertIn('对花生过敏', self.auto.question())
        self.assertEqual(self.auto.question(), '')
        self.assertIn('对花生过敏', self.auto.system_text())
        self.now = 600
        self.assertEqual(self.auto.question(), '')
        self.assertNotIn('对花生过敏', self.auto.system_text())
        self.auto.begin_turn()
        with self.assertRaisesRegex(ValueError, '过期'):
            self.auto.validate_explicit('remember', {'kind': 'fact', 'content': '对花生过敏'}, '对')

    async def test_memory_tools_skip_without_creating_task(self):
        self.auto.schedule('记住我不吃香菜', 'turn', self.auto.begin_turn(), ['remember'])
        self.assertEqual(self.auto.tasks, set())
        self.extract.assert_not_called()

    async def test_expired_content_can_be_asked_again_after_user_restates_it(self):
        self.extract.return_value = [candidate('ask', '对花生过敏', kind='fact')]
        self.auto.schedule('对花生过敏', 'first', self.auto.begin_turn(), [])
        await self.finish()
        self.auto.question()
        self.now = 600
        self.auto.schedule('对花生过敏', 'again', self.auto.begin_turn(), [])
        await self.finish()
        self.assertTrue(self.auto.question())
        self.auto.begin_turn()
        self.auto.validate_explicit('remember', {'kind': 'fact', 'content': '对花生过敏'}, '对')

    async def test_new_turn_discards_late_result_and_close_cancels(self):
        started = asyncio.Event()
        async def blocked(*args):
            started.set()
            await asyncio.Future()
        self.extract.side_effect = blocked
        self.auto.schedule('我不吃香菜', 'turn', self.auto.begin_turn(), [])
        await started.wait()
        tasks = list(self.auto.tasks)
        await self.auto.close()
        self.assertTrue(all(t.cancelled() for t in tasks))
        self.assertEqual(self.store.list(), [])

    async def test_newer_input_discards_previous_result(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def blocked(*args):
            started.set()
            await release.wait()
            return [candidate()]
        self.extract.side_effect = blocked
        self.auto.schedule('我不吃香菜', 'turn', self.auto.begin_turn(), [])
        await started.wait()
        self.auto.begin_turn()
        release.set()
        await self.finish()
        self.assertEqual(self.store.list(), [])
