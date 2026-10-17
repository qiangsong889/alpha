"""持久记忆的容量、验证、去重及事务行为，不调用外部服务。"""

from pathlib import Path
import sqlite3
import tempfile
import unittest

from core.memory import MemoryStore


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(':memory:')
        self.addCleanup(self.store.close)

    def test_persistence_and_operations_table_coexist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'operations.db'
            db = sqlite3.connect(path)
            db.execute('CREATE TABLE operations(id TEXT)')
            db.close()
            first = MemoryStore(path)
            ident = first.add('空调喜欢 26 度', 'preference', 'turn-1')
            first.close()
            second = MemoryStore(path)
            try:
                row = second.list()[0]
                self.assertEqual((row.id, row.source_turn), (ident, 'turn-1'))
                self.assertEqual(second._db.execute('SELECT count(*) FROM operations').fetchone()[0], 0)
            finally:
                second.close()

    def test_trim_and_duplicate_returns_same_id(self):
        ident = self.store.add(' 空调喜欢 26 度 ', 'preference', 'first')
        self.assertEqual(self.store.add('空调喜欢 26 度', 'preference', 'second'), ident)
        self.assertEqual(len(self.store.list()), 1)
        self.assertEqual(self.store.list()[0].source_turn, 'first')

    def test_empty_kind_and_secrets_rejected_on_add_and_update(self):
        ident = self.store.add('太太叫小林', 'fact', 'turn')
        for content in (' ', '密码是 abc123', 'API_KEY=secretvalue', '验证码：123456',
                        'sk-abcdefghijklmnop', '-----BEGIN PRIVATE KEY-----'):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    self.store.add(content, 'fact', 'turn')
                with self.assertRaises(ValueError):
                    self.store.update(ident, content)
        with self.assertRaises(ValueError):
            self.store.add('偏好', 'unknown', 'turn')
        self.assertEqual(self.store.list()[0].content, '太太叫小林')

    def test_normal_secret_related_text_is_allowed_on_add_and_update(self):
        for content in ('我用 1Password 管理密码', '我最喜欢的书是 The Secret Garden',
                        '验证码一般发到我的 Google Voice 号码', '密码管理器是 1Password',
                        'bearer bond', 'bearer bond 是什么', 'Bearer 是一个英文单词'):
            with self.subTest(content=content):
                ident = self.store.add(content, 'fact', 'turn')
                self.store.update(ident, content + '。')
                self.assertEqual(self.store.list()[-1].content, content + '。')

    def test_separators_and_bearer_token_boundary_on_add_and_update(self):
        ident = self.store.add('普通事实', 'fact', 'turn')
        for content in ('密码是 abc123', 'api key: sk-xxxx', 'api key: sk-xxxxxxxxxx',
                        'Bearer eyJhbGciOiJIUzI1NiJ9.xxxx', '密码 为 abc123',
                        'Bearer ' + 'a' * 16, 'bearer Ab01._~+/=-xyz123456',
                        'AIza' + 'a' * 20):
            with self.subTest(content=content):
                with self.assertRaisesRegex(ValueError, '秘密信息'):
                    self.store.add(content, 'fact', 'turn')
                with self.assertRaisesRegex(ValueError, '秘密信息'):
                    self.store.update(ident, content)
        # 这里只验证格式边界；是否允许存储任意秘密仍由模型规则约束。
        self.store.add('Bearer ' + 'a' * 15, 'fact', 'turn')
        self.assertEqual(self.store.list()[0].content, '普通事实')

    def test_missing_ids_error_and_deleted_ids_are_not_reused(self):
        for method, args in ((self.store.update, (99, '内容')), (self.store.delete, (99,))):
            with self.assertRaisesRegex(ValueError, '不存在'):
                method(*args)
        ident = self.store.add('旧事实', 'fact', 'turn')
        self.store.delete(ident)
        self.assertGreater(self.store.add('新事实', 'fact', 'turn'), ident)

    def test_count_limit_and_duplicate_at_capacity(self):
        for index in range(50):
            self.store.add(f'偏好{index}', 'preference', 'turn')
        self.assertEqual(self.store.add('偏好0', 'preference', 'turn'), 1)
        with self.assertRaisesRegex(ValueError, '记忆已满'):
            self.store.add('更多', 'fact', 'turn')
        self.store.update(1, '新的偏好')
        self.assertEqual(len(self.store.list()), 50)

    def test_total_character_limit_applies_to_updates_and_recovers_after_delete(self):
        ident = self.store.add('甲' * 1999, 'fact', 'turn')
        other = self.store.add('乙', 'fact', 'turn')
        with self.assertRaisesRegex(ValueError, '记忆已满'):
            self.store.add('丙', 'fact', 'turn')
        with self.assertRaisesRegex(ValueError, '记忆已满'):
            self.store.update(other, '乙乙')
        self.assertEqual(self.store.list()[1].content, '乙')
        self.store.delete(ident)
        self.store.update(other, '乙乙')
        self.assertEqual(self.store.list()[0].content, '乙乙')

    def test_update_collision_does_not_destroy_original(self):
        first = self.store.add('甲', 'fact', 'turn')
        self.store.add('乙', 'fact', 'turn')
        with self.assertRaisesRegex(ValueError, '相同记忆'):
            self.store.update(first, '乙')
        self.assertEqual([r.content for r in self.store.list()], ['甲', '乙'])
