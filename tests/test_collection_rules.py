"""当前语音入口的未完句规则；不调用分类服务。"""
import unittest
from voice.collection_rules import unfinished


class CollectionRuleTests(unittest.TestCase):
    def test_only_unfinished_tail_extends_window(self):
        for text in ['关灯，然后……', '再帮我', '还有。', '把牛奶', '等等']:
            self.assertTrue(unfinished(text), text)
        for text in ['再帮我关厨房灯', '关灯，然后把牛奶加到购物清单', '把灯关掉']:
            self.assertFalse(unfinished(text), text)
