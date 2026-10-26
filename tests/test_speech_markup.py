"""朗读标记：语气和声音事件从回答里分开，用户听到的文字不带标记。"""

from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock

from voice.qwen_alpha import BrainBridge, tts_sentences
from voice.speech_markup import plain_speech, split_style


class MarkupTests(TestCase):
    def test_split_style_finds_the_marker_anywhere_and_keeps_events(self):
        self.assertEqual(split_style('[严肃] 门没锁。'), ('门没锁。', 'serious'))
        self.assertEqual(split_style('查到了，[严肃]燃气报警了。'), ('查到了，燃气报警了。', 'serious'))
        self.assertEqual(split_style('[笑] 又改闹钟了。'), ('[笑] 又改闹钟了。', None))

    def test_plain_speech_removes_every_marker_but_ordinary_brackets(self):
        self.assertEqual(plain_speech('[严肃][叹气] 门没锁。(Laugh) OK.'), '门没锁。OK.')
        self.assertEqual(plain_speech('[会议] 改到三点 (周四)。'), '[会议] 改到三点 (周四)。')

    def test_marker_only_fragment_joins_the_next_sentence(self):
        self.assertEqual(tts_sentences('[笑]。真的假的？'), ['[笑]。真的假的？'])


class BridgeMarkupTests(IsolatedAsyncioTestCase):
    async def test_report_text_is_plain_and_speech_keeps_events(self):
        bridge = BrainBridge(Mock(), 'http://brain')
        bridge.say('[严肃][叹气] 前门没锁。')
        report = bridge.reports[0]
        self.assertEqual((report.text, report.speech, report.style), ('前门没锁。', '[叹气] 前门没锁。', 'serious'))
        bridge.say(None, kind='tone')
        self.assertIsNone(bridge.reports[1].speech)

