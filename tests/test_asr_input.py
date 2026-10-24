"""独立ASR接线测试：不打开麦克风、不连接服务端或业务工具。"""
import asyncio
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from core.costs import CostLedger
from voice.asr_client import AsrEvent, MODEL
from voice.asr_input import AsrInput, SileroActivity
from voice.qwen_alpha import Receiver, TurnCollector, input_mode, run_asr_connection
from voice.omni_transport import CostTracker
from voice.segmenter import FRAME, Segmenter, SpeechStarted, SegmentReady


class InputConfigTests(TestCase):
    def test_default_asr_and_explicit_omni_and_invalid(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(input_mode(), 'asr')
        with patch.dict('os.environ', {'QWEN_ALPHA_INPUT': 'omni'}):
            self.assertEqual(input_mode(), 'omni')
        with patch.dict('os.environ', {'QWEN_ALPHA_INPUT': 'asr'}):
            self.assertEqual(input_mode(), 'asr')
        with patch.dict('os.environ', {'QWEN_ALPHA_INPUT': 'bad'}):
            with self.assertRaises(ValueError):
                input_mode()


class AsrInputTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.submit = Mock()
        self.router = NS(turn_complete=AsyncMock(return_value=True),
                         interruption_and_turn=AsyncMock(return_value=(True, True)))
        self.collector = TurnCollector(self.submit, Mock(), self.router, mode='window', window_s=.001)
        self.addAsyncCleanup(self.collector.close)
        self.player = NS(gain=1, clear=Mock(), pause=Mock(), resume=Mock())
        self.bridge = NS(new_input=Mock(), changed=asyncio.Event())
        self.receiver = Receiver(None, self.player, self.collector, self.bridge, CostTracker('singapore'),
                                 'test', SileroActivity(), tts=NS(close=AsyncMock()))
        self.addAsyncCleanup(self.receiver.close)
        self.segmenter = Segmenter(lambda frame: 1.)
        self.adapter = AsrInput(self.receiver, self.segmenter)
        self.adapter.client = NS(submit=AsyncMock())

    async def local_segment(self, ident):
        await self.adapter.local_event(SpeechStarted(ident, 0, 0, .16))
        await self.adapter.local_event(SegmentReady(ident, b'\0' * 3200, 0, .7, .1, 'silence'))
        await self.adapter.notify('committed', ident)

    async def test_fake_local_and_asr_events_have_expected_order_and_local_ids(self):
        handle = AsyncMock()
        self.adapter.receiver = NS(handle=handle, activity=None)
        await self.local_segment('local-1')
        await self.adapter.result(AsrEvent('transcript', 'local-1', '关灯', .2))
        events = [call.args[0] for call in handle.call_args_list]
        self.assertEqual([e['type'] for e in events], [
            'input_audio_buffer.speech_started', 'input_audio_buffer.speech_stopped',
            'input_audio_buffer.committed', 'conversation.item.input_audio_transcription.completed'])
        self.assertEqual([e['item_id'] for e in events], ['local-1'] * 4)
        self.assertEqual(events[-1]['transcript'], '关灯')

    async def test_failure_resolves_pending_and_later_sentence_submits(self):
        await self.local_segment('bad')
        await self.adapter.result(AsrEvent('failed', 'bad', '等待超时', 5))
        self.assertFalse(self.collector.pending)
        await self.local_segment('next')
        await self.adapter.result(AsrEvent('transcript', 'next', '关客厅灯', .2))
        await asyncio.sleep(.03)
        self.submit.assert_called_once_with('关客厅灯')

    async def test_local_speech_ducks_then_pauses_without_clearing_or_cancelling(self):
        self.receiver.active = True
        self.receiver.current_report = NS(text='正在汇报')
        await self.adapter.local_event(SpeechStarted('local', 0, 0, .16))
        self.assertEqual(self.player.gain, .2)
        self.player.clear.assert_not_called()
        self.player.pause.assert_not_called()
        self.receiver.activity.voiced_ms = 450
        task = asyncio.create_task(self.receiver.confirm_speech())
        await asyncio.sleep(.025)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.player.pause.assert_called_once()
        self.player.clear.assert_not_called()
        self.assertFalse(self.receiver.cancelled)

    async def test_jev_side_chat_restores_playback_in_asr_mode(self):
        self.router.interruption_and_turn.return_value = (False, True)
        self.receiver.current_report = NS(text='正在汇报')
        await self.local_segment('side')
        await self.adapter.result(AsrEvent('transcript', 'side', '我吃过了', .2))
        await asyncio.sleep(.03)
        self.assertFalse(self.collector.pending)
        self.assertEqual(self.player.gain, 1.)
        self.player.resume.assert_called()
        self.player.clear.assert_not_called()
        self.submit.assert_not_called()

    async def test_alpha_hearing_itself_is_ignored_before_jev(self):
        # 2026-10-23：回声消除还没收敛时，Alpha 自己的话被转写出来，Jev 判成“对 Alpha 说话”。
        self.receiver.current_report = NS(text='购物清单还有牛奶、鸡蛋和水。您是指清单内容吗？')
        await self.local_segment('echo')
        with self.assertLogs('alpha.qwen_alpha', level='INFO') as logs:
            await self.adapter.result(AsrEvent('transcript', 'echo', '购物清单还有您。', .2))
            await asyncio.sleep(.03)
        self.assertIn('[回声忽略]', '\n'.join(logs.output))
        self.router.interruption_and_turn.assert_not_awaited()  # 不交给 Jev
        self.assertFalse(self.collector.pending)
        self.player.resume.assert_called()
        self.player.clear.assert_not_called()
        self.bridge.new_input.assert_not_called()
        self.submit.assert_not_called()

    async def test_a_real_interruption_during_playback_still_reaches_jev(self):
        self.bridge.report_interruption = Mock()
        report = NS(text='客厅灯开了，厨房灯也开了。', turn_id=None, kind='reply')
        self.receiver.current_report = report
        await self.local_segment('user')
        await self.adapter.result(AsrEvent('transcript', 'user', '厨房灯关了', .2))
        await asyncio.sleep(.03)
        self.router.interruption_and_turn.assert_awaited()
        self.bridge.new_input.assert_called_once()
        self.player.clear.assert_called()  # 真的停下了旧播报

    async def test_repeating_alpha_after_it_finished_is_not_treated_as_echo(self):
        # 只在 Alpha 说话期间比对：说完之后复述它的话是正常输入。
        self.receiver.current_report = None
        await self.local_segment('after')
        await self.adapter.result(AsrEvent('transcript', 'after', '可以听到。', .2))
        await asyncio.sleep(.03)
        self.assertTrue(self.collector.parts or self.submit.called)

    async def test_echo_tail_right_after_playback_is_ignored(self):
        # 2026-10-23：介绍完自己 118ms 后转写出“资料。”，正是最后两个字，被当成新请求。
        import time
        self.receiver.current_report = None
        self.receiver.finished_report = ('……查找和读取文档与邮件，也能联网搜索资料。', time.perf_counter())
        await self.local_segment('tail')
        with self.assertLogs('alpha.qwen_alpha', level='INFO') as logs:
            await self.adapter.result(AsrEvent('transcript', 'tail', '资料。', .2))
            await asyncio.sleep(.03)
        self.assertIn('刚说完', '\n'.join(logs.output))
        self.assertFalse(self.collector.parts)
        self.submit.assert_not_called()

    async def test_speech_starting_later_than_the_tail_window_is_normal_input(self):
        import time
        from voice.collection_rules import ECHO_TAIL_S
        self.receiver.current_report = None
        self.receiver.finished_report = ('……也能联网搜索资料。', time.perf_counter() - ECHO_TAIL_S - .1)
        await self.local_segment('later')
        await self.adapter.result(AsrEvent('transcript', 'later', '资料。', .2))
        await asyncio.sleep(.03)
        self.assertTrue(self.collector.parts or self.submit.called)

    async def test_a_quick_reply_that_is_not_alpha_s_words_is_kept(self):
        import time
        self.receiver.current_report = None
        self.receiver.finished_report = ('要我把客厅灯打开吗？', time.perf_counter())
        await self.local_segment('reply')
        await self.adapter.result(AsrEvent('transcript', 'reply', '好的谢谢', .2))
        await asyncio.sleep(.03)
        self.assertTrue(self.collector.parts or self.submit.called)

    async def test_tail_window_only_compares_with_the_end_of_the_sentence(self):
        # 回声尾巴只会是最后几个字；句子前面出现过的词，快速说出来仍然是你的输入。
        import time
        said = '购物清单上有牛奶、鸡蛋和水，另外明天早上八点半有一个会议，需要提前准备好材料。'
        self.receiver.current_report = None
        self.receiver.finished_report = (said, time.perf_counter())
        await self.local_segment('early')
        await self.adapter.result(AsrEvent('transcript', 'early', '牛奶鸡蛋', .2))
        await asyncio.sleep(.03)
        self.assertTrue(self.collector.parts or self.submit.called)

    async def test_forced_split_cannot_submit_before_physical_silence(self):
        probability = [1.]
        self.adapter.segmenter = Segmenter(lambda frame: probability[0], max_s=.32)
        for _ in range(10):
            await self.adapter.process(b'\0' * (FRAME * 2))
        segment = self.adapter.client.submit.call_args.args[0]
        self.assertEqual(segment.reason, 'max_duration')
        await self.adapter.result(AsrEvent('transcript', segment.seg_id, '先关客厅灯', .2))
        await asyncio.sleep(.03)
        self.submit.assert_not_called()
        # 强制切段后没有足够100ms续段：仍要在真实静音后释放占位，不能卡住。
        probability[0] = 0.
        for _ in range(20):
            await self.adapter.process(b'\0' * (FRAME * 2))
        await asyncio.sleep(.03)
        self.submit.assert_called_once_with('先关客厅灯')
        self.assertFalse(self.collector.pending)

    async def test_usage_is_authoritative_and_deduplicated_even_with_empty_transcript(self):
        ledger = CostLedger(':memory:')
        self.addCleanup(ledger.close)
        with patch('core.costs._ledger', ledger):
            await self.adapter.notify('uploading', 'usage')
            await self.adapter.notify('committed', 'usage')
            self.assertEqual(ledger.summary()['ASR']['unpriced'], 1)
            await self.adapter.notify('usage', 'usage', {'duration': 3, 'total_tokens': 7})
            await self.adapter.notify('usage', 'usage', {'duration': 3, 'total_tokens': 7})
            self.assertEqual(ledger.summary()['ASR'], {'usd': '0.00027', 'unpriced': 0, 'events': 1})
            await self.adapter.result(AsrEvent('failed', 'usage', '空转写', .2))
            row = ledger.db.execute('SELECT model,usage FROM cost_events').fetchone()
            self.assertEqual(row[0], MODEL)
            self.assertNotIn('sent_audio_seconds', row[1])

    async def test_long_continuation_segments_join_in_order_after_silence(self):
        probability = [1.]
        self.adapter.segmenter = Segmenter(lambda frame: probability[0], max_s=.32)
        for _ in range(10):
            await self.adapter.process(b'\0' * (FRAME * 2))
        first = self.adapter.client.submit.call_args.args[0]
        await self.adapter.result(AsrEvent('transcript', first.seg_id, '把牛奶加到清单', .2))
        # 续段持续128ms，未到最长时限，然后正常静音。
        for _ in range(4):
            await self.adapter.process(b'\0' * (FRAME * 2))
        await asyncio.sleep(.02)
        self.submit.assert_not_called()
        probability[0] = 0.
        for _ in range(20):
            await self.adapter.process(b'\0' * (FRAME * 2))
        second = self.adapter.client.submit.call_args.args[0]
        self.assertNotEqual(second.seg_id, first.seg_id)
        self.assertEqual(second.reason, 'max_duration')  # 15秒上限也包含句中静音。
        await self.adapter.result(AsrEvent('transcript', second.seg_id, '然后加鸡蛋', .2))
        # 第二次强制切段后仍要等满600ms实际静音。
        for _ in range(20):
            await self.adapter.process(b'\0' * (FRAME * 2))
        await asyncio.sleep(.03)
        self.submit.assert_called_once_with('把牛奶加到清单；然后加鸡蛋')

    async def test_asr_preconnects_before_mic_without_omni_or_upload_gate(self):
        trace = []
        async def ready():
            trace.append('ready')
        async def mic():
            trace.append('mic')
            raise RuntimeError('end-test')
        client = NS(ready=AsyncMock(side_effect=ready), close=AsyncMock())
        # 长任务在TaskGroup退出时一起取消。
        async def wait():
            await asyncio.Future()
        client.run = wait
        tts = NS(voice='Ethan', ready=AsyncMock(), close=AsyncMock())
        with patch('voice.asr_client.AsrClient', return_value=client), \
             patch('voice.segmenter.SileroVAD', return_value=lambda frame: 0), \
             patch('voice.qwen_alpha.QwenTTS', return_value=tts), \
             patch('voice.asr_input.AsrInput.microphone', side_effect=mic), \
             patch('voice.qwen_alpha.connect') as omni, \
             patch('voice.qwen_alpha.make_upload_gate') as gate, \
             patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'fake', 'QWEN_WORKSPACE_ID': 'test'}):
            with self.assertRaises(ExceptionGroup):
                await run_asr_connection(self.player, self.collector, self.bridge, CostTracker('singapore'))
        self.assertEqual(trace, ['ready', 'mic'])
        omni.assert_not_called()
        gate.assert_not_called()
        client.close.assert_awaited()
        tts.close.assert_awaited()


class EchoTextTests(TestCase):
    def test_echo_ratio_on_real_transcripts(self):
        from voice.collection_rules import echo_ratio, is_echo
        said = '购物清单还有牛奶、鸡蛋和水。您是指清单内容吗？'
        self.assertTrue(is_echo('可以听到。', '可以听到。'))
        self.assertTrue(is_echo('购物清单还有您。', said))       # 86%
        self.assertTrue(is_echo('还有牛奶鸡蛋。', said))         # 100%，忽略标点
        self.assertFalse(is_echo('厨房灯关了', '客厅灯开了，厨房灯也开了'))  # 80%：真实插话
        self.assertFalse(is_echo('看一下我的购物清单', said))
        self.assertFalse(is_echo('停', '停一下'))                # 太短，不下结论
        self.assertFalse(is_echo('购物清单', ''))
        self.assertAlmostEqual(echo_ratio('ALPHA 你好', 'alpha，你好'), 1.0)


class CallOnlyInterruptionTests(IsolatedAsyncioTestCase):
    """QWEN_ALPHA_INTERRUPT_MODE=call：播报期间只接受“Alpha……”、“停”或提问后的简短回答。"""
    INTRO = '我是 Alpha，你的家居和个人助手。能控制灯光、窗帘和空调，查天气，管理清单、日程和提醒。'
    local_segment = AsrInputTests.local_segment  # 只复用搭建代码，不继承 jev 模式的测试

    async def asyncSetUp(self):
        await AsrInputTests.asyncSetUp(self)
        self.receiver.interrupt_mode = 'call'
        self.bridge.report_interruption = Mock()
        self.report = NS(text=self.INTRO, turn_id=None, kind='reply')

    async def hear(self, ident, text):
        await self.local_segment(ident)
        with self.assertLogs('alpha.qwen_alpha', level='INFO') as logs:
            await self.adapter.result(AsrEvent('transcript', ident, text, .2))
            await asyncio.sleep(.03)
        return '\n'.join(logs.output)

    async def test_garbled_echo_during_playback_is_ignored_without_jev(self):
        # 2026-10-23 日志：回声被转写成“我是安好”“你的家居就是”，文字比对拦不住，Jev 判成对 Alpha 说话。
        self.receiver.current_report = self.report
        for ident, text in (('a', '我是安好。'), ('b', '你的家居就是。'), ('c', 'Alpha 你的家居就是。')):
            self.assertIn('[播报中忽略]', await self.hear(ident, text))
        self.router.interruption_and_turn.assert_not_awaited()
        self.bridge.new_input.assert_not_called()
        self.player.clear.assert_not_called()
        self.submit.assert_not_called()

    async def test_echo_right_after_playback_is_ignored(self):
        # “现在下午五点二十一分。”说完 244ms 后转写出“二十一分了。”，Alpha 回答“是的……”和自己对话。
        import time
        self.receiver.current_report = None
        self.receiver.finished_report = ('现在下午五点二十一分。', time.perf_counter())
        self.assertIn('[播报中忽略]', await self.hear('tail', '二十一分了。'))
        self.submit.assert_not_called()

    async def test_calling_alpha_interrupts_and_reaches_the_brain(self):
        self.receiver.current_report = self.report
        output = await self.hear('call', 'Alpha，把客厅灯关了。')
        self.assertIn('呼叫 Alpha', output)
        self.router.interruption_and_turn.assert_not_awaited()  # 叫了名字，不需要 Jev
        self.bridge.new_input.assert_called_once()
        self.player.clear.assert_called()
        await asyncio.sleep(.03)
        self.assertTrue(self.collector.parts or self.submit.called)

    async def test_a_misheard_name_still_counts_as_calling_alpha(self):
        # 中文识别常把 Alpha 认成“阿发”“奥特”；它们也必须直接当作叫名字，不再交给 Jev。
        self.receiver.current_report = self.report
        self.assertIn('呼叫 Alpha', await self.hear('misheard', '阿发，我的购物清单上有什么？'))
        self.router.interruption_and_turn.assert_not_awaited()
        self.bridge.new_input.assert_called_once()

    async def test_stop_only_stops_and_is_not_sent_to_the_brain(self):
        self.receiver.current_report = self.report
        self.receiver.inputs['stop'] = self.receiver.speech_input()
        self.assertIn('[停止播报]', await self.hear('stop', '停一下。'))
        self.bridge.new_input.assert_called_once()
        self.player.clear.assert_called()
        await asyncio.sleep(.03)
        self.assertFalse(self.collector.parts)
        self.submit.assert_not_called()

    async def test_short_answer_to_alpha_s_question_is_accepted_right_after_it(self):
        import time
        now = time.perf_counter()
        self.receiver.current_report = None
        self.receiver.finished_report = ('要我把客厅灯打开吗？', now)
        self.receiver.question_finished_at, self.receiver.question_deadline = now - .01, now + 10
        await self.local_segment('answer')
        await self.adapter.result(AsrEvent('transcript', 'answer', '好的', .2))
        await asyncio.sleep(.03)
        self.assertTrue(self.collector.parts or self.submit.called)

    async def test_speech_when_alpha_is_quiet_is_normal_input(self):
        self.receiver.current_report = None
        await self.local_segment('idle')
        await self.adapter.result(AsrEvent('transcript', 'idle', '把客厅灯关了', .2))
        await asyncio.sleep(.03)
        self.assertTrue(self.collector.parts or self.submit.called)

    async def test_playback_is_not_paused_for_local_voice(self):
        # 播报中的人声多半是回声，提前暂停只会让播报卡一下；jev 模式仍然暂停。
        self.receiver.current_report = self.report
        self.receiver.activity.voiced_ms = 500
        speaking = NS(speaking=True)
        for mode, paused in (('call', False), ('jev', True)):
            self.player.pause.reset_mock()
            self.receiver.interrupt_mode, self.receiver.paused = mode, False
            with patch.object(self.receiver, 'collector', speaking):
                task = asyncio.create_task(self.receiver.confirm_speech())
                await asyncio.sleep(.05)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(self.player.pause.called, paused, mode)


    async def test_playback_volume_is_not_ducked_for_local_voice(self):
        # 回声每次触发“人声开始”都会把音量压到 20%，转写被忽略后再恢复：听起来忽大忽小。
        self.receiver.current_report = self.report
        for mode, gain in (('call', 1), ('jev', .2)):
            self.receiver.interrupt_mode, self.player.gain = mode, 1
            await self.adapter.local_event(SpeechStarted(f'duck-{mode}', 0, 0, .16))
            self.assertEqual(self.player.gain, gain, mode)


class PlaybackCommandTests(TestCase):
    def test_commands(self):
        from voice.collection_rules import playback_command
        said = CallOnlyInterruptionTests.INTRO
        cases = {'停。': 'stop', '停一下': 'stop', 'Alpha 停': 'stop', '阿尔法，等一下': 'stop', '别说了': 'stop',
                 'Alpha，把客厅灯关了': 'call', 'alpha': 'call', '嘿 Alpha 现在几点': 'call',
                 'Alpha 你的家居就是。': None, '把客厅灯关了': None, '好的': None, '我是安好': None,
                 # ASR 实际认错过的名字写法
                 '奥特，现在几点了？': 'call', '阿发，我的购物清单上有什么': 'call', '阿尔发停': 'stop',
                 '奥特曼好看吗': None}
        for heard, expected in cases.items():
            self.assertEqual(playback_command(heard, said), expected, heard)

    def test_a_real_question_sharing_a_few_words_with_alpha_is_not_echo(self):
        from voice.collection_rules import playback_command
        # 重合 60%，但最长只连续 2 个字（“现在”）：是真心在问，不是回声。
        self.assertEqual(playback_command('奥特，现在几点了？', '现在下午五点二十一分。'), 'call')
        # 回声是原话连续的一段（“你的家居”）。
        self.assertIsNone(playback_command('Alpha 你的家居就是。', '我是 Alpha，你的家居和个人助手。'))
