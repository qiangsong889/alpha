"""独立ASR接线测试：不打开麦克风、不连接服务端或业务工具。"""
import asyncio
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from core.costs import CostLedger
from voice.asr_client import AsrEvent, MODEL
from voice.asr_input import AsrInput, SileroActivity
from voice.qwen_alpha import Receiver, TurnCollector, input_mode, run_asr_connection
from voice.qwen_realtime import CostTracker
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
