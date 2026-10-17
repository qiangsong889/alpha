"""混合语音入口不打开音频设备、不访问真实工具。"""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

import httpx2
from voice.qwen_alpha import BrainBridge, Receiver, TurnCollector, session_config
from voice.qwen_realtime import CostTracker


class ConfigTests(TestCase):
    def test_no_tools_and_no_automatic_responses(self):
        config = session_config('Tina')
        self.assertEqual(config['tools'], [])
        self.assertFalse(config['turn_detection']['create_response'])
        self.assertFalse(config['turn_detection']['interrupt_response'])


class CollectionTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.submit, self.prompt = Mock(), Mock()
        self.router = NS(turn_complete=AsyncMock(return_value=True))
        self.gate = TurnCollector(self.submit, self.prompt, self.router, window_s=.01, continuation_s=.01)
        self.addAsyncCleanup(self.gate.close)

    async def settle(self):
        await asyncio.sleep(.03)

    async def test_long_request_and_correction_are_submitted_as_one(self):
        self.gate.started('1')
        self.gate.stopped()
        self.gate.transcript('1', '把牛奶加到清单')
        self.gate.started('2')
        await self.settle()
        self.submit.assert_not_called()
        self.gate.stopped()
        self.gate.transcript('2', '明天八点提醒开会，牛奶换成酱油，执行吧')
        await self.settle()
        self.submit.assert_called_once_with('把牛奶加到清单；明天八点提醒开会，牛奶换成酱油')

    async def test_no_submit_until_all_transcripts_arrive_and_duplicate_events_do_not_repeat(self):
        self.gate.started('1')
        self.gate.stopped()
        self.gate.committed('2')
        self.gate.transcript('1', '关客厅灯')
        await self.settle()
        self.submit.assert_not_called()
        self.gate.transcript('2', '然后关窗帘')
        self.gate.committed('2')
        self.gate.transcript('2', '然后关窗帘')
        await self.settle()
        self.submit.assert_called_once_with('关客厅灯；然后关窗帘')

    async def test_incomplete_prompts_once_without_executing_then_can_continue(self):
        self.router.turn_complete.return_value = False
        self.gate.transcript('1', '把牛奶')
        await self.settle()
        await self.settle()
        self.prompt.assert_called_once()
        self.submit.assert_not_called()
        self.gate.transcript('2', '加到购物清单，执行吧')
        await self.settle()
        self.submit.assert_called_once_with('把牛奶；加到购物清单')

    async def test_out_of_order_transcripts_preserve_correction_order_and_end_command(self):
        self.gate.started('1')
        self.gate.stopped()
        self.gate.started('2')
        self.gate.stopped()
        self.gate.transcript('2', '牛奶改成酱油，执行吧')
        await self.settle()
        self.submit.assert_not_called()
        self.gate.transcript('1', '加牛奶')
        await self.settle()
        self.submit.assert_called_once_with('加牛奶；牛奶改成酱油')

    async def test_speech_cancels_slow_classification_before_dispatch(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def slow(*args):
            started.set()
            await release.wait()
            return True
        self.router.turn_complete.side_effect = slow
        self.gate.transcript('1', '开客厅灯')
        await started.wait()
        self.gate.started('2')
        release.set()
        await self.settle()
        self.submit.assert_not_called()

    async def test_reconnect_preserves_unsubmitted_text_but_requires_new_input(self):
        self.gate.transcript('1', '加牛奶')
        self.gate.disconnected()
        await self.settle()
        self.submit.assert_not_called()
        self.gate.transcript('2', '执行吧')
        await self.settle()
        self.submit.assert_called_once_with('加牛奶')

    async def test_unknown_classification_falls_back_to_complete_text_and_short_yes_survives(self):
        self.router.turn_complete.return_value = None
        self.gate.transcript('1', '对')
        await self.settle()
        self.submit.assert_called_once_with('对')


class BridgeTests(IsolatedAsyncioTestCase):
    async def test_late_old_answer_is_not_queued_after_new_input(self):
        started, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def handle(request):
            text = json.loads(request.content)['text']
            calls.append(text)
            if text == '旧任务':
                started.set()
                await release.wait()
            answer = '旧的长汇报，邮件发了。' if text == '旧任务' else '发了。'
            return httpx2.Response(200, text=json.dumps({'type': 'delta', 'text': answer}) + '\n' +
                                   json.dumps({'type': 'done'}))
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            bridge = BrainBridge(client, 'http://brain')
            task = asyncio.create_task(bridge.run())
            try:
                bridge.submit('旧任务')
                await asyncio.wait_for(started.wait(), 1)
                bridge.new_input()
                bridge.submit('发了吗？')
                release.set()
                await asyncio.wait_for(bridge.requests.join(), 1)
                self.assertEqual([r.text for r in bridge.reports], ['发了。'])
                self.assertEqual(calls, ['旧任务', '发了吗？'])
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_new_input_clears_waiting_reports_but_not_business_queue(self):
        bridge = BrainBridge(None, 'http://brain')
        bridge.say('邮件发了，要创建文档吗？')
        bridge.submit('旧任务')
        bridge.new_input()
        self.assertEqual(list(bridge.reports), [])
        self.assertEqual(bridge.requests.qsize(), 1)

    async def execute(self, events):
        calls = []
        def handle(request):
            calls.append(json.loads(request.content))
            return httpx2.Response(200, text='\n'.join(json.dumps(e) for e in events))
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            bridge = BrainBridge(client, 'http://brain')
            task = asyncio.create_task(bridge.run())
            bridge.submit('任务')
            await asyncio.wait_for(bridge.requests.join(), 1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(calls, [{'text': '任务'}])
        return [report.text for report in bridge.reports]

    async def test_web_progress_is_sent_before_final_result(self):
        self.assertEqual(await self.execute([
            {'type': 'delta', 'text': '我查一下。'}, {'type': 'delta', 'text': '查到了。'},
            {'type': 'done'}]), ['我查一下。', '查到了。'])

    async def test_device_success_becomes_local_tone(self):
        self.assertEqual(await self.execute([{'type': 'signal', 'name': 'ha_done'}, {'type': 'done'}]), [None])

    async def test_broken_stream_is_unknown_and_not_retried(self):
        reports = await self.execute([{'type': 'delta', 'text': '未完成'}])
        self.assertIn('可能已经生效', reports[0])
        self.assertEqual(len(reports), 1)


class ReceiverTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = NS(send=AsyncMock())
        self.player = NS(clear=Mock(), append=Mock(), finish=Mock(), gain=1, resume=Mock(), pause=Mock(),
                         wait_played=AsyncMock(return_value=True))
        self.gate = TurnCollector(Mock(), Mock(), NS(turn_complete=AsyncMock(return_value=True)), window_s=10)
        self.addAsyncCleanup(self.gate.close)
        self.bridge = BrainBridge(None, 'http://brain')
        self.receiver = Receiver(self.ws, self.player, self.gate, self.bridge, CostTracker('singapore'), 'test')

    async def test_user_input_alone_never_creates_response_or_calls_brain(self):
        await self.receiver.handle({'type': 'input_audio_buffer.speech_started', 'item_id': '1'})
        await self.receiver.handle({'type': 'input_audio_buffer.speech_stopped'})
        await self.receiver.handle({'type': 'conversation.item.input_audio_transcription.completed',
                                    'item_id': '1', 'transcript': '加牛奶'})
        self.ws.send.assert_not_awaited()
        self.assertTrue(self.bridge.requests.empty())
        self.assertEqual(self.gate.parts, ['加牛奶'])

    async def test_pending_response_is_cancelled_even_before_created_ack(self):
        self.receiver.active = True
        await self.receiver.interrupt()
        await self.receiver.handle({'type': 'response.created', 'response': {'id': 'late'}})
        await self.receiver.handle({'type': 'response.audio.delta', 'response_id': 'late',
                                    'item_id': 'a', 'delta': 'AAA='})
        self.player.append.assert_not_called()
        self.assertIn('late', self.receiver.ignored)

    async def test_unsolicited_qwen_response_cannot_act_or_speak(self):
        await self.receiver.handle({'type': 'response.created', 'response': {'id': 'unexpected'}})
        await self.receiver.handle({'type': 'response.audio.delta', 'response_id': 'unexpected',
                                    'item_id': 'a', 'delta': 'AAA='})
        self.player.append.assert_not_called()
        self.assertEqual(json.loads(self.ws.send.call_args.args[0])['type'], 'response.cancel')

    async def test_noise_after_local_pause_resumes_without_interrupting(self):
        self.receiver.active = True
        self.receiver.paused = True
        await self.receiver.handle({'type': 'conversation.item.input_audio_transcription.completed',
                                    'item_id': 'noise', 'transcript': '咳嗽'})
        self.player.resume.assert_called_once()
        self.player.clear.assert_not_called()
        self.ws.send.assert_not_awaited()

    async def test_response_create_only_follows_queued_brain_report(self):
        self.bridge.say('已添加酱油。')
        task = asyncio.create_task(self.receiver.speak())
        try:
            for _ in range(10):
                await asyncio.sleep(0)
                if self.ws.send.await_count >= 3:
                    break
            calls = [json.loads(c.args[0]) for c in self.ws.send.call_args_list]
            self.assertEqual([c['type'] for c in calls], ['session.update', 'conversation.item.create', 'response.create'])
            self.assertIn('已添加酱油', calls[0]['session']['instructions'])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_old_playback_finishing_does_not_pop_the_new_report(self):
        started, release = asyncio.Event(), asyncio.Event()
        played = []
        async def play(text):
            played.append(text)
            if text == '旧汇报':
                started.set()
                await release.wait()
        self.receiver._play_report = play
        self.bridge.say('旧汇报')
        task = asyncio.create_task(self.receiver.speak())
        try:
            await asyncio.wait_for(started.wait(), 1)
            self.bridge.new_input()
            self.bridge.say('发了。')
            release.set()
            for _ in range(20):
                await asyncio.sleep(0)
                if len(played) == 2:
                    break
            self.assertEqual(played, ['旧汇报', '发了。'])
            self.assertEqual(list(self.bridge.reports), [])
            self.assertFalse(task.done())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
