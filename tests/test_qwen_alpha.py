"""混合语音入口不打开音频设备、不访问真实工具。"""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

import httpx2
from voice.qwen_alpha import BrainBridge, Receiver, TurnCollector, session_config
from voice.omni_transport import CostTracker
from voice.qwen_alpha import Report
from voice.batch_timing import BatchTiming


class ConfigTests(TestCase):
    def test_no_tools_and_no_automatic_responses(self):
        config = session_config('Tina')
        self.assertEqual(config['tools'], [])
        self.assertFalse(config['turn_detection']['create_response'])
        self.assertFalse(config['turn_detection']['interrupt_response'])


class CollectionTests(IsolatedAsyncioTestCase):
    async def test_semantic_check_starts_during_window_and_does_not_shorten_it(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def classify(*args):
            started.set()
            await release.wait()
            return True
        self.router.turn_complete.side_effect = classify
        self.gate.window_s = .15
        self.gate.transcript('1', '关客厅灯')
        await asyncio.wait_for(started.wait(), .08)
        release.set()
        await asyncio.sleep(.03)
        self.submit.assert_not_called()
        await asyncio.sleep(.16)
        self.submit.assert_called_once_with('关客厅灯')

    async def test_speech_cancels_inflight_jev_request(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def classify(*args):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        self.router.turn_complete.side_effect = classify
        self.gate.transcript('1', '加牛奶')
        await started.wait()
        self.gate.started('2')
        await asyncio.wait_for(cancelled.wait(), 1)
        self.submit.assert_not_called()

    async def test_submission_snapshots_stop_time_and_later_batch_cannot_change_it(self):
        with patch('voice.qwen_alpha.time.perf_counter', return_value=100):
            self.gate.started('1')
            self.gate.stopped('1')
        self.gate.transcript('1', '关灯，执行吧')
        await self.settle()
        first = self.gate.last_submission
        self.assertEqual(first.stopped_at, 100)
        with patch('voice.qwen_alpha.time.perf_counter', return_value=200):
            self.gate.started('2')
            self.gate.stopped('2')
        self.assertEqual(first.stopped_at, 100)

    async def test_missing_stop_does_not_borrow_an_unrelated_noise_timestamp(self):
        self.gate.stopped('noise')
        self.gate.transcript('noise', '咳嗽')
        self.gate.transcript('actual', '关灯，执行吧')
        await self.settle()
        self.assertIsNone(self.gate.last_submission.stopped_at)
    async def asyncSetUp(self):
        self.submit, self.prompt = Mock(), Mock()
        self.router = NS(turn_complete=AsyncMock(return_value=True))
        self.gate = TurnCollector(self.submit, self.prompt, self.router, window_s=.01, continuation_s=.01, mode='window')
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
    async def test_batch_id_and_timing_follow_http_and_reports_without_cross_talk(self):
        timing = BatchTiming(stopped_at=100, submitted_at=101, id='batch-one')
        def handle(request):
            body = json.loads(request.content)
            self.assertEqual(body['request_id'], timing.id)
            events = [
                {'type': 'tool_timing', 'request_id': 'another-batch', 'name': 'wrong', 'phase': 'started', 'elapsed_s': 0},
                {'type': 'tool_timing', 'request_id': timing.id, 'name': 'list_items', 'phase': 'started', 'elapsed_s': .2},
                {'type': 'tool_timing', 'request_id': timing.id, 'name': 'list_items', 'phase': 'finished', 'elapsed_s': .3, 'duration_s': .1},
                {'type': 'delta', 'text': '牛奶'}, {'type': 'done'}]
            return httpx2.Response(200, text='\n'.join(json.dumps(e) for e in events))
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            bridge = BrainBridge(client, 'http://brain')
            worker = asyncio.create_task(bridge.run())
            try:
                bridge.submit('查清单', timing)
                await asyncio.wait_for(bridge.requests.join(), 1)
                self.assertEqual([e['name'] for e in timing.tool_events], ['list_items', 'list_items'])
                self.assertIs(bridge.reports[0].timing, timing)
                self.assertEqual(timing.stopped_at, 100)
                self.assertIsNotNone(timing.http_started_at)
                self.assertIsNotNone(timing.brain_done_at)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

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
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['text'], '任务')
        self.assertRegex(calls[0]['request_id'], r'^[a-f0-9]{12}$')
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
                         wait_played=AsyncMock(return_value=True), wait_started=AsyncMock(return_value=115))
        self.gate = TurnCollector(Mock(), Mock(), NS(turn_complete=AsyncMock(return_value=True)), window_s=10, mode='window')
        self.addAsyncCleanup(self.gate.close)
        self.bridge = BrainBridge(None, 'http://brain')
        self.receiver = Receiver(self.ws, self.player, self.gate, self.bridge, CostTracker('singapore'), 'test')
        self.addAsyncCleanup(self.receiver.close)

    async def test_first_playback_uses_original_report_batch_not_latest_speech(self):
        self.receiver.active = True
        self.receiver.current_report = Report('旧汇报', 0, BatchTiming(stopped_at=100, id='first'))
        with self.assertLogs('alpha.qwen_alpha', 'INFO') as logs:
            await self.receiver.handle({'type': 'response.audio.delta', 'response_id': 'r',
                                        'item_id': 'audio', 'delta': 'AAA='})
            self.receiver.current_report = Report('新回答', 1, BatchTiming(stopped_at=200, id='second'))
            await asyncio.gather(*list(self.receiver.playback_tasks))
        playback = [line for line in logs.output if '声卡估计' in line]
        self.assertEqual(len(playback), 1)
        self.assertIn('first', playback[0])
        self.assertIn('15.00s', playback[0])
        self.assertNotIn('second', playback[0])

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


class JevSubmissionTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.submitted = asyncio.Event()
        self.submit = Mock(side_effect=lambda text: self.submitted.set())
        self.prompt = Mock()
        self.router = NS(turn_complete=AsyncMock(return_value=True))
        self.gate = TurnCollector(self.submit, self.prompt, self.router, window_s=10, continuation_s=.08)
        self.addAsyncCleanup(self.gate.close)

    async def test_complete_submits_without_fixed_window(self):
        self.gate.transcript('1', '关客厅灯')
        await asyncio.wait_for(self.submitted.wait(), .2)
        self.submit.assert_called_once_with('关客厅灯')
        self.prompt.assert_not_called()

    async def test_incomplete_waits_then_asks_once_without_execution(self):
        self.router.turn_complete.return_value = False
        self.gate.transcript('1', '加牛奶')
        await asyncio.sleep(.02)
        self.prompt.assert_not_called()
        await asyncio.sleep(.09)
        self.prompt.assert_called_once()
        self.submit.assert_not_called()
        await asyncio.sleep(.09)
        self.prompt.assert_called_once()

    async def test_continuation_cancels_incomplete_followup_and_submits_merged_text(self):
        self.router.turn_complete.return_value = False
        self.gate.transcript('1', '加牛奶')
        await asyncio.sleep(.02)
        self.gate.started('2')
        self.router.turn_complete.return_value = True
        self.gate.stopped('2')
        self.gate.transcript('2', '再加鸡蛋')
        await asyncio.wait_for(self.submitted.wait(), .2)
        self.submit.assert_called_once_with('加牛奶；再加鸡蛋')
        await asyncio.sleep(.09)
        self.prompt.assert_not_called()

    async def test_uncertain_and_exception_use_window_fallback(self):
        self.gate.window_s = .08
        for index, outcome in enumerate((None, TimeoutError('classification failed'))):
            with self.subTest(outcome=type(outcome).__name__):
                self.submitted.clear()
                self.submit.reset_mock()
                self.router.turn_complete.side_effect = outcome if isinstance(outcome, Exception) else None
                self.router.turn_complete.return_value = None
                self.gate.transcript(str(index), '关客厅灯')
                await asyncio.sleep(.02)
                self.submit.assert_not_called()
                await asyncio.wait_for(self.submitted.wait(), .2)
                self.submit.assert_called_once_with('关客厅灯')

    async def test_obviously_unfinished_phrase_cannot_be_forced_by_complete_classification(self):
        self.gate.transcript('1', '把牛奶')
        await asyncio.sleep(.11)
        self.submit.assert_not_called()
        self.prompt.assert_called_once()

    async def test_new_speech_cancels_slow_jev(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def classify(*args):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        self.router.turn_complete.side_effect = classify
        self.gate.transcript('1', '关灯')
        await started.wait()
        self.gate.started('2')
        await asyncio.wait_for(cancelled.wait(), .2)
        self.submit.assert_not_called()

    async def test_execute_command_skips_jev(self):
        self.router.turn_complete.side_effect = AssertionError('执行口令不能请求 Jev')
        self.gate.transcript('1', '关灯，执行吧')
        await asyncio.wait_for(self.submitted.wait(), .2)
        self.router.turn_complete.assert_not_called()

    async def test_outstanding_transcript_prevents_early_submission(self):
        self.gate.started('1')
        self.gate.stopped('1')
        self.gate.committed('2')
        self.gate.transcript('1', '加牛奶')
        await asyncio.sleep(.02)
        self.router.turn_complete.assert_not_called()
        self.gate.transcript('2', '牛奶换成酱油')
        await asyncio.wait_for(self.submitted.wait(), .2)
        self.submit.assert_called_once_with('加牛奶；牛奶换成酱油')


class InterruptionGateTests(IsolatedAsyncioTestCase):
    # 复用假设备夹具，不继承原测试方法。
    async def asyncSetUp(self):
        await ReceiverTests.asyncSetUp(self)
        self.gate.mode = 'jev'
        self.gate.window_s = .01
        self.gate.router.interruption_and_turn = AsyncMock(return_value=(True, True))
        self.report = Report('购物清单里有牛奶。', 0)
        self.receiver.current_report = self.report
        self.receiver.active = True
        self.receiver.response_id = 'old'

    async def utterance(self, text, item='speech'):
        await self.receiver.handle({'type': 'input_audio_buffer.speech_started', 'item_id': item})
        await self.receiver.handle({'type': 'input_audio_buffer.speech_stopped', 'item_id': item})
        await self.receiver.handle({'type': 'conversation.item.input_audio_transcription.completed',
                                    'item_id': item, 'transcript': text})

    async def settle_inputs(self):
        await asyncio.gather(*list(self.receiver.input_tasks))
        await asyncio.sleep(.03)

    async def test_side_chat_keeps_report_queue_and_resumes_without_history_or_cancel(self):
        self.gate.router.interruption_and_turn.return_value = (False, True)
        self.bridge.say('后续汇报')
        await self.utterance('小王，晚饭吃什么？')
        await self.settle_inputs()
        self.player.clear.assert_not_called()
        self.player.resume.assert_called_once()
        self.assertEqual(self.bridge.epoch, 0)
        self.assertEqual(len(self.bridge.reports), 1)
        self.assertEqual(self.gate.parts, [])
        self.assertEqual(self.gate.history, [])
        self.gate.submit.assert_not_called()
        self.ws.send.assert_not_awaited()

    async def test_addressed_speech_interrupts_and_reuses_completeness_result(self):
        await self.utterance('再加鸡蛋')
        await self.settle_inputs()
        self.player.clear.assert_called_once()
        self.gate.submit.assert_called_once_with('再加鸡蛋')
        self.gate.router.interruption_and_turn.assert_awaited_once()
        self.gate.router.turn_complete.assert_not_awaited()
        self.assertEqual(json.loads(self.ws.send.call_args.args[0])['type'], 'response.cancel')

    async def test_legacy_uncertain_accept_mode_can_be_selected(self):
        self.receiver.uncertain_interruption = 'accept'
        self.gate.router.interruption_and_turn.return_value = (None, None)
        await self.utterance('那个改一下')
        await self.settle_inputs()
        self.gate.submit.assert_called_once_with('那个改一下')
        self.player.clear.assert_called_once()
        self.gate.router.turn_complete.assert_not_awaited()

    async def test_classification_error_preserves_report_in_hold_mode(self):
        self.gate.router.interruption_and_turn.side_effect = TimeoutError()
        await self.utterance('再加鸡蛋')
        await self.settle_inputs()
        self.gate.submit.assert_not_called()
        self.player.clear.assert_not_called()
        self.assertEqual(self.bridge.epoch, 0)
        self.assertFalse(self.gate.pending)

    async def test_uncertain_casual_statement_does_not_interrupt_or_reach_brain(self):
        self.gate.router.interruption_and_turn.return_value = (None, True)
        self.bridge.say('待播的汇报')
        with self.assertLogs('alpha.qwen_alpha', 'INFO') as logs:
            await self.utterance('我吃过了。')
            await self.settle_inputs()
        self.gate.submit.assert_not_called()
        self.player.clear.assert_not_called()
        self.player.resume.assert_called_once()
        self.ws.send.assert_not_awaited()
        self.assertEqual(self.bridge.epoch, 0)
        self.assertEqual(len(self.bridge.reports), 1)
        self.assertIn('暂不打断', ' '.join(logs.output))

    async def test_idle_voice_does_not_pause_an_empty_player(self):
        self.receiver.active = False
        self.receiver.current_report = None
        self.receiver.activity = NS(voiced_ms=500)
        self.gate.speaking = True
        task = asyncio.create_task(self.receiver.confirm_speech())
        try:
            await asyncio.sleep(.03)
            self.player.pause.assert_not_called()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_jev_wait_does_not_block_websocket_events(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def classify(*args, **kwargs):
            started.set()
            await release.wait()
            return False, None
        self.gate.router.interruption_and_turn.side_effect = classify
        await asyncio.wait_for(self.utterance('小王晚上吃什么'), .2)
        await started.wait()
        await self.receiver.handle({'type': 'response.audio.delta', 'response_id': 'old',
                                    'item_id': 'old-audio', 'delta': 'AAA='})
        self.player.append.assert_called_once()
        self.player.clear.assert_not_called()
        release.set()
        await self.settle_inputs()

    async def test_overlap_is_remembered_after_original_report_finishes(self):
        self.gate.router.interruption_and_turn.return_value = (False, True)
        await self.receiver.handle({'type': 'input_audio_buffer.speech_started', 'item_id': 'late'})
        self.receiver.current_report = None
        self.receiver.active = False
        await self.receiver.handle({'type': 'input_audio_buffer.speech_stopped', 'item_id': 'late'})
        await self.receiver.handle({'type': 'conversation.item.input_audio_transcription.completed',
                                    'item_id': 'late', 'transcript': '小王你说呢'})
        await self.settle_inputs()
        self.gate.router.interruption_and_turn.assert_awaited_once()
        self.gate.submit.assert_not_called()

    async def test_late_decision_cannot_cancel_a_different_report(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def classify(*args, **kwargs):
            started.set()
            await release.wait()
            return True, True
        self.gate.router.interruption_and_turn.side_effect = classify
        await self.utterance('再加鸡蛋')
        await started.wait()
        self.receiver.current_report = Report('另一条报告', 1)
        self.receiver.response_id = 'new'
        release.set()
        await self.settle_inputs()
        self.ws.send.assert_not_awaited()
        self.player.clear.assert_not_called()

    async def test_wake_name_bypasses_address_filter(self):
        await self.utterance('Alpha，关客厅灯')
        await self.settle_inputs()
        self.gate.router.interruption_and_turn.assert_not_awaited()
        self.player.clear.assert_called_once()
        self.gate.submit.assert_called_once_with('Alpha，关客厅灯')

    async def test_wake_name_does_not_require_a_comma_before_chinese(self):
        speech = self.receiver.speech_input()
        self.assertEqual(self.receiver.shortcut('Alpha停一下', speech), '呼叫 Alpha')
        self.assertIsNone(self.receiver.shortcut('alphabet 是什么', speech))

    async def test_idle_input_skips_addressee_classification(self):
        self.receiver.current_report = None
        self.receiver.active = False
        await self.utterance('加牛奶')
        await self.settle_inputs()
        self.gate.router.interruption_and_turn.assert_not_awaited()
        self.gate.submit.assert_called_once_with('加牛奶')

    async def test_filter_can_be_disabled_without_changing_completeness_mode(self):
        self.receiver.interruption_filter = False
        await self.utterance('再加鸡蛋')
        await self.settle_inputs()
        self.gate.router.interruption_and_turn.assert_not_awaited()
        self.gate.submit.assert_called_once_with('再加鸡蛋')

    async def test_duplicate_transcript_while_classifying_does_not_call_twice(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def classify(*args, **kwargs):
            started.set()
            await release.wait()
            return True, True
        self.gate.router.interruption_and_turn.side_effect = classify
        await self.utterance('再加鸡蛋')
        await started.wait()
        await self.receiver.handle({'type': 'conversation.item.input_audio_transcription.completed',
                                    'item_id': 'speech', 'transcript': '再加鸡蛋'})
        release.set()
        await self.settle_inputs()
        self.gate.router.interruption_and_turn.assert_awaited_once()
        self.gate.submit.assert_called_once_with('再加鸡蛋')

    async def test_close_cancels_pending_classification(self):
        started = asyncio.Event()
        async def classify(*args, **kwargs):
            started.set()
            await asyncio.Future()
        self.gate.router.interruption_and_turn.side_effect = classify
        await self.utterance('再加鸡蛋')
        await started.wait()
        tasks = list(self.receiver.input_tasks)
        await self.receiver.close()
        self.assertTrue(all(task.cancelled() for task in tasks))

    async def test_answer_after_played_question_bypasses_both_classifiers(self):
        import time
        self.receiver.current_report = None
        self.receiver.active = False
        self.receiver.question_finished_at = time.perf_counter() - 1
        self.receiver.question_deadline = time.perf_counter() + 9
        await self.utterance('对。')
        await self.settle_inputs()
        self.gate.router.interruption_and_turn.assert_not_awaited()
        self.gate.router.turn_complete.assert_not_awaited()
        self.gate.submit.assert_called_once_with('对。')

    async def test_question_window_opens_only_after_successful_playback(self):
        for played in (False, True):
            self.receiver.question_finished_at = self.receiver.question_deadline = 0
            self.player.wait_played.return_value = played
            count = 0
            async def send(raw):
                nonlocal count
                count += 1
                if count == 3:
                    self.receiver.audio_items = {'answer'}
                    self.receiver.response_done.set()
            self.ws.send.side_effect = send
            await self.receiver._play_report('要创建文档吗？')
            self.assertEqual(self.receiver.question_deadline > 0, played)
