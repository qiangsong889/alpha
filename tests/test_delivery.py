"""语音层拿到大脑的 turn_id，并把每一轮回答的实际播放结果报告回去；不连接真实服务。"""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock

import httpx2

from voice.qwen_alpha import BatchTiming, BrainBridge, Receiver, Report, TurnCollector, tts_sentences
from voice.omni_transport import CostTracker

TURN = 'a' * 32


class SentenceTests(TestCase):
    def test_unreadable_parts_merge_without_changing_original_text(self):
        cases = [
            ('好的。 ', ['好的。 ']),
            ('Hi...', ['Hi...']),
            ('。 \n你好。再见！ ', ['。 \n你好。', '再见！ ']),
            ('温度26.5度。OK!', ['温度26.5度。', 'OK!']),
            ('好吗？\n  ', ['好吗？\n  ']),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                actual = tts_sentences(text)
                self.assertEqual(actual, expected)
                self.assertEqual(''.join(actual), text)
                self.assertTrue(all(any(c.isalnum() for c in part) for part in actual))

    def test_entirely_unreadable_text_has_no_sentences(self):
        for text in ['', ' ', '... ', '。！？\n ', '🙂']:
            with self.subTest(text=text):
                self.assertEqual(tts_sentences(text), [])


class BridgeTurnTests(IsolatedAsyncioTestCase):
    async def test_turn_event_is_attached_to_final_report_but_not_from_another_batch(self):
        timing = BatchTiming(id='batch-one')

        def handle(request):
            events = [{'type': 'turn', 'request_id': 'other-batch', 'turn_id': 'b' * 32},
                      {'type': 'turn', 'request_id': timing.id, 'turn_id': TURN},
                      {'type': 'delta', 'text': '我查一下。'},
                      {'type': 'delta', 'text': '查到了'}, {'type': 'done'}]
            return httpx2.Response(200, text='\n'.join(json.dumps(e) for e in events))

        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            bridge = BrainBridge(client, 'http://brain')
            worker = asyncio.create_task(bridge.run())
            try:
                bridge.submit('搜一下', timing)
                await asyncio.wait_for(bridge.requests.join(), 1)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
        self.assertEqual([(r.kind, r.turn_id) for r in bridge.reports],
                         [('search_progress', TURN), ('reply', TURN)])

    async def test_delivery_is_posted_once_and_progress_is_not_reported(self):
        posted = []

        def handle(request):
            posted.append((request.url.path, json.loads(request.content)))
            return httpx2.Response(200, json={'ok': True})

        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            bridge = BrainBridge(client, 'http://brain')
            bridge.report_delivery(Report('我查一下。', 0, kind='search_progress', turn_id=TURN), 'played')
            bridge.report_delivery(Report('好', 0, turn_id=TURN), 'played')
            bridge.report_delivery(Report('好', 0, turn_id=TURN), 'interrupted')  # 同一轮不重复报告
            bridge.report_delivery(Report('好', 0), 'played')  # 没有 turn_id 的报告无法关联
            await asyncio.gather(*bridge.delivery_tasks)
        self.assertEqual(posted, [('/delivered', {'turn_id': TURN, 'delivery': 'played'})])

    async def test_next_request_waits_for_interruption_notification(self):
        started, release = asyncio.Event(), asyncio.Event()
        order = []
        async def handle(request):
            if request.url.path == '/interrupted':
                started.set()
                await release.wait()
                order.append('interrupted')
                return httpx2.Response(200, json={'ok': True})
            order.append('ask')
            return httpx2.Response(200, text=json.dumps({'type': 'done'}))
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            bridge = BrainBridge(client, 'http://brain')
            bridge.report_interruption(Report('第一句。第二句。', 0, turn_id=TURN), '第一句。', '第二句。')
            await started.wait()
            worker = asyncio.create_task(bridge.run())
            try:
                bridge.submit('继续')
                await asyncio.sleep(.01)
                self.assertEqual(order, [])
                release.set()
                await asyncio.wait_for(bridge.requests.join(), 1)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
        self.assertEqual(order, ['interrupted', 'ask'])

    async def test_failed_notification_is_carried_in_next_request(self):
        bodies = []
        def handle(request):
            if request.url.path == '/interrupted':
                return httpx2.Response(500)
            bodies.append(json.loads(request.content))
            return httpx2.Response(200, text=json.dumps({'type': 'done'}))
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            bridge = BrainBridge(client, 'http://brain')
            bridge.report_interruption(Report('第一句。第二句。', 0, turn_id=TURN), '第一句。', '第二句。')
            worker = asyncio.create_task(bridge.run())
            try:
                bridge.submit('继续')
                await asyncio.wait_for(bridge.requests.join(), 1)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
        self.assertEqual(bodies[0]['interruptions'], [{'turn_id': TURN,
            'spoken': '第一句。', 'uncertain_text': '第二句。'}])
        self.assertFalse(bridge.interruptions)

    async def test_delivery_failure_is_logged_not_raised(self):
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda r: httpx2.Response(500))) as client:
            bridge = BrainBridge(client, 'http://brain')
            with self.assertLogs('alpha.qwen_alpha', 'WARNING') as logs:
                bridge.report_delivery(Report('好', 0, turn_id=TURN), 'played')
                await asyncio.gather(*bridge.delivery_tasks)
        self.assertIn('报告失败', ' '.join(logs.output))

    async def test_replaced_reports_are_reported_not_played(self):
        bridge = BrainBridge(None, 'http://brain')
        seen = []
        bridge.report_delivery = lambda report, delivery: seen.append((report.turn_id, delivery))
        bridge.say('旧回答', turn_id=TURN)
        bridge.new_input()                          # 还没播的旧回答被新输入作废
        bridge.say('更旧的', epoch=0, turn_id='c' * 32)  # 迟到的旧批次结果直接作废
        self.assertEqual(seen, [(TURN, 'not_played'), ('c' * 32, 'not_played')])

    async def test_unplayed_reports_notify_brain_with_empty_heard_prefix(self):
        posted = []
        def handle(request):
            posted.append((request.url.path, json.loads(request.content)))
            return httpx2.Response(200, json={'ok': True})

        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            bridge = BrainBridge(client, 'http://brain')
            bridge.say('排队但没播的回答', turn_id=TURN, session_id='d' * 32)
            bridge.new_input()
            bridge.say('晚到且没播的回答', epoch=0, turn_id='c' * 32, session_id='d' * 32)
            await asyncio.gather(*bridge.delivery_tasks, *bridge.interruption_tasks)

        notifications = [body for path, body in posted if path == '/interrupted']
        self.assertCountEqual(notifications, [
            {'turn_id': turn, 'session_id': 'd' * 32, 'spoken': '',
             'uncertain_text': '', 'delivery': 'not_played'}
            for turn in (TURN, 'c' * 32)
        ])
        self.assertCountEqual([body for path, body in posted if path == '/delivered'], [
            {'turn_id': turn, 'delivery': 'not_played'} for turn in (TURN, 'c' * 32)
        ])


class ReceiverDeliveryTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        player = NS(clear=Mock(), append=Mock(), finish=Mock(), gain=1, resume=Mock(), pause=Mock(),
                    wait_played=AsyncMock(return_value=True), wait_started=AsyncMock(return_value=1))
        gate = TurnCollector(Mock(), Mock(), NS(turn_complete=AsyncMock(return_value=True)), window_s=10, mode='window')
        self.addAsyncCleanup(gate.close)
        self.bridge = BrainBridge(None, 'http://brain')
        self.seen = []
        self.bridge.report_delivery = lambda report, delivery: self.seen.append((report.turn_id, delivery))
        self.receiver = Receiver(NS(send=AsyncMock()), player, gate, self.bridge, CostTracker('singapore'), 'test')
        self.addAsyncCleanup(self.receiver.close)

    async def run_one(self, play):
        self.receiver._play_report = play
        task = asyncio.create_task(self.receiver.speak())
        try:
            for _ in range(50):
                await asyncio.sleep(0)
                if self.seen:
                    break
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_completed_playback_is_played(self):
        self.bridge.say('好的', turn_id=TURN)
        await self.run_one(AsyncMock(return_value=True))
        self.assertEqual(self.seen, [(TURN, 'played')])
        self.assertEqual(self.receiver.finished_report[0], '好的')  # 用于识别紧接着的回声尾巴

    async def test_cancelled_playback_is_interrupted(self):
        async def play(text):
            self.receiver.cancelled = True
            return False
        self.bridge.say('好的', turn_id=TURN)
        await self.run_one(play)
        self.assertEqual(self.seen, [(TURN, 'interrupted')])
        self.assertIsNone(self.receiver.finished_report)  # 被打断时你正在说话，不能把你的话当成回声

    async def test_omni_interruption_never_claims_the_full_answer_was_heard(self):
        report = Report('第一句。第二句。', 0, turn_id=TURN)
        self.receiver.current_report = report
        self.receiver.active = True
        self.receiver.response_id = 'omni-response'
        self.receiver.audio_items = {'received-audio'}  # 收到音频不代表文字已有精确对齐。
        self.bridge.report_interruption = Mock()

        await self.receiver.interrupt()

        self.bridge.report_interruption.assert_called_once_with(report, '', report.text)
        self.receiver.player.clear.assert_called_once()
        self.assertTrue(self.receiver.cancelled)
        self.assertEqual(json.loads(self.receiver.ws.send.call_args.args[0])['type'], 'response.cancel')

    async def test_failed_playback_without_interruption_is_not_played(self):
        self.bridge.say('好的', turn_id=TURN)
        await self.run_one(AsyncMock(return_value=False))
        self.assertEqual(self.seen, [(TURN, 'not_played')])

    async def test_tts_cost_is_attributed_to_the_reports_turn(self):
        calls = []

        async def stream(text, turn_id=None):
            calls.append(turn_id)
            return
            yield b''

        self.receiver.tts = NS(stream=stream, voice='Ethan', close=AsyncMock())
        self.receiver.current_report = Report('好的', 0, turn_id=TURN)
        await self.receiver._play_tts('好的')
        self.assertEqual(calls, [TURN])

    async def test_unreadable_text_skips_tts_and_does_not_open_question_window(self):
        self.receiver.tts = NS(stream=Mock(), voice='Ethan', close=AsyncMock())
        self.receiver.question_deadline = 100
        self.assertFalse(await self.receiver._play_tts('？ ... \n'))
        self.receiver.tts.stream.assert_not_called()
        self.receiver.player.clear.assert_not_called()
        self.assertEqual(self.receiver.question_deadline, 0)

    async def test_question_with_trailing_whitespace_uses_one_tts_request_and_opens_window(self):
        calls = []
        async def stream(text, **kwargs):
            calls.append(text)
            yield b'\x00\x20' * 480
        self.receiver.tts = NS(stream=stream, voice='Ethan', close=AsyncMock())
        self.assertTrue(await self.receiver._play_tts('好吗？ '))
        self.assertEqual(calls, ['好吗？ '])
        self.assertGreater(self.receiver.question_deadline, self.receiver.question_finished_at)
        self.receiver.player.clear.assert_not_called()


class PlaybackProgressTests(IsolatedAsyncioTestCase):
    async def test_only_completed_sentence_is_reported_when_second_sentence_is_interrupted(self):
        import numpy as np
        from voice.audio import StreamingPlayer
        player = StreamingPlayer()
        player.ready = True
        player.latency = 0
        second = asyncio.Event()
        synthesized = []
        async def stream(text, **kwargs):
            synthesized.append(text)
            yield b'\x00\x20' * 480
            if len(synthesized) == 2:
                second.set()
                await asyncio.Future()
        tts = NS(stream=stream, voice='Ethan', close=AsyncMock())
        bridge = BrainBridge(None, 'http://brain')
        bridge.report_interruption = Mock()
        collector = TurnCollector(Mock(), Mock(), NS(turn_complete=AsyncMock()), window_s=10)
        receiver = Receiver(None, player, collector, bridge, CostTracker('singapore'), 'test', tts=tts)
        report = Report('第一句。第二句。', 0, turn_id=TURN)
        receiver.current_report = report
        task = asyncio.create_task(receiver._play_tts(report.text))
        try:
            await asyncio.wait_for(second.wait(), 1)
            player.callback(np.zeros((600, 1), dtype=np.float32), 600, None, None)
            self.assertTrue(player.playback_finished(receiver.tts_segments[0][0]))
            self.assertFalse(player.playback_finished(receiver.tts_segments[1][0]))
            await receiver.interrupt()
            self.assertFalse(await task)
            bridge.report_interruption.assert_called_once_with(report, '第一句。', '第二句。')
            self.assertEqual(synthesized, ['第一句。', '第二句。'])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await receiver.close()
            await collector.close()

    async def test_paused_or_buffered_audio_is_not_counted_as_heard(self):
        import numpy as np
        from voice.audio import StreamingPlayer
        player = StreamingPlayer()
        player.ready = True
        player.latency = 0
        player.append(b'\x00\x20' * 480, 'first', 0)
        player.finish('first')
        player.pause()
        player.callback(np.zeros((480, 1), dtype=np.float32), 480, None, None)
        self.assertFalse(player.playback_finished('first'))
        player.resume()
        player.callback(np.zeros((480, 1), dtype=np.float32), 480, None, None)
        self.assertTrue(player.playback_finished('first'))
