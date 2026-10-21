"""语音层拿到大脑的 turn_id，并把每一轮回答的实际播放结果报告回去；不连接真实服务。"""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import httpx2

from voice.qwen_alpha import BatchTiming, BrainBridge, Receiver, Report, TurnCollector
from voice.qwen_realtime import CostTracker

TURN = 'a' * 32


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

    async def test_cancelled_playback_is_interrupted(self):
        async def play(text):
            self.receiver.cancelled = True
            return False
        self.bridge.say('好的', turn_id=TURN)
        await self.run_one(play)
        self.assertEqual(self.seen, [(TURN, 'interrupted')])

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
