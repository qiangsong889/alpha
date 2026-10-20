import asyncio
import json
import unittest
from unittest.mock import patch

from core.costs import CostLedger, record_cost

from voice.asr_client import AsrClient, PREFIX
from voice.segmenter import SegmentReady


def segment(ident, size=3200):
    return SegmentReady(ident, b'\1\0' * (size // 2), 0, .1, .1, 'silence')


class Socket:
    def __init__(self, label, trace, behavior='ok', gate=None):
        self.label, self.trace, self.behavior, self.gate = label, trace, behavior, gate
        self.events = asyncio.Queue()
        self.events.put_nowait({'type': 'session.created'})
        self.committed = asyncio.Event()
        self.closed = False
        self.tasks = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.events.get()
        self.trace.append((self.label, 'received', event['type']))
        return json.dumps(event)

    async def send(self, raw):
        event = json.loads(raw)
        self.trace.append((self.label, 'sent', event['type']))
        if event['type'] == 'session.update':
            await self.events.put({'type': 'session.updated', 'session': {}})  # null可省略。
        if event['type'] != 'input_audio_buffer.commit':
            return
        self.committed.set()
        if self.behavior == 'hang':
            return
        if self.behavior == 'error':
            await self.events.put({'type': 'error', 'error': {'message': '故意模拟服务错误，没有code'}})
            return
        await self.events.put({'type': 'input_audio_buffer.committed'})
        async def complete():
            if self.gate is not None:
                await self.gate.wait()
            event = {'type': PREFIX + 'completed', 'transcript': f'文字-{self.label}'}
            if self.behavior == 'cumulative':
                self.completion_count = getattr(self, 'completion_count', 0) + 1
                event['usage'] = {'duration': 39 + 2 * self.completion_count,
                                  'output_tokens_details': {'audio_tokens': 726 + 8 * self.completion_count}}
            await self.events.put(event)
        self.tasks.append(asyncio.create_task(complete()))

    async def close(self):
        self.closed = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


class AsrClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_cumulative_usage_ledger_and_reset_on_new_connection(self):
        client, sockets, results, trace = await self.make_client([('cumulative', None), ('cumulative', None)])
        ledger = CostLedger(':memory:')
        self.addCleanup(ledger.close)
        async def notify(kind, seg_id, usage):
            if kind == 'usage':
                record_cost('ASR', 'qwen3-asr-flash-realtime', usage, event_id=seg_id)
        client.notify = notify
        with patch('core.costs._ledger', ledger):
            await client.submit(segment('one'))
            await client.submit(segment('two'))
            await asyncio.wait_for(client.queue.join(), 1)
            self.assertEqual(len(sockets), 1)
            await client.close()
            await client.submit(segment('new-connection'))
            await asyncio.wait_for(client.queue.join(), 1)
        usages = [json.loads(row[0]) for row in ledger.db.execute('SELECT usage FROM cost_events ORDER BY rowid')]
        self.assertEqual([u['duration'] for u in usages], [41, 2, 41])
        self.assertEqual([u['output_tokens_details']['audio_tokens'] for u in usages], [734, 8, 734])
        self.assertEqual(ledger.summary()['ASR'], {'usd': '0.00756', 'unpriced': 0, 'events': 3})
        self.assertEqual(len(sockets), 2)

    async def make_client(self, behaviors, timeout=.3):
        trace, sockets, results = [], [], []
        async def emit(result):
            results.append(result)
        async def connector(*args, **kwargs):
            behavior, gate = behaviors[len(sockets)]
            socket = Socket(len(sockets) + 1, trace, behavior, gate)
            sockets.append(socket)
            return socket
        client = AsrClient('wss://fake', 'fake-key', emit, timeout_s=timeout, connector=connector)
        worker = asyncio.create_task(client.run())
        async def cleanup():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        self.addAsyncCleanup(cleanup)
        return client, sockets, results, trace

    async def test_second_append_follows_first_completed_and_connection_is_reused(self):
        gate = asyncio.Event()
        client, sockets, results, trace = await self.make_client([('ok', gate)])
        await client.submit(segment('first'))
        await client.submit(segment('second'))
        while not sockets:
            await asyncio.sleep(0)
        await sockets[0].committed.wait()
        self.assertEqual(sum(kind == 'input_audio_buffer.append' for _, _, kind in trace), 1)
        gate.set()
        await asyncio.wait_for(client.queue.join(), 1)
        appends = [i for i, (_, action, kind) in enumerate(trace) if action == 'sent' and kind == 'input_audio_buffer.append']
        first_completed = trace.index((1, 'received', PREFIX + 'completed'))
        self.assertGreater(appends[1], first_completed)
        self.assertEqual(len(sockets), 1)
        self.assertEqual([(r.kind, r.seg_id) for r in results], [('transcript', 'first'), ('transcript', 'second')])

    async def test_timeout_fails_once_then_reconnects_and_old_results_cannot_leak(self):
        client, sockets, results, trace = await self.make_client([('hang', None), ('ok', None)], timeout=.03)
        await client.submit(segment('timeout'))
        await client.submit(segment('next'))
        await asyncio.wait_for(client.queue.join(), 1)
        await sockets[0].events.put({'type': PREFIX + 'completed', 'transcript': '旧连接的晚到文字'})
        self.assertEqual(len(sockets), 2)
        self.assertTrue(sockets[0].closed)
        self.assertEqual([(r.kind, r.seg_id) for r in results], [('failed', 'timeout'), ('transcript', 'next')])
        self.assertEqual(results[1].text, '文字-2')
        self.assertIn('超时', results[0].text)

    async def test_server_error_without_code_fails_then_next_segment_gets_new_connection(self):
        client, sockets, results, _ = await self.make_client([('error', None), ('ok', None)])
        await client.submit(segment('error'))
        await client.submit(segment('next'))
        await asyncio.wait_for(client.queue.join(), 1)
        self.assertEqual([r.kind for r in results], ['failed', 'transcript'])
        self.assertIn('没有code', results[0].text)
        self.assertTrue(sockets[0].closed)

    async def test_too_short_never_connects_or_commits(self):
        client, sockets, results, _ = await self.make_client([])
        await client.submit(segment('noise', 1024))
        await asyncio.wait_for(client.queue.join(), 1)
        self.assertEqual(sockets, [])
        self.assertEqual(results[0].kind, 'failed')

    async def test_ready_preconnect_and_commit_notification_precedes_result(self):
        client, sockets, results, trace = await self.make_client([('ok', None)])
        notices = []
        async def notify(kind, seg_id, payload):
            notices.append((kind, seg_id, payload))
        client.notify = notify
        await client.ready()
        self.assertEqual(len(sockets), 1)
        self.assertEqual(results, [])
        await client.submit(segment('local'))
        await asyncio.wait_for(client.queue.join(), 1)
        self.assertEqual([n[:2] for n in notices], [('uploading', 'local'), ('committed', 'local'), ('usage', 'local')])
        self.assertEqual(results[0].seg_id, 'local')
        self.assertEqual(len(sockets), 1)

    async def test_empty_completed_still_reports_usage_before_failed(self):
        client, sockets, results, trace = await self.make_client([('hang', None)])
        notices = []
        async def notify(kind, seg_id, payload):
            notices.append((kind, payload))
        client.notify = notify
        await client.submit(segment('empty'))
        while not sockets:
            await asyncio.sleep(0)
        await sockets[0].committed.wait()
        await sockets[0].events.put({'type': PREFIX + 'completed', 'transcript': '', 'usage': {'duration': 2}})
        await asyncio.wait_for(client.queue.join(), 1)
        self.assertEqual(notices, [('uploading', None), ('committed', None), ('usage', {'duration': 2})])
        self.assertEqual(results[0].kind, 'failed')
        self.assertTrue(sockets[0].closed)
