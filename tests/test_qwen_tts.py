import asyncio
import base64
import json
from decimal import Decimal
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from voice.qwen_tts import QwenTTS
from voice.qwen_alpha import Receiver
from core.costs import CostLedger


class TTSTests(IsolatedAsyncioTestCase):
    def service(self):
        return QwenTTS('fake-key', 'workspace', voice='Ethan')

    async def test_cumulative_usage_is_differenced_in_ledger_and_log_and_reset_on_reconnect(self):
        def socket(characters):
            events = [{'type': 'session.created'}, {'type': 'session.updated', 'session': {'voice': 'Ethan'}}]
            events += [{'type': 'response.done', 'response': {'status': 'completed', 'usage': {
                'characters': n, 'output_tokens_details': {'audio_tokens': n * 3}}}} for n in characters]
            return NS(send=AsyncMock(), recv=AsyncMock(side_effect=[json.dumps(e) for e in events]), close=AsyncMock())
        first, second = socket([4, 8]), socket([2])
        service = self.service()
        self.addAsyncCleanup(service.close)
        ledger = CostLedger(':memory:')
        self.addCleanup(ledger.close)
        with patch('core.costs._ledger', ledger), \
             patch('voice.qwen_tts.connect', AsyncMock(side_effect=[first, second])) as connect, \
             patch('voice.qwen_tts.logger.info') as log:
            for _ in range(2):
                self.assertEqual([part async for part in service.stream('你好')], [])
            self.assertEqual(connect.await_count, 1)
            await service.close()
            self.assertEqual([part async for part in service.stream('好')], [])
        usages = [json.loads(row[0]) for row in ledger.db.execute('SELECT usage FROM cost_events ORDER BY rowid')]
        self.assertEqual([u['characters'] for u in usages], [4, 4, 2])
        self.assertEqual([u['output_tokens_details']['audio_tokens'] for u in usages], [12, 12, 6])
        self.assertEqual([json.loads(c.args[1])['characters'] for c in log.call_args_list
                          if c.args[0] == '[TTS用量] %s'], [4, 4, 2])
        summary = ledger.summary()['TTS']
        self.assertEqual(Decimal(summary['usd']), Decimal('0.00013'))
        self.assertEqual((summary['unpriced'], summary['events']), (0, 3))

    async def test_exact_text_and_reuse(self):
        events = [
            {'type': 'session.created'},
            {'type': 'session.updated', 'session': {'voice': 'Ethan'}},
            {'type': 'response.audio.delta', 'delta': base64.b64encode(b'pcm').decode()},
            {'type': 'response.done'},
            {'type': 'response.done'},
        ]
        ws = NS(send=AsyncMock(), recv=AsyncMock(side_effect=[json.dumps(e) for e in events]), close=AsyncMock())
        service = self.service()
        with patch('voice.qwen_tts.connect', AsyncMock(return_value=ws)) as connect:
            self.assertEqual([part async for part in service.stream('没有发送邮件。')], [b'pcm'])
            self.assertEqual([part async for part in service.stream('第二句')], [])
            self.assertEqual(connect.call_count, 1)
        messages = [json.loads(call.args[0]) for call in ws.send.call_args_list]
        self.assertEqual(messages[0]['session']['voice'], 'Ethan')
        self.assertEqual(messages[1]['text'], '没有发送邮件。')
        self.assertFalse(any(m['type'] == 'response.create' for m in messages))
        await service.close()

    async def test_error_closes_connection_without_retry(self):
        service = self.service()
        ws = NS(send=AsyncMock(), recv=AsyncMock(return_value=json.dumps({'type': 'error', 'error': {'message': 'bad'}})), close=AsyncMock())
        service.ws = ws
        with self.assertRaisesRegex(RuntimeError, 'bad'):
            async for _ in service.stream('test'):
                pass
        ws.close.assert_awaited_once()
        self.assertIsNone(service.ws)

    async def test_cancel_closes_connection(self):
        service = self.service()
        started = asyncio.Event()
        async def receive():
            started.set()
            await asyncio.Future()
        service.ws = NS(send=AsyncMock(), recv=receive, close=AsyncMock())
        ws = service.ws
        async def consume():
            async for _ in service.stream('test'):
                pass
        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        ws.close.assert_awaited_once()

    async def test_tts_playback_uses_existing_player_and_no_omni_generation(self):
        async def stream(text):
            self.assertEqual(text, '没有发送邮件。')
            yield b'pcm'
        tts = NS(voice='Ethan', stream=stream, close=AsyncMock())
        player = NS(append=Mock(), finish=Mock(), wait_played=AsyncMock(return_value=True))
        ws = NS(send=AsyncMock())
        receiver = Receiver(ws, player, None, None, None, 'sess', tts=tts)
        receiver.watch_playback = Mock()
        await receiver._play_report('没有发送邮件。')
        player.append.assert_called_once()
        player.finish.assert_called_once()
        ws.send.assert_not_awaited()
        self.assertFalse(receiver.active)
        self.assertTrue(receiver.response_done.is_set())

    async def test_interruption_cancels_tts_not_microphone_socket(self):
        started = asyncio.Event()
        closed = asyncio.Event()
        async def stream(text):
            try:
                started.set()
                await asyncio.Future()
                yield b'never'
            finally:
                closed.set()
        player = NS(clear=Mock(), finish=Mock(), wait_played=AsyncMock(return_value=False))
        ws = NS(send=AsyncMock())
        receiver = Receiver(ws, player, None, None, None, 'sess',
                            tts=NS(voice='Ethan', stream=stream, close=AsyncMock()))
        task = asyncio.create_task(receiver._play_report('test'))
        await started.wait()
        await receiver.interrupt()
        await asyncio.wait_for(task, 1)
        self.assertTrue(closed.is_set())
        ws.send.assert_not_awaited()
        player.clear.assert_called_once()
