"""独立 ASR 探测协议测试；不使用网络、麦克风或真实凭据。"""

import asyncio
import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from devtools.asr_probe import CHUNK_BYTES, PREFIX, Probe, read_wav, websocket_url


class FakeSocket:
    def __init__(self, *, partial=True, omit_null=False, omit_item=False, early_partial=False):
        self.events = asyncio.Queue()
        self.events.put_nowait({'type': 'session.created', 'session': {'id': 'single-session'}})
        self.sent, self.pcm = [], bytearray()
        self.item, self.partial = 0, partial
        self.omit_null = omit_null
        self.omit_item, self.early_partial = omit_item, early_partial

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.events.get()
        if self.omit_item:
            event.pop('item_id', None)
            if 'item' in event:
                event['item'].pop('id', None)
        return json.dumps(event)

    async def send(self, message):
        event = json.loads(message)
        self.sent.append(event)
        kind = event['type']
        if kind == 'session.update':
            session = dict(event['session'])
            if self.omit_null:
                session.pop('turn_detection')
            await self.events.put({'type': 'session.updated', 'session': session})
        elif kind == 'input_audio_buffer.append':
            self.pcm.extend(base64.b64decode(event['audio']))
            if self.early_partial:
                await self.events.put({'type': PREFIX + 'text', 'text': '', 'stash': '提前返回'})
        elif kind == 'input_audio_buffer.commit':
            if not self.pcm:
                await self.events.put({'type': 'error', 'error': {
                    'code': 'fake_empty_buffer', 'message': 'fake error, not a real API result',
                    'event_id': event['event_id']}})
                return
            self.pcm.clear()
            self.item += 1
            item_id = f'item-{self.item}'
            await self.events.put({'type': 'input_audio_buffer.committed', 'item_id': item_id})
            await self.events.put({'type': 'conversation.item.created', 'item': {'id': item_id}})
            if self.partial:
                await self.events.put({'type': PREFIX + 'text', 'item_id': item_id,
                                       'text': '测试', 'stash': '文字'})
            await self.events.put({'type': PREFIX + 'completed', 'item_id': item_id, 'transcript': '测试文字'})
        elif kind == 'session.finish':
            await self.events.put({'type': 'session.finished'})


class AsrProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_back_to_back_timeout_still_reports_partial_results(self):
        class MissingResultSocket(FakeSocket):
            async def __anext__(self):
                while True:
                    message = await super().__anext__()
                    event = json.loads(message)
                    if event.get('item_id') == 'item-2' and event['type'] == PREFIX + 'completed':
                        continue
                    return message
        output = io.StringIO()
        probe = Probe(MissingResultSocket(), timeout_s=.02)
        with redirect_stdout(output), self.assertRaises(TimeoutError):
            await probe.run([b'\0' * CHUNK_BYTES] * 2, back_to_back=True)
        self.assertEqual(len(probe.completed), 1)
        self.assertIn('[completed数量] 1/2', output.getvalue())
        self.assertIn('[completed到达顺序 #1]', output.getvalue())
        self.assertNotIn('[连续提交结束]', output.getvalue())

    async def test_back_to_back_sends_second_commit_before_waiting_and_preserves_actual_order(self):
        class ConsecutiveSocket(FakeSocket):
            def __init__(self, reverse):
                super().__init__(omit_item=True)
                self.reverse, self.pending = reverse, []

            async def send(self, message):
                event = json.loads(message)
                if event['type'] != 'input_audio_buffer.commit' or not self.pcm:
                    return await super().send(message)
                self.sent.append(event)
                text = '太阳系长句' if self.pcm[0] == 1 else '苹果短句'
                self.pcm.clear()
                self.pending.append(text)
                await self.events.put({'type': 'input_audio_buffer.committed'})
                # 两次提交都发出才交回结果；若实现偷偷等第一段，本测试会超时。
                if len(self.pending) == 2:
                    for text in self.pending[::(-1 if self.reverse else 1)]:
                        await self.events.put({'type': PREFIX + 'completed', 'transcript': text})

        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                ws = ConsecutiveSocket(reverse)
                probe, output = Probe(ws, timeout_s=1), io.StringIO()
                with redirect_stdout(output), patch('devtools.asr_probe.asyncio.sleep',
                                                     new=unittest.mock.AsyncMock()) as sleep:
                    await probe.run([b'\x01\0' * 3200, b'\x02\0' * 1600], back_to_back=True)
                self.assertEqual([r['transcript'] for r in probe.completed],
                                 ['苹果短句', '太阳系长句'] if reverse else ['太阳系长句', '苹果短句'])
                self.assertTrue(all(r['commits_sent'] == 2 for r in probe.completed))
                self.assertIn('第一个completed：True', output.getvalue())
                sleep.assert_not_awaited()

    async def test_two_clips_one_connection_then_empty_commit_prints_all_events(self):
        ws = FakeSocket()
        probe = Probe(ws, timeout_s=1)
        first = b'\x01\x00' * 2000  # 125ms；最后一块短于100ms。
        second = b'\x02\x00' * 1600
        output = io.StringIO()
        with redirect_stdout(output), patch('devtools.asr_probe.asyncio.sleep', new=unittest.mock.AsyncMock()):
            await probe.run([first, second])
        events = ws.sent
        self.assertEqual(events[0]['session']['turn_detection'], None)
        self.assertEqual([e['type'] for e in events].count('session.update'), 1)
        self.assertEqual([e['type'] for e in events].count('input_audio_buffer.commit'), 3)
        self.assertEqual(events[-1]['type'], 'session.finish')
        chunks = [base64.b64decode(e['audio']) for e in events if e['type'] == 'input_audio_buffer.append']
        self.assertEqual(list(map(len, chunks)), [CHUNK_BYTES, 800, CHUNK_BYTES])
        self.assertEqual(b''.join(chunks), first + second)
        self.assertEqual(len({e['event_id'] for e in events}), len(events))
        for word in ('session.created', 'session.updated', 'conversation.item.created',
                     'text', 'stash', 'transcript', 'item-1', 'item-2',
                     '第1段 commit+', '第2段 commit+', '首个中间结果', 'fake_empty_buffer', 'session.finished'):
            self.assertIn(word, output.getvalue())
        self.assertEqual(output.getvalue().count('[接收结束]'), 0)
        self.assertEqual(set(probe.items['item-1'].timings), {
            'input_audio_buffer.committed', PREFIX + 'text', PREFIX + 'completed'})
        self.assertIsNot(probe.items['item-1'], probe.items['item-2'])

    async def test_final_without_intermediate_still_finishes(self):
        output = io.StringIO()
        with redirect_stdout(output):
            await Probe(FakeSocket(partial=False), timeout_s=1).run([b'\0' * CHUNK_BYTES] * 2)
        self.assertEqual(output.getvalue().count('首个中间结果：未收到'), 2)

    async def test_server_may_omit_null_turn_detection(self):
        output = io.StringIO()
        with redirect_stdout(output):
            await Probe(FakeSocket(omit_null=True), timeout_s=1).run([b'\0' * CHUNK_BYTES] * 2)
        self.assertIn('服务端省略', output.getvalue())
        self.assertIn('[复用成功]', output.getvalue())

    async def test_live_protocol_without_item_ids_and_partial_before_commit(self):
        output = io.StringIO()
        probe = Probe(FakeSocket(omit_null=True, omit_item=True, early_partial=True), timeout_s=1)
        with redirect_stdout(output):
            await probe.run([b'\0' * CHUNK_BYTES] * 2)
        self.assertIn('committed 未提供 item_id', output.getvalue())
        self.assertEqual(output.getvalue().count('首个中间结果：-'), 2)
        self.assertIn('[复用成功]', output.getvalue())

    async def test_unconfirmed_manual_mode_stops_before_audio(self):
        ws = FakeSocket()
        original_send = ws.send
        async def bad_config(message):
            event = json.loads(message)
            if event['type'] == 'session.update':
                await ws.events.put({'type': 'session.updated', 'session': {'turn_detection': {'type': 'server_vad'}}})
            else:
                await original_send(message)
        ws.send = bad_config
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, '手动模式'):
            await Probe(ws, timeout_s=1).run([b'\0' * CHUNK_BYTES] * 2)
        self.assertEqual(ws.sent, [])

    async def test_timeout_cleans_up_receiver(self):
        ws = FakeSocket()
        await ws.events.get()
        with self.assertRaises(TimeoutError):
            await Probe(ws, timeout_s=.01).run([b'\0' * CHUNK_BYTES] * 2)
        self.assertFalse(any(t.get_name() == 'asr-probe-receiver' for t in asyncio.all_tasks()))

    def test_wav_is_validated_and_only_pcm_frames_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.wav'
            for rate, channels, width, frames, valid in (
                (16000, 1, 2, 1600, True), (24000, 1, 2, 1600, False),
                (16000, 2, 2, 1600, False), (16000, 1, 1, 1600, False),
                (16000, 1, 2, 0, False), (16000, 1, 2, 100, False),
            ):
                with self.subTest(rate=rate, channels=channels, width=width, frames=frames):
                    pcm = b'\0' * (frames * channels * width)
                    with wave.open(str(path), 'wb') as audio:
                        audio.setparams((channels, width, rate, frames, 'NONE', 'not compressed'))
                        audio.writeframes(pcm)
                    if valid:
                        self.assertEqual(read_wav(path), pcm)
                    else:
                        with self.assertRaises(ValueError):
                            read_wav(path)

    def test_region_url_and_log_redaction(self):
        self.assertIn('qwen3-asr-flash-realtime', websocket_url('workspace-test', 'singapore'))
        self.assertIn('cn-beijing', websocket_url('workspace-test', 'beijing'))
        with self.assertRaises(ValueError):
            websocket_url('bad/workspace', 'singapore')
        output = io.StringIO()
        with redirect_stdout(output):
            Probe(None, redact=('secret-key',)).log('error: secret-key')
        self.assertEqual(output.getvalue(), 'error: [REDACTED]\n')
