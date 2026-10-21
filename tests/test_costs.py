import asyncio
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

import core.costs as costs
from core.costs import CostLedger, quote, AudioCostMeter, costed_call
from voice_pipeline.search_router import JevSearchRouter


class PricingTests(TestCase):
    def test_context_summary_has_separate_component_and_haiku_price(self):
        self.assertEqual(quote('Context', 'claude-haiku-5-5', {'input_tokens': 100, 'output_tokens': 20}),
                         quote('Memory', 'claude-haiku-5-5', {'input_tokens': 100, 'output_tokens': 20}))

    def test_claude_cache_ttls_and_search_are_not_regular_input(self):
        value, reason = quote('Claude', 'claude-sonnet-5', {
            'input_tokens': 100, 'output_tokens': 200,
            'cache_read_input_tokens': 1000, 'cache_creation_input_tokens': 300,
            'cache_creation': {'ephemeral_5m_input_tokens': 200, 'ephemeral_1h_input_tokens': 100},
            'server_tool_use': {'web_search_requests': 2}})
        self.assertEqual(value, Decimal('.0233'))
        self.assertEqual(reason, '')

    def test_unknown_model_is_not_sonnet_price(self):
        self.assertEqual(quote('Claude', 'unknown', {'input_tokens': 1, 'output_tokens': 1})[1],
                         'unknown_model_price')

    def test_haiku_long_context_price(self):
        value, reason = quote('Memory', 'claude-haiku-5-5', {'input_tokens': 100001, 'output_tokens': 2})
        self.assertEqual(value, Decimal('.0500055'))
        self.assertFalse(reason)

    def test_tts_uses_reported_characters_and_no_audio_token_charge(self):
        self.assertEqual(quote('TTS', 'qwen3-tts-flash-realtime', {'characters': 10000, 'output_tokens': 900})[0], Decimal('.13'))
        self.assertTrue(quote('TTS', 'qwen3-tts-flash-realtime', {'output_tokens': 900})[1])

    def test_jev_real_gateway_cost_overrides_rounded_price(self):
        self.assertEqual(quote('Jev', 'typesafe-ai/jev', {'input_tokens': 314, 'gateway_usd': '.000013188'}),
                         (Decimal('.000013188'), ''))
        self.assertTrue(quote('Jev', 'typesafe-ai/jev', {'gateway_usd': 'NaN'})[1])

    def test_omni_all_modalities_and_missing_data(self):
        self.assertEqual(quote('Omni', 'qwen3.8-omni-flash-realtime', {
            'input_tokens_details': {'text_tokens': 100, 'audio_tokens': 100},
            'output_tokens_details': {'text_tokens': 100, 'audio_tokens': 100}})[0], Decimal('.000373'))
        self.assertTrue(quote('Omni', 'qwen3.8-omni-flash-realtime', {})[1])

    def test_asr_is_explicit_reference_not_authoritative_charge(self):
        value, reason = quote('ASR', 'qwen3-asr-flash-realtime', {'sent_audio_seconds': 10})
        self.assertEqual(value, Decimal('.0009'))
        self.assertEqual(reason, 'embedded_asr_billing_unverified')

    def test_independent_asr_uses_reported_duration_and_rejects_invalid_usage(self):
        self.assertEqual(quote('ASR', 'qwen3-asr-flash-realtime', {'duration': 3}),
                         (Decimal('.00027'), ''))
        for duration in (-1, 'NaN', 'Infinity', None):
            self.assertEqual(quote('ASR', 'qwen3-asr-flash-realtime', {'duration': duration})[1],
                             'invalid_usage_or_price')


class LedgerTests(TestCase):
    def test_summary_includes_independent_asr_but_not_embedded_reference(self):
        store = CostLedger(':memory:')
        self.addCleanup(store.close)
        store.record('old', 'ASR', 'qwen3-asr-flash-realtime', {'sent_audio_seconds': 100})
        store.record('new', 'ASR', 'qwen3-asr-flash-realtime', {'duration': 3})
        with patch.object(costs, '_ledger', store), patch.object(costs.logger, 'info') as log:
            costs.log_summary()
        self.assertEqual(log.call_args.args[2], Decimal('.00027'))

    def test_duplicate_event_is_upsert_and_unknown_can_be_resolved(self):
        store = CostLedger(':memory:')
        self.addCleanup(store.close)
        store.record('a', 'TTS', 'qwen3-tts-flash-realtime')
        self.assertEqual(store.summary()['TTS']['unpriced'], 1)
        for _ in range(2):
            store.record('a', 'TTS', 'qwen3-tts-flash-realtime', {'characters': 100})
        self.assertEqual(store.summary()['TTS'], {'usd': '0.0013', 'unpriced': 0, 'events': 1})

    def test_late_incomplete_duplicate_cannot_erase_known_cost(self):
        store = CostLedger(':memory:')
        self.addCleanup(store.close)
        store.record('a', 'TTS', 'qwen3-tts-flash-realtime', {'characters': 100})
        store.record('a', 'TTS', 'qwen3-tts-flash-realtime')
        self.assertEqual(store.summary()['TTS'], {'usd': '0.0013', 'unpriced': 0, 'events': 1})

    def test_two_process_style_connections_and_restart_preserve_totals(self):
        with TemporaryDirectory() as d:
            p = Path(d) / 'costs.db'
            first, second = CostLedger(p), CostLedger(p)
            first.record('a', 'TTS', 'qwen3-tts-flash-realtime', {'characters': 100})
            second.record('b', 'TTS', 'qwen3-tts-flash-realtime', {'characters': 100})
            self.assertEqual(first.summary()['TTS']['events'], 2)
            first.close(); second.close()
            third = CostLedger(p)
            self.assertEqual(third.summary()['TTS']['usd'], '0.0026')
            third.close()

    def test_audio_bytes_not_wall_clock_and_includes_tail_once(self):
        ledger = CostLedger(':memory:')
        self.addCleanup(ledger.close)
        with patch.object(costs, '_ledger', ledger):
            meter = AudioCostMeter('one', 'singapore')
            meter.sent(b'\0' * 320000)
            meter.sent(b'\0' * 32000)
            meter.flush(); meter.flush()
        self.assertEqual(ledger.summary()['ASR']['events'], 1)
        self.assertEqual(Decimal(ledger.summary()['ASR']['usd']), Decimal('0.00099'))
        self.assertEqual(ledger.summary()['ASR']['unpriced'], 1)


class RuntimeTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ledger = CostLedger(':memory:')
        self.patcher = patch.object(costs, '_ledger', self.ledger)
        self.patcher.start()

    async def asyncTearDown(self):
        self.patcher.stop()
        self.ledger.close()

    async def test_timeout_keeps_unknown_not_free(self):
        async def fail():
            raise TimeoutError()
        with self.assertRaises(TimeoutError):
            await costed_call(fail(), component='Claude', model='claude-sonnet-5')
        self.assertEqual(self.ledger.summary()['Claude']['unpriced'], 1)

    async def test_success_resolves_attempt_once(self):
        api = AsyncMock(return_value=NS(model='claude-sonnet-5', usage={'input_tokens': 100, 'output_tokens': 10}))
        await costed_call(api(), component='Claude', model='claude-sonnet-5')
        self.assertEqual(self.ledger.summary()['Claude'], {'usd': '0.0003', 'unpriced': 0, 'events': 1})

    async def test_combined_jev_request_counts_one_not_two_questions(self):
        response = NS(raise_for_status=lambda: None, json=lambda: {
            'model': 'typesafe-ai/jev', 'usage': {'input_tokens': 314},
            'provider_metadata': {'gateway': {'cost': '.000013188'}},
            'answers': {}})
        client = NS(post=AsyncMock(return_value=response))
        router = JevSearchRouter(client, api_key='fake', provider='vercel')
        await router.interruption_and_turn('停', combined_text='停', assistant='abc', question_open=False, history=[])
        self.assertEqual(self.ledger.summary()['Jev'], {'usd': '0.000013188', 'unpriced': 0, 'events': 1})

    async def test_cancelled_attempt_is_still_unknown(self):
        started = asyncio.Event()
        async def wait():
            started.set()
            await asyncio.Future()
        task = asyncio.create_task(costed_call(wait(), component='Jev', model='jev-latest'))
        await started.wait(); task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.ledger.summary()['Jev']['unpriced'], 1)

    async def test_missing_search_count_is_flagged_but_token_cost_kept(self):
        api = AsyncMock(return_value=NS(model='claude-sonnet-5',
            usage={'input_tokens': 100, 'output_tokens': 10},
            content=[NS(type='server_tool_use', name='web_search')]))
        await costed_call(api(), component='Claude', model='claude-sonnet-5')
        row = self.ledger.summary()['Claude']
        self.assertEqual(row['unpriced'], 1)
        self.assertEqual(row['usd'], '0.0003')

    async def test_omni_missing_final_usage_remains_pending(self):
        from voice.qwen_alpha import Receiver
        ws = NS(send=AsyncMock())
        player = NS(finish=lambda _: None)
        receiver = Receiver(ws, player, None, None, NS(record=lambda *_: Decimal(0), total=Decimal(0)), 'session')
        async def stop(*args, **kwargs):
            if args[1] == 'response.create':
                raise ConnectionError()
        with patch('voice.qwen_alpha.send_event', side_effect=stop):
            with self.assertRaises(ConnectionError):
                await receiver._play_report('你好')
        self.assertEqual(self.ledger.summary()['Omni']['unpriced'], 1)
        await receiver.handle({'type': 'response.created', 'response': {'id': 'reply'}})
        event = {'type': 'response.done', 'response': {'id': 'reply', 'usage': {
            'input_tokens_details': {'text_tokens': 100, 'audio_tokens': 100},
            'output_tokens_details': {'text_tokens': 100, 'audio_tokens': 100}}}}
        await receiver.handle(event)
        await receiver.handle(event)
        row = self.ledger.summary()['Omni']
        self.assertEqual(row['events'], 1)
        self.assertEqual(row['unpriced'], 0)
