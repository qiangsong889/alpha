"""仅验证分类适配和 LiveKit 配置，不发网络请求，不打开麦克风。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from livekit.agents import LanguageCode, llm

from agent import AlphaVoiceAgent, create_session
from turn_detector import JevTurnDetector


def context():
    result = llm.ChatContext()
    result.add_message(role="user", content="我想加购物清单")
    result.add_message(role="assistant", content="要加什么？")
    result.add_message(role="user", content="把牛奶")
    result.add_message(role="user", content="和鸡蛋加到购物清单")
    return result


class TurnDetectorTests(unittest.IsolatedAsyncioTestCase):
    def detector(self, decision):
        router = SimpleNamespace(model="fake", turn_complete=AsyncMock(return_value=decision))
        fallback = SimpleNamespace(
            supports_language=AsyncMock(return_value=True),
            predict_end_of_turn=AsyncMock(return_value=.1),
            unlikely_threshold=AsyncMock(return_value=.0066))
        return JevTurnDetector(router, fallback), router, fallback

    async def test_complete_and_incomplete_replace_local_model_and_merge_fragments(self):
        for decision, expected in [(True, 1.), (False, 0.)]:
            detector, router, fallback = self.detector(decision)
            self.assertEqual(await detector.predict_end_of_turn(context()), expected)
            router.turn_complete.assert_awaited_once_with(
                "把牛奶 和鸡蛋加到购物清单", [
                    {"role": "user", "content": "我想加购物清单"},
                    {"role": "assistant", "content": "要加什么？"}])
            fallback.predict_end_of_turn.assert_not_called()
            self.assertEqual(await detector.unlikely_threshold(LanguageCode("zh")), .5)

    async def test_unknown_falls_back_using_original_threshold(self):
        detector, router, fallback = self.detector(None)
        self.assertEqual(await detector.predict_end_of_turn(context()), 1.)
        fallback.predict_end_of_turn.assert_awaited_once()
        fallback.predict_end_of_turn.return_value = .001
        self.assertEqual(await detector.predict_end_of_turn(context()), 0.)

    async def test_cancellation_propagates_and_never_uses_stale_prediction(self):
        detector, router, fallback = self.detector(None)
        entered = asyncio.Event()
        async def hang(*args):
            entered.set()
            await asyncio.Future()
        router.turn_complete.side_effect = hang
        task = asyncio.create_task(detector.predict_end_of_turn(context()))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        fallback.predict_end_of_turn.assert_not_called()

    async def test_outer_timeout_and_empty_input_use_long_window(self):
        detector, router, fallback = self.detector(None)
        self.assertEqual(await detector.predict_end_of_turn(llm.ChatContext()), 0.)
        router.turn_complete.assert_not_called()
        async def hang(*args):
            await asyncio.Future()
        router.turn_complete.side_effect = hang
        self.assertEqual(await detector.predict_end_of_turn(context(), timeout=.01), 0.)

    async def test_jev_session_replaces_detector_and_disables_extra_collection(self):
        router = SimpleNamespace(api_key="fake")
        with patch.dict("os.environ", {"ALPHA_TURN_DETECTOR": "jev",
                                       "ALPHA_COLLECTION_ENABLED": "1", "ALPHA_TURN_MAX_DELAY": "5"}), \
                patch("agent.silero.VAD.load"), patch("agent.MultilingualModel"), \
                patch("agent.openai.STT"), patch("agent.openai.TTS"), \
                patch("agent.tts.StreamAdapter"), patch("agent.AgentSession") as session:
            create_session(jev_router=router)
            self.assertIsInstance(session.call_args.kwargs["turn_handling"]["turn_detection"], JevTurnDetector)
            self.assertEqual(session.call_args.kwargs["turn_handling"]["endpointing"],
                             {"min_delay": .5, "max_delay": 5})
            with patch.dict("os.environ", {"AI_GATEWAY_API_KEY": "", "TYPESAFE_API_KEY": ""}):
                agent = AlphaVoiceAgent(None)
            self.assertFalse(agent.collection_enabled)
            self.assertFalse(session.call_args.kwargs["turn_handling"]["preemptive_generation"]["enabled"])

    async def test_missing_key_does_not_silently_enable_experiment(self):
        with patch.dict("os.environ", {"ALPHA_TURN_DETECTOR": "jev"}), patch("agent.MultilingualModel"):
            with self.assertRaisesRegex(ValueError, "API key"):
                create_session()
