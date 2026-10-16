"""在 voice_pipeline 内运行：.venv/bin/python -m unittest -v test_agent。"""

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, PropertyMock, patch

import httpx
from livekit import rtc
from livekit.agents import Agent, AgentSession, llm, stt, tokenize
from livekit.agents.voice import io
from types import SimpleNamespace
from array import array

from livekit.agents.voice.events import ConversationItemAddedEvent

from agent import AlphaVoiceAgent, create_session, log_conversation, explicit_web_request
from interruptions import DuckingOutput, SpeechGate, is_meaningful
from sounds import confirmation_tone


def stream_response(answer):
    return httpx.Response(200, content=(
        json.dumps({"type": "delta", "text": answer}, ensure_ascii=False) + '\n{"type":"done"}\n'
    ).encode())


async def collect(agent, context):
    return "".join([part async for part in agent.llm_node(context, [], None)])


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_uses_livekit_turn_without_extra_window_or_rule_prompt(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return stream_response("请继续。")

        # 即使保留旧的窗口时长配置，也不会意外重新打开额外等待。
        with patch.dict("os.environ", {"ALPHA_COLLECTION_ENABLED": "0",
                                       "ALPHA_COLLECTION_SECONDS": "30",
                                       "ALPHA_CONTINUATION_SECONDS": "30"}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                agent = AlphaVoiceAgent(client)
                self.assertFalse(agent.collection_enabled)
                for text in ["关厨房灯", "你好", "把牛奶加到清单，然后……"]:
                    context = llm.ChatContext()
                    message = llm.ChatMessage(role="user", content=[text])
                    context.items.append(message)
                    await agent.on_user_turn_completed(context, message)
                    self.assertEqual(agent.collection.parts, [])
                    self.assertEqual(await asyncio.wait_for(collect(agent, context), 0.5), "请继续。")
                self.assertIsNone(agent.collection.prompt)
        self.assertEqual([r["text"] for r in requests], ["关厨房灯", "你好", "把牛奶加到清单，然后……"])

    async def test_extra_collection_is_explicit_opt_in(self):
        with patch.dict("os.environ", {"ALPHA_COLLECTION_ENABLED": "1"}):
            async with httpx.AsyncClient() as client:
                self.assertTrue(AlphaVoiceAgent(client).collection_enabled)

    async def test_search_ack_is_synthesized_without_waiting_for_final_answer(self):
        release = asyncio.Event()
        inputs = []
        frame = rtc.AudioFrame(data=array("h", [0]).tobytes(), sample_rate=24000,
                               num_channels=1, samples_per_channel=1)

        async def text():
            yield "好的，我查一查。"
            await release.wait()
            yield "查到结果了。"

        async def synthesize(agent, stream, settings):
            inputs.append("".join([part async for part in stream]))
            yield frame

        async with httpx.AsyncClient() as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            with patch.object(Agent.default, "tts_node", side_effect=synthesize):
                output = agent.tts_node(text(), None)
                self.assertIs(await asyncio.wait_for(anext(output), 1), frame)
                self.assertEqual(inputs, ["好的，我查一查。"])
                release.set()
                self.assertEqual([f async for f in output], [frame])
                self.assertEqual(inputs, ["好的，我查一查。", "查到结果了。"])

    def test_explicit_search_detection_excludes_local_queries_and_negation(self):
        for text in ["帮我搜一下苹果发布会", "上网查一下天气", "查一下最新新闻", "今天有什么新闻"]:
            self.assertTrue(explicit_web_request(text), text)
        for text in ["购物清单里有什么", "查一下客厅灯状态", "不用上网查", "不要搜索新闻"]:
            self.assertFalse(explicit_web_request(text), text)

    async def test_search_ack_precedes_brain_post_and_sends_dedup_marker(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return stream_response("查到新闻了。")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            context = llm.ChatContext()
            context.add_message(role="user", content="帮我搜一下最新新闻")
            output = agent.llm_node(context, [], None)
            self.assertEqual(await anext(output), "好的，我查一查。")
            self.assertEqual(requests, [])
            self.assertEqual("".join([part async for part in output]), "查到新闻了。")
        self.assertEqual(requests, [{"text": "帮我搜一下最新新闻", "search_announced": True}])

    async def test_interrupting_early_search_ack_sends_no_brain_request(self):
        transport = httpx.MockTransport(lambda request: self.fail("不应提交被取消的请求"))
        async with httpx.AsyncClient(transport=transport) as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            context = llm.ChatContext()
            context.add_message(role="user", content="上网查一下天气")
            output = agent.llm_node(context, [], None)
            self.assertEqual(await anext(output), "好的，我查一查。")
            await output.aclose()

    async def test_incomplete_prompt_does_not_call_brain_then_continuation_submits(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return stream_response("已完成。")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            agent.collection.continuation_s = 0
            context = llm.ChatContext()
            first = llm.ChatMessage(role="user", content=["关灯，然后……"])
            context.items.append(first)
            await agent.on_user_turn_completed(context, first)
            self.assertIn("接下来", await collect(agent, context))
            self.assertEqual(requests, [])
            self.assertEqual(await collect(agent, context), "")
            second = llm.ChatMessage(role="user", content=["加牛奶到购物清单"])
            context.items.append(second)
            await agent.on_user_turn_completed(context, second)
            self.assertEqual(await collect(agent, context), "已完成。")
            self.assertEqual(requests, [{"text": "关灯，然后……；加牛奶到购物清单"}])

    async def test_collection_merges_turns_before_any_brain_request(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return stream_response("已完成。")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0.03)
            context = llm.ChatContext()
            first = llm.ChatMessage(role="user", content=["关厨房灯"])
            context.items.append(first)
            await agent.on_user_turn_completed(context, first)
            old = asyncio.create_task(collect(agent, context))
            await asyncio.sleep(0.01)
            agent.collection.speech_started()
            await asyncio.sleep(0.04)
            self.assertEqual(requests, [])
            old.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await old
            agent.collection.speech_stopped()
            agent.collection.transcript_resolved()
            second = llm.ChatMessage(role="user", content=["再加牛奶"])
            context.items.append(second)
            await agent.on_user_turn_completed(context, second)
            self.assertEqual(await collect(agent, context), "已完成。")
        self.assertEqual(requests, [{"text": "关厨房灯；再加牛奶"}])

    async def test_ha_signal_plays_local_tone_without_tts_or_chat_message(self):
        class Sink(io.AudioOutput):
            def __init__(self):
                super().__init__(label="tone-test", capabilities=io.AudioOutputCapabilities(pause=False))
                self.frames = []
                self.played = asyncio.Event()

            async def capture_frame(self, frame):
                await super().capture_frame(frame)
                self.frames.append(frame)
                if len(self.frames) == 1:
                    self.on_playback_started(created_at=time.time())

            def flush(self):
                super().flush()
                self.on_playback_finished(playback_position=sum(f.duration for f in self.frames), interrupted=False)
                self.played.set()

            def clear_buffer(self):
                self.on_playback_finished(playback_position=0, interrupted=True)

        sink = Sink()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, content=b'{"type":"signal","name":"ha_done"}\n{"type":"done"}\n'
        ))) as client:
            session = AgentSession(turn_handling={"interruption": {"enabled": False}})
            session.output.audio = sink
            try:
                await session.start(agent=AlphaVoiceAgent(client, collection_window_s=0))
                await session.generate_reply(user_input="开灯")
                await asyncio.wait_for(sink.played.wait(), 1)
                self.assertEqual(sum(f.samples_per_channel for f in sink.frames), 2880)
                self.assertFalse(any(item.type == "message" and item.role == "assistant"
                                     for item in session.history.items))
            finally:
                await session.aclose()

    async def test_confirmation_tone_has_fades_and_safe_amplitude(self):
        frames = [frame async for frame in confirmation_tone()]
        samples = [sample for frame in frames for sample in frame.data]
        self.assertEqual(len(samples), 2880)
        self.assertEqual(samples[0], 0)
        self.assertEqual(samples[-1], 0)
        self.assertLessEqual(max(abs(sample) for sample in samples), 4916)

    async def test_real_session_manual_interrupt_only_after_meaningful_transcript(self):
        requested, release = asyncio.Event(), asyncio.Event()

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                requested.set()
                await release.wait()
                yield b'{"type":"delta","text":"report"}\n{"type":"done"}\n'

        async def transcript(text):
            yield stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                  alternatives=[stt.SpeechData(language="zh", text=text)])

        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=Body()))) as client:
            session = AgentSession(turn_handling={"interruption": {
                "enabled": False, "discard_audio_if_uninterruptible": False,
            }})
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            try:
                await session.start(agent=agent)
                report = session.generate_reply(user_input="汇报一下")
                await asyncio.wait_for(requested.wait(), 1)
                self.assertFalse(report.allow_interruptions)
                with patch.object(Agent.default, "stt_node", return_value=transcript("咳")):
                    self.assertEqual([e async for e in agent.stt_node(None, None)], [])
                self.assertFalse(report.interrupted)
                with patch.object(Agent.default, "stt_node", return_value=transcript("停")):
                    result = [e async for e in agent.stt_node(None, None)]
                self.assertEqual(result[0].alternatives[0].text, "停")
                self.assertTrue(report.interrupted)
            finally:
                release.set()
                await session.aclose()

    def test_noise_filter_preserves_meaningful_short_replies(self):
        for text in ["", "...", "咳", "咳咳。", "[cough]", "（咳嗽）", "[breathing]"]:
            with self.subTest(text=text):
                self.assertFalse(is_meaningful(text))
        for text in ["对", "對", "停", "是的", "好", "嗯", "改成九点", "我咳嗽了，等一下"]:
            with self.subTest(text=text):
                self.assertTrue(is_meaningful(text))

    async def test_stt_drops_noise_before_framework_and_only_interrupts_for_final_text(self):
        async def events():
            for kind, text in [
                (stt.SpeechEventType.INTERIM_TRANSCRIPT, "停"),
                (stt.SpeechEventType.FINAL_TRANSCRIPT, "咳咳"),
                (stt.SpeechEventType.FINAL_TRANSCRIPT, "停"),
            ]:
                yield stt.SpeechEvent(type=kind, alternatives=[stt.SpeechData(language="zh", text=text)])

        async with httpx.AsyncClient() as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            agent.interruptions.speech_started()
            session = SimpleNamespace(agent_state="speaking", interrupt=AsyncMock())
            with patch.object(AlphaVoiceAgent, "session", new_callable=PropertyMock, return_value=session), \
                    patch.object(Agent.default, "stt_node", return_value=events()):
                output = [event async for event in agent.stt_node(None, None)]
            self.assertEqual([event.alternatives[0].text for event in output], ["停"])
            session.interrupt.assert_awaited_once_with(force=True)
            self.assertEqual(agent.interruptions.gain, 1.0)

    async def test_noise_alone_keeps_original_report_running(self):
        async def events():
            yield stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                  alternatives=[stt.SpeechData(language="zh", text="[cough]")])

        async with httpx.AsyncClient() as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            agent.interruptions.speech_started()
            session = SimpleNamespace(agent_state="speaking", interrupt=AsyncMock())
            with patch.object(AlphaVoiceAgent, "session", new_callable=PropertyMock, return_value=session), \
                    patch.object(Agent.default, "stt_node", return_value=events()):
                self.assertEqual([e async for e in agent.stt_node(None, None)], [])
            session.interrupt.assert_not_called()
            self.assertEqual(agent.interruptions.gain, 1.0)

    async def test_missing_transcript_restores_gain_after_timeout(self):
        gate = SpeechGate(timeout_s=0.01)
        gate.speech_started()
        self.assertEqual(gate.gain, 0.2)
        gate.speech_stopped()
        await asyncio.sleep(0.03)
        self.assertEqual(gate.gain, 1.0)
        gate.close()

    async def test_close_cancels_pending_gain_timer(self):
        gate = SpeechGate(timeout_s=0.01)
        gate.speech_started()
        gate.speech_stopped()
        gate.close()
        self.assertIsNone(gate._timer)
        self.assertEqual(gate.gain, 1.0)

    async def test_ducking_scales_output_without_mutating_or_clearing_original(self):
        class Sink(io.AudioOutput):
            def __init__(self):
                super().__init__(label="fake", capabilities=io.AudioOutputCapabilities(pause=False))
                self.frames = []
                self.cleared = False

            async def capture_frame(self, frame):
                await super().capture_frame(frame)
                self.frames.append(frame)

            def clear_buffer(self):
                self.cleared = True

            def flush(self):
                super().flush()

        sink, gate = Sink(), SpeechGate()
        output = DuckingOutput(sink, lambda: gate.gain)
        original = rtc.AudioFrame(data=array("h", [1000, -1000]).tobytes(),
                                  sample_rate=24000, num_channels=1, samples_per_channel=2)
        gate.speech_started()
        await output.capture_frame(original)
        gate.restore()
        await output.capture_frame(original)
        self.assertEqual(list(sink.frames[0].data), [200, -200])
        self.assertEqual(list(sink.frames[1].data), [1000, -1000])
        self.assertEqual(list(original.data), [1000, -1000])
        self.assertFalse(sink.cleared)

    async def test_short_chinese_sentence_is_emitted_before_end_of_input(self):
        tokenizer = tokenize.blingfire.SentenceTokenizer(
            min_sentence_len=4, stream_context_len=2, retain_format=True,
        )
        stream = tokenizer.stream()
        try:
            stream.push_text("清单里有牛奶。你还需要")
            token = await asyncio.wait_for(anext(stream), 1)
            self.assertEqual(token.token, "清单里有牛奶。")
        finally:
            await stream.aclose()

    async def test_real_session_runs_custom_node_without_provider_llm(self):
        requests = []

        def handle(request):
            requests.append(request)
            return stream_response("已关闭客厅灯。")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            session = AgentSession()
            try:
                await session.start(agent=AlphaVoiceAgent(client, collection_window_s=0))
                await session.generate_reply(user_input="把客厅灯关了")
                self.assertEqual(len(requests), 1)
                self.assertEqual(session.history.items[-1].text_content, "已关闭客厅灯。")
            finally:
                await session.aclose()

    def test_handoff_is_not_a_spoken_message(self):
        event = ConversationItemAddedEvent(
            item=llm.AgentHandoff(old_agent_id="old", new_agent_id="voice")
        )
        log_conversation(event)

    def test_latency_uses_playout_metric(self):
        item = llm.ChatMessage(role="assistant", content=["已关闭。"],
                               metrics={"e2e_latency": 2.34})
        with self.assertLogs("alpha.voice_pipeline", level="INFO") as logs:
            log_conversation(ConversationItemAddedEvent(item=item))
        self.assertIn("首帧播放：2.34s", logs.output[0])

    async def test_only_latest_user_text_is_posted_and_answer_returned(self):
        requests = []

        def handle(request):
            requests.append(request)
            self.assertEqual(request.content, '{"text":"对"}'.encode())
            return stream_response("听到了。")

        context = llm.ChatContext()
        context.add_message(role="user", content="把客厅灯关了")
        context.add_message(role="assistant", content="已关闭客厅灯。")
        context.add_message(role="user", content="对")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            answer = await collect(AlphaVoiceAgent(client, collection_window_s=0), context)
        self.assertEqual(answer, "听到了。")
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), "http://127.0.0.1:8765/ask/stream")

    async def test_timeout_is_not_retried(self):
        calls = []

        def handle(request):
            calls.append(request)
            raise httpx.ReadTimeout("timeout", request=request)

        context = llm.ChatContext()
        context.add_message(role="user", content="加牛奶")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with self.assertLogs("alpha.voice_pipeline", level="ERROR"):
                answer = await collect(AlphaVoiceAgent(client, collection_window_s=0), context)
        self.assertEqual(len(calls), 1)
        self.assertIn("可能已经生效", answer)

    async def test_cancellation_propagates(self):
        async def handle(request):
            raise asyncio.CancelledError()

        context = llm.ChatContext()
        context.add_message(role="user", content="你好")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(asyncio.CancelledError):
                await collect(AlphaVoiceAgent(client, collection_window_s=0), context)

    async def test_text_is_forwarded_before_stream_finishes(self):
        gate = asyncio.Event()

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield '{"type":"delta","text":"第一句。"}\n'.encode()
                await gate.wait()
                yield '{"type":"delta","text":"第二句。"}\n{"type":"done"}\n'.encode()

        context = llm.ChatContext()
        context.add_message(role="user", content="你好")
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=Body())
        )) as client:
            output = AlphaVoiceAgent(client, collection_window_s=0).llm_node(context, [], None)
            self.assertEqual(await anext(output), "第一句。")
            gate.set()
            self.assertEqual("".join([part async for part in output]), "第二句。")

    async def test_truncated_stream_is_not_reported_as_complete_or_retried(self):
        calls = []

        def handle(request):
            calls.append(request)
            return httpx.Response(200, content=b'{"type":"delta","text":"hello"}\n')

        context = llm.ChatContext()
        context.add_message(role="user", content="你好")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with self.assertLogs("alpha.voice_pipeline", level="ERROR"):
                text = await collect(AlphaVoiceAgent(client, collection_window_s=0), context)
        self.assertIn("先查询确认", text)
        self.assertEqual(len(calls), 1)

    async def test_configuration_uses_no_llm_and_disables_preemptive_requests(self):
        with patch("agent.silero.VAD.load"), patch("agent.MultilingualModel"), \
                patch("agent.openai.STT") as stt, patch("agent.openai.TTS") as tts, \
                patch("agent.AgentSession") as session, patch("agent.tts.StreamAdapter"), \
                patch("agent.tokenize.blingfire.SentenceTokenizer") as tokenizer:
            create_session()
        self.assertEqual(stt.call_args.kwargs["language"], "zh")
        self.assertTrue(stt.call_args.kwargs["use_realtime"])
        self.assertEqual(tts.call_args.kwargs["model"], "gpt-4o-mini-tts")
        self.assertEqual(tts.call_args.kwargs["response_format"], "pcm")
        self.assertEqual(tokenizer.call_args.kwargs["min_sentence_len"], 4)
        self.assertGreater(tts.call_args.kwargs["speed"], 1.0)
        options = session.call_args.kwargs
        self.assertIsNone(options["llm"])
        self.assertFalse(options["turn_handling"]["preemptive_generation"]["enabled"])
        self.assertEqual(options["turn_handling"]["interruption"], {
            "mode": "vad", "enabled": False,
            "discard_audio_if_uninterruptible": False,
        })


if __name__ == "__main__":
    unittest.main()
