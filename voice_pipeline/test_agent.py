"""在 voice_pipeline 内运行：.venv/bin/python -m unittest -v test_agent。"""

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import httpx
from livekit import rtc
from livekit.agents import Agent, AgentSession, llm, stt, tokenize, tts
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
    async def asyncSetUp(self):
        # 测试不能因为用户后来填入真实密钥而意外发出 Jev 请求。
        env = patch.dict("os.environ", {"ALPHA_SEARCH_ROUTER": "rules", "TYPESAFE_API_KEY": "",
                                       "AI_GATEWAY_API_KEY": "", "JEV_PROVIDER": "typesafe",
                                       "ALPHA_JEV_INTERRUPTIONS": "0", "ALPHA_TURN_DETECTOR": "livekit"})
        env.start()
        self.addCleanup(env.stop)

    async def test_jev_gate_filters_before_livekit_or_brain_sees_text(self):
        for decision in [True, False, None]:
            with self.subTest(decision=decision):
                async def events():
                    yield stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                          alternatives=[stt.SpeechData(language="zh", text="小王晚上吃什么")])
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda request: self.fail("STT 门控不应请求大脑"))) as client:
                    agent = AlphaVoiceAgent(client)
                    agent.jev_interruptions = True
                    agent._reply_text = "购物清单里有牛奶。"
                    agent.search_router.interruption = AsyncMock(return_value=decision)
                    agent.interruptions.speech_started()
                    session = SimpleNamespace(agent_state="speaking", interrupt=AsyncMock())
                    with patch.object(AlphaVoiceAgent, "session", new_callable=PropertyMock, return_value=session), \
                            patch.object(Agent.default, "stt_node", return_value=events()):
                        output = [event async for event in agent.stt_node(None, None)]
                    self.assertEqual(len(output), int(decision is True))
                    self.assertEqual(session.interrupt.await_count, int(decision is True))
                    self.assertEqual(agent.interruptions.gain, 1.0)
                    agent.search_router.interruption.assert_awaited_once_with(
                        "小王晚上吃什么", assistant="购物清单里有牛奶。", question_open=False)

    async def test_wake_name_bypasses_jev_and_question_window_accepts_short_answers(self):
        async with httpx.AsyncClient() as client:
            agent = AlphaVoiceAgent(client)
            agent.jev_interruptions = True
            agent.search_router.interruption = AsyncMock(return_value=False)
            session = SimpleNamespace(agent_state="speaking")
            with patch.object(AlphaVoiceAgent, "session", new_callable=PropertyMock, return_value=session):
                self.assertTrue(await agent._accept_transcript("Alpha，停一下"))
                agent.search_router.interruption.assert_not_called()
                with patch("agent.time.monotonic", return_value=100):
                    agent._reply_played(ConversationItemAddedEvent(item=llm.ChatMessage(
                        role="assistant", content=["你是说牛奶吗？"])))
                with patch("agent.time.monotonic", return_value=105):
                    self.assertTrue(await agent._accept_transcript("对。"))
                    self.assertTrue(await agent._accept_transcript("不要了"))
                agent.search_router.interruption.assert_not_called()
                with patch("agent.time.monotonic", return_value=111):
                    self.assertFalse(await agent._accept_transcript("对"))
                agent.search_router.interruption.assert_awaited_once()

    async def test_interrupted_question_does_not_open_answer_window(self):
        async with httpx.AsyncClient() as client:
            agent = AlphaVoiceAgent(client)
            agent.jev_interruptions = True
            agent.search_router.interruption = AsyncMock(return_value=False)
            agent._reply_played(ConversationItemAddedEvent(item=llm.ChatMessage(
                role="assistant", content=["要牛奶吗？"], interrupted=True)))
            session = SimpleNamespace(agent_state="speaking")
            with patch.object(AlphaVoiceAgent, "session", new_callable=PropertyMock, return_value=session):
                self.assertFalse(await agent._accept_transcript("对"))
            self.assertEqual(agent._question_deadline, 0)

    async def test_speech_that_started_during_report_stays_gated_after_playback_ends(self):
        async with httpx.AsyncClient() as client:
            agent = AlphaVoiceAgent(client)
            agent.jev_interruptions = True
            agent.search_router.interruption = AsyncMock(return_value=False)
            agent.interruptions.overlapped = True
            agent.interruptions.utterance_started_at = 99
            with patch("agent.time.monotonic", return_value=100):
                agent._reply_played(ConversationItemAddedEvent(item=llm.ChatMessage(
                    role="assistant", content=["还有吗？"])))
            session = SimpleNamespace(agent_state="listening")
            with patch.object(AlphaVoiceAgent, "session", new_callable=PropertyMock, return_value=session), \
                    patch("agent.time.monotonic", return_value=101):
                self.assertFalse(await agent._accept_transcript("对"))
            agent.search_router.interruption.assert_awaited_once()

    async def test_idle_requests_do_not_pay_extra_interruption_classification(self):
        async with httpx.AsyncClient() as client:
            agent = AlphaVoiceAgent(client)
            agent.jev_interruptions = True
            agent.search_router.interruption = AsyncMock()
            session = SimpleNamespace(agent_state="listening")
            with patch.object(AlphaVoiceAgent, "session", new_callable=PropertyMock, return_value=session):
                self.assertTrue(await agent._accept_transcript("买牛奶"))
            agent.search_router.interruption.assert_not_called()

    async def test_jev_controls_ack_and_never_blocks_forwarding_to_brain(self):
        for decision in [True, False, None]:
            requests = []
            def handle(request):
                requests.append(json.loads(request.content))
                return stream_response("回答。")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                agent = AlphaVoiceAgent(client, collection_window_s=0)
                agent.search_router_mode = "jev"
                agent.search_router.classify = AsyncMock(return_value=decision)
                context = llm.ChatContext()
                context.add_message(role="user", content="帮我查一下附近餐馆")
                context.add_message(role="assistant", content="附近有中餐馆。")
                context.add_message(role="user", content="台湾的呢？")
                result = await collect(agent, context)
                self.assertEqual(result, ("我查一下。" if decision is True else "") + "回答。")
                self.assertEqual(requests, [{"text": "台湾的呢？", **({"search_announced": True} if decision is True else {})}])
                agent.search_router.classify.assert_awaited_once_with("台湾的呢？", [
                    {"role": "user", "content": "帮我查一下附近餐馆"},
                    {"role": "assistant", "content": "附近有中餐馆。"},
                ])

    async def test_jev_timeout_goes_direct_to_brain_even_if_keyword_rule_would_match(self):
        requests = []
        async def handle(request):
            if request.url.host == "api.typesafe.ai":
                await asyncio.Future()
            requests.append(json.loads(request.content))
            return stream_response("大脑回答。")
        with patch.dict("os.environ", {"ALPHA_SEARCH_ROUTER": "jev", "TYPESAFE_API_KEY": "fake",
                                       "JEV_TIMEOUT_MS": "10"}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                agent = AlphaVoiceAgent(client, collection_window_s=0)
                context = llm.ChatContext()
                context.add_message(role="user", content="帮我上网查一下资料")
                result = await asyncio.wait_for(collect(agent, context), 0.5)
        self.assertEqual(result, "大脑回答。")
        self.assertEqual(requests, [{"text": "帮我上网查一下资料"}])

    def test_log_search_phrases_and_contextual_followups_are_detected(self):
        for text in ["帮我查一下家里附近好吃的中餐馆", "我帮我查一下Easville有没有RAS。"]:
            self.assertTrue(explicit_web_request(text), text)
        for text in ["有没有台湾的小吃的餐馆", "有没 没有台湾的小吃的餐馆。", "还有别家吗", "台湾的呢"]:
            self.assertTrue(explicit_web_request(text, web_context=True), text)
            self.assertFalse(explicit_web_request(text), text)
        for text in ["查一下购物清单", "查一下客厅灯的状态", "查一下明天的天气", "不用查了", "算了"]:
            self.assertFalse(explicit_web_request(text, web_context=True), text)

    async def test_search_followup_is_announced_before_post_and_local_turn_resets_context(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return stream_response("这是结果。")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            for text, expected_ack in [("帮我查一下家里附近好吃的中餐馆", True),
                                       ("有没 没有台湾的小吃的餐馆。", True),
                                       ("查一下购物清单", False), ("还有别家吗", False)]:
                context = llm.ChatContext()
                context.add_message(role="user", content=text)
                output = agent.llm_node(context, [], None)
                count = len(requests)
                first = await anext(output)
                if expected_ack:
                    self.assertEqual(first, "我查一下。")
                    self.assertEqual(len(requests), count)
                else:
                    self.assertEqual(first, "这是结果。")
                _ = [part async for part in output]
                self.assertEqual(requests[-1].get("search_announced", False), expected_ack)

    async def test_backend_search_ack_enables_next_turn_followup(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: stream_response("我查一下。"))) as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            context = llm.ChatContext()
            context.add_message(role="user", content="附近有没有餐馆")
            await collect(agent, context)
            self.assertTrue(agent._web_context)

    async def test_search_post_does_not_require_speaking_state(self):
        session = SimpleNamespace(output=SimpleNamespace(audio_enabled=True, audio=object()),
                                  on=Mock(), off=Mock())
        requests = []
        def handle(request):
            requests.append(request)
            return stream_response("查到了。")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            context = llm.ChatContext()
            context.add_message(role="user", content="上网查一下资料")
            with patch.object(AlphaVoiceAgent, "session", new_callable=PropertyMock, return_value=session):
                output = agent.llm_node(context, [], None)
                self.assertEqual(await anext(output), "我查一下。")
                self.assertEqual(await asyncio.wait_for(anext(output), 0.5), "查到了。")
                self.assertEqual([part async for part in output], [])
                self.assertEqual(len(requests), 1)
                session.on.assert_not_called()
                session.off.assert_not_called()

    async def test_real_session_plays_search_ack_while_brain_result_is_still_blocked(self):
        release, requested, audible = asyncio.Event(), asyncio.Event(), asyncio.Event()
        tts_release = asyncio.Event()
        synthesized = []

        class SlowAnswer(httpx.AsyncByteStream):
            async def __aiter__(self):
                requested.set()
                await release.wait()
                yield '{"type":"delta","text":"查到结果了。"}\n{"type":"done"}\n'.encode()

        class FakeStream(tts.ChunkedStream):
            async def _run(self, output_emitter):
                synthesized.append(self.input_text)
                if self.input_text == "我查一下。":
                    await tts_release.wait()  # 模拟 TTS 首包慢，但不应挡住搜索请求。
                output_emitter.initialize(request_id="fake", sample_rate=24000,
                                          num_channels=1, mime_type="audio/pcm", frame_size_ms=20)
                output_emitter.push(bytes(4800))
                output_emitter.end_input()

        class FakeTTS(tts.TTS):
            def __init__(self):
                super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                                 sample_rate=24000, num_channels=1)

            def synthesize(self, text, *, conn_options):
                return FakeStream(tts=self, input_text=text, conn_options=conn_options)

        class Sink(io.AudioOutput):
            def __init__(self):
                super().__init__(label="ack-order-test", capabilities=io.AudioOutputCapabilities(pause=False))
                self.duration = 0

            async def capture_frame(self, frame):
                await super().capture_frame(frame)
                if self.duration == 0:
                    self.on_playback_started(created_at=time.time())
                self.duration += frame.duration
                audible.set()

            def flush(self):
                super().flush()
                self.on_playback_finished(playback_position=self.duration, interrupted=False)

            def clear_buffer(self):
                self.on_playback_finished(playback_position=self.duration, interrupted=True)

        def handle_request(request):
            self.assertFalse(audible.is_set(), "TTS 仍被测试阻塞，查询应已并行发起")
            return httpx.Response(200, stream=SlowAnswer())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as client:
            adapter = tts.StreamAdapter(tts=FakeTTS(), sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(
                min_sentence_len=4, stream_context_len=2, retain_format=True))
            session = AgentSession(tts=adapter, turn_handling={"interruption": {"enabled": False}})
            session.output.audio = Sink()
            handle = None
            try:
                await session.start(agent=AlphaVoiceAgent(client, collection_window_s=0))
                handle = session.generate_reply(user_input="帮我查一下家里附近好吃的中餐馆")
                await asyncio.wait_for(requested.wait(), 1)
                self.assertFalse(audible.is_set())
                tts_release.set()
                await asyncio.wait_for(audible.wait(), 1)
                self.assertFalse(release.is_set())
                self.assertEqual(synthesized, ["我查一下。"])
            finally:
                tts_release.set()
                release.set()
                if handle is not None:
                    await handle
                await session.aclose()
                await adapter.aclose()
            self.assertEqual(synthesized, ["我查一下。", "查到结果了。"])

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
            yield "我查一下。"
            await release.wait()
            yield "查到了。"

        async def synthesize(agent, stream, settings):
            inputs.append("".join([part async for part in stream]))
            yield frame

        async with httpx.AsyncClient() as client:
            agent = AlphaVoiceAgent(client, collection_window_s=0)
            with patch.object(Agent.default, "tts_node", side_effect=synthesize):
                output = agent.tts_node(text(), None)
                self.assertIs(await asyncio.wait_for(anext(output), 1), frame)
                self.assertEqual(inputs, ["我查一下。"])
                release.set()
                self.assertEqual([f async for f in output], [frame])
                self.assertEqual(inputs, ["我查一下。", "查到了。"])

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
            self.assertEqual(await anext(output), "我查一下。")
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
            self.assertEqual(await anext(output), "我查一下。")
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
