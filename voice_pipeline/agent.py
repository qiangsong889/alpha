"""本机语音入口：LiveKit 负责听和说，brain.server 负责理解和执行工具。

运行：.venv/bin/python agent.py console
首次运行先下载轮次检测模型：.venv/bin/python agent.py download-files
API 按 LiveKit Agents 1.8.8 编写。
"""

from __future__ import annotations

import logging
import json
import math
import os
import re
import time
from pathlib import Path
from collections.abc import AsyncIterable, AsyncIterator

import httpx
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, llm, stt, tokenize, tts
from livekit.agents.voice import ModelSettings
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    UserInputTranscribedEvent,
    MetricsCollectedEvent,
    UserStateChangedEvent,
)
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

if __package__:
    from .collection import RequestCollection
    from .interruptions import SpeechGate, is_meaningful
    from .sounds import confirmation_tone
    from .search_router import JevSearchRouter
    from .turn_detector import JevTurnDetector
else:
    from collection import RequestCollection
    from interruptions import SpeechGate, is_meaningful
    from sounds import confirmation_tone
    from search_router import JevSearchRouter
    from turn_detector import JevTurnDetector


DIRECTORY = Path(__file__).resolve().parent
# 独立的语音配置优先；没有配置的项沿用 Alpha 的 .env，不依赖启动目录。
load_dotenv(DIRECTORY / ".env")
load_dotenv(DIRECTORY.parent / ".env")

logger = logging.getLogger("alpha.voice_pipeline")
BRAIN_URL = "http://127.0.0.1:8765/ask/stream"
SEARCH_ACK = "我查一下。"
server = AgentServer()


def explicit_web_request(text: str, *, web_context: bool = False) -> bool:
    """识别搜索表达和紧接搜索的追问；这是本地启发式，不做模型推理。"""
    for clause in re.split(r"[，。！？；,;!?]", text):
        if re.search(r"(?:不要|不用|不需要|别|別|不必|无需|無需).{0,8}(?:搜|上网|上網|联网|聯網|查)", clause):
            continue
        if re.fullmatch(r"\s*(?:算了|不用了|取消|等等|等一下|我还没说完)\s*", clause):
            continue
        if re.search(
            r"搜索|搜一下|搜一搜|(?:上网|上網|联网|聯網|网上|網上|网络上).{0,8}(?:查|搜)"
            r"|(?:最新|今天|今日).{0,6}(?:新闻|新聞|消息|资讯|資訊)", clause
        ):
            return True
        # 模糊的“查一下”和追问只有在不是本地工具问题时才提前提示。
        if re.search(r"清单|清單|待办|待辦|日历|日曆|提醒|购物|購物|灯|燈|空调|空調|窗帘|窗簾|设备|設備|天气|天氣|气温|氣溫", clause):
            continue
        if re.search(r"查一下|查一查|查查|查找|找一下|找一找|推荐|推薦", clause):
            return True
        if web_context and re.search(r"有没有|有没|有沒有|有沒|没有|沒有|还有|還有|哪[个個家]|换|換|呢|怎么|怎麼|如何", clause):
            return True
    return False


def last_user_text(chat_ctx: llm.ChatContext) -> str:
    """只转发最后一句用户文字；对话历史由 AlphaBrain 自己维护。"""
    for item in reversed(chat_ctx.items):
        if isinstance(item, llm.ChatMessage) and item.role == "user":
            return (item.text_content or "").strip()
    return ""


class BrainNodeLLM(llm.LLM):
    """启用框架的回答流程；没有模型客户端，也不发起推理请求。

    1.8.8 在 llm=None 时会直接跳过回答，所以需要这个占位对象。
    真正的生成入口始终是 AlphaVoiceAgent.llm_node。
    """

    @property
    def model(self) -> str:
        return "AlphaBrain"

    @property
    def provider(self) -> str:
        return "local-http"

    def chat(self, **kwargs) -> llm.LLMStream:
        raise RuntimeError("必须通过 AlphaVoiceAgent.llm_node 调用本机大脑")


class AlphaVoiceAgent(Agent):
    def __init__(self, http_client: httpx.AsyncClient, *, collection_window_s: float | None = None) -> None:
        super().__init__(
            instructions="你是 Alpha 的语音接口。原样朗读大脑返回的回答。",
            llm=BrainNodeLLM(),
        )
        self.http_client = http_client
        self.interruptions = SpeechGate()
        self._web_context = False
        gateway_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
        provider = os.getenv("JEV_PROVIDER", "vercel" if gateway_key else "typesafe").strip().lower()
        key = gateway_key if provider == "vercel" else os.getenv("TYPESAFE_API_KEY", "").strip()
        self.search_router_mode = os.getenv("ALPHA_SEARCH_ROUTER", "jev" if key else "rules").strip().lower()
        if self.search_router_mode not in {"jev", "rules", "off"}:
            raise ValueError("ALPHA_SEARCH_ROUTER 必须是 jev、rules 或 off")
        self.search_router = JevSearchRouter(
            http_client, api_key=key, provider=provider, model=os.getenv("JEV_MODEL") or None,
            timeout_s=float(os.getenv("JEV_TIMEOUT_MS", "300")) / 1000,
            min_confidence=float(os.getenv("JEV_MIN_CONFIDENCE", "0.8")),
        )
        self.jev_interruptions = os.getenv("ALPHA_JEV_INTERRUPTIONS", "1" if key else "0") == "1"
        self.answer_window_s = float(os.getenv("ALPHA_ANSWER_WINDOW_SECONDS", "10"))
        if not math.isfinite(self.answer_window_s) or self.answer_window_s <= 0:
            raise ValueError("ALPHA_ANSWER_WINDOW_SECONDS 必须大于零且有限")
        self._reply_text = ""
        self._question_finished_at = 0.0
        self._question_deadline = 0.0
        # 默认只由 LiveKit 判断轮次结束，不再叠加业务层的静音计时。
        # 显式传 window 参数供旧窗口的回归测试使用；运行时可通过环境开关对比。
        self.collection_enabled = (collection_window_s is not None
                                   or os.getenv("ALPHA_COLLECTION_ENABLED", "0") == "1")
        if os.getenv("ALPHA_TURN_DETECTOR", "livekit").strip().lower() == "jev":
            self.collection_enabled = False  # 替换轮次模型，不叠加旧的收集窗口。
        self.collection = RequestCollection(
            float(os.getenv("ALPHA_COLLECTION_SECONDS", "2"))
            if collection_window_s is None else collection_window_s,
            continuation_s=float(os.getenv("ALPHA_CONTINUATION_SECONDS", "5")),
        )

    async def on_enter(self) -> None:
        self.interruptions.attach(self.session)
        self.session.on("conversation_item_added", self._reply_played)
        if self.collection_enabled:
            self.session.on("user_state_changed", self._collection_user_state)

    async def on_exit(self) -> None:
        self.interruptions.close()
        self.session.off("conversation_item_added", self._reply_played)
        if self.collection_enabled:
            self.session.off("user_state_changed", self._collection_user_state)
        self.collection.close()

    def _reply_played(self, event: ConversationItemAddedEvent) -> None:
        # 当前安装的 LiveKit 在实际播放结束后发出 assistant 的此事件。
        item = event.item
        if not isinstance(item, llm.ChatMessage) or item.role != "assistant":
            return
        self._question_deadline = 0.0
        text = (item.text_content or "").strip()
        if not item.interrupted and re.search(r"[？?]$|(?:吗|嗎|呢)[。！!]*$", text):
            self._question_finished_at = time.monotonic()
            self._question_deadline = self._question_finished_at + self.answer_window_s

    async def _accept_transcript(self, text: str) -> bool:
        if not self.jev_interruptions:
            return True  # 无 key 或显式关闭实验时，保持原先的转写打断行为。
        if re.match(r"^\s*(?:(?:嘿|你好|hey)[，,\s]*)?(?:alpha\b|阿尔法|阿爾法)", text, re.I):
            logger.info("[打断判断] 明确呼叫 Alpha，接收")
            return True
        now = time.monotonic()
        started = self.interruptions.utterance_started_at
        in_answer_window = (now <= self._question_deadline and
                            (started is None or started >= self._question_finished_at))
        if in_answer_window and re.fullmatch(
                r"\s*(?:对|對|是的|不是|不对|不對|好|好的|可以|不用了|不要了|取消|嗯)[。！!，,\s]*", text):
            logger.info("[打断判断] 提问后的短回答，接收")
            return True
        overlapping = self.interruptions.overlapped or self.session.agent_state == "speaking"
        if not overlapping:
            return True  # 目前尚未接入待机时的 wake word 门控。
        # 在 STT 的 yield 之前决定；只有明确的旁聊才丢弃，不能把不确定当成拒绝。
        decision = await self.search_router.interruption(
            text, assistant=self._reply_text, question_open=in_answer_window)
        if decision is False:
            logger.info("[打断忽略] 明确旁聊或附和：%r，继续汇报", text)
            return False
        if decision is None:
            logger.info("[打断回退] 判断不确定或服务不可用，接收有效转写：%r", text)
        else:
            logger.info("[打断判断] 对 Alpha 说话，接收")
        return True

    def _collection_user_state(self, event: UserStateChangedEvent) -> None:
        if event.new_state == "speaking":
            self.collection.speech_started()
        elif event.old_state == "speaking":
            self.collection.speech_stopped()

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        # 这里只登记，不等待；LiveKit 会等这个 hook 返回，才能安排下一轮。
        if self.collection_enabled:
            self.collection.add(new_message.text_content or "")

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings,
    ) -> AsyncIterator[stt.SpeechEvent]:
        async for event in Agent.default.stt_node(self, audio, model_settings):
            if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                # 中间结果可能从“咳”改成正常词，也可能反过来。它不参与打断或轮次提交。
                continue
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                text = event.alternatives[0].text if event.alternatives else ""
                self.interruptions.restore()
                if not is_meaningful(text):
                    self.interruptions.overlapped = False
                    self.interruptions.utterance_started_at = None
                    self.collection.transcript_resolved()
                    logger.info("[忽略] 非语言转写 %r，恢复音量，继续汇报", text)
                    continue  # 必须在进入 LiveKit 的取消逻辑之前丢弃。
                accepted = await self._accept_transcript(text)
                self.interruptions.overlapped = False
                self.interruptions.utterance_started_at = None
                self.interruptions.restore()
                if not accepted:
                    self.collection.transcript_resolved()
                    continue
                self._question_deadline = 0.0  # 回答机会只用于当前这一句。
                if self.session.agent_state in {"speaking", "thinking"}:
                    logger.info("[确认打断] 有效转写 %r，停止旧回答", text)
                # 状态可能仍显示 listening，但队列中已有回答；也要取消这些旧任务。
                await self.session.interrupt(force=True)
                self.collection.transcript_resolved()
            yield event

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings,
    ) -> AsyncIterator[rtc.AudioFrame]:
        iterator = text.__aiter__()
        try:
            first = await anext(iterator)
        except StopAsyncIteration:
            return

        async def head() -> AsyncIterator[str]:
            yield first

        if first.strip().rstrip("。") in {SEARCH_ACK.rstrip("。"), "好的，我查一查"}:
            pending_period = not first.strip().endswith("。")
            first = first.strip().rstrip("。") + "。"
            # 单句分词器默认会留住最后一句等后文；把固定提示作为独立、已结束的
            # TTS 输入。LiveKit 的 Markdown 过滤器还可能单独扣住句尾句号，
            # 所以不能只匹配带句号的完整原文。
            async for frame in Agent.default.tts_node(self, head(), model_settings):
                yield frame

            async def after_ack() -> AsyncIterator[str]:
                nonlocal pending_period
                async for part in iterator:
                    if pending_period and part:
                        part = part.removeprefix("。")
                        pending_period = False
                    if part:
                        yield part
            rest = after_ack()
        else:
            async def combined() -> AsyncIterator[str]:
                yield first
                async for part in iterator:
                    yield part
            rest = combined()
        async for frame in Agent.default.tts_node(self, rest, model_settings):
            yield frame

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterator[str]:
        # 这是框架生成回答的入口。这里不调用 OpenAI LLM，也不在语音层执行工具。
        text = last_user_text(chat_ctx)
        if not text:
            return
        self._reply_text = ""
        self._question_deadline = 0.0
        if self.collection_enabled:
            if not self.collection.parts:
                # console 文字输入/直接 generate_reply 不经过语音 turn-completed hook。
                self.collection.add(text)
            text = await self.collection.take()
            if not text:
                if self.collection.prompt:
                    prompt, self.collection.prompt = self.collection.prompt, None
                    yield prompt  # 本地生成追问，走现有 TTS，不请求大脑或执行工具。
                return
        # 固定进度提示先送入 LiveKit 的 TTS，再请求大脑；不是模型生成的答案。
        if self.search_router_mode == "jev":
            recent = [{"role": item.role, "content": item.text_content or ""}
                      for item in chat_ctx.items if isinstance(item, llm.ChatMessage)
                      and item.role in {"user", "assistant"}]
            if recent and recent[-1]["role"] == "user":
                recent.pop()  # 最新请求已在 request 字段里，不重复传一份。
            announced = await self.search_router.classify(text, recent) is True
            # 低置信度、超时、失败不再套关键词规则，直接把原请求交给 Claude。
        elif self.search_router_mode == "rules":
            announced = explicit_web_request(text, web_context=self._web_context)
        else:
            announced = False
        logger.info("[搜索判断] mode=%s 提前提示=%s ｜ 上轮搜索=%s",
                    self.search_router_mode, announced, self._web_context)
        self._web_context = announced  # 闲聊或本地任务会结束搜索追问上下文。
        payload: dict[str, str | bool] = {"text": text}
        if announced:
            logger.info("[搜索提示] 先输出提示文字，TTS 与查询并行")
            # tts_node 将固定提示独立合成，并按顺序播放提示和答案；
            # 此处不等待 TTS 首包或 speaking 状态，避免串行增加查询延迟。
            yield SEARCH_ACK
            payload["search_announced"] = True
        # 窗口只能阻止尚未提交的请求；发出 POST 后不能撤回已执行的操作。
        started = time.perf_counter()
        answer_parts: list[str] = []
        completed = False
        try:
            async with self.http_client.stream("POST", BRAIN_URL, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("大脑流式事件必须是对象")
                    if event.get("type") == "delta":
                        delta = event.get("text")
                        if not isinstance(delta, str):
                            raise ValueError("文字增量必须是字符串")
                        if delta:
                            if delta.startswith(SEARCH_ACK):
                                self._web_context = True  # 未提前识别、由大脑发出的实际搜索提示。
                            if not answer_parts:
                                logger.info("[大脑] 首段文字 %.2fs", time.perf_counter() - started)
                            answer_parts.append(delta)
                            self._reply_text = "".join(answer_parts)
                            yield delta  # LiveKit 按句送给 TTS，不再等待全部回答。
                    elif event.get("type") == "signal":
                        if event.get("name") != "ha_done":
                            raise ValueError("未知的提示音信号")
                        # 不 await：提示音排在当前生成之后播放，等待它会阻塞当前回答。
                        self.session.say("", audio=confirmation_tone(), add_to_chat_ctx=False)
                        logger.info("[HA] 调用成功，播放提示音，不朗读结果")
                    elif event.get("type") == "done":
                        completed = True
                        break
                    elif event.get("type") == "error":
                        raise ValueError(str(event.get("text", "大脑处理失败")))
                    else:
                        raise ValueError("未知的流式事件")
            if not completed:
                raise ValueError("大脑连接在完成之前中断")
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            # 请求可能已经执行了工具；不能自动重发 POST，避免重复创建提醒等内容。
            logger.exception("[大脑] 请求失败；没有自动重试")
            yield "这次没能拿到完整结果，操作可能已经生效，请先查询确认。"
            return
        logger.info("[大脑] 完整回答 %.2fs ｜ Alpha：%s", time.perf_counter() - started, "".join(answer_parts))
        self._web_context = self._web_context or announced


def log_conversation(event: ConversationItemAddedEvent) -> None:
    item = event.item
    if not isinstance(item, llm.ChatMessage) or item.role != "assistant":
        return
    # LiveKit 用 VAD 回推的停说时间和首帧播放时间计算此值，包含轮次检测、
    # STT、大脑、TTS 及播放等待。它不是麦克风再次测到扬声器发声的声学测量。
    latency = item.metrics.get("e2e_latency")
    if latency is not None:
        logger.info("[延迟] 说完 → 首帧播放：%.2fs%s", latency,
                    "（回答被打断）" if item.interrupted else "")


def log_transcript(event: UserInputTranscribedEvent) -> None:
    if event.is_final:
        logger.info("你：%s", event.transcript)


def log_metrics(event: MetricsCollectedEvent) -> None:
    metric = event.metrics
    if metric.type == "eou_metrics":
        logger.info("[分段耗时] 停说→转写 %.2fs ｜ 停说→轮次确认 %.2fs",
                    metric.transcription_delay, metric.end_of_utterance_delay)
    elif metric.type == "tts_metrics":
        logger.info("[分段耗时] TTS 首个音频包 %.2fs", metric.ttfb)


def create_session(*, jev_router: JevSearchRouter | None = None) -> AgentSession:
    turn_mode = os.getenv("ALPHA_TURN_DETECTOR", "livekit").strip().lower()
    if turn_mode not in {"livekit", "jev"}:
        raise ValueError("ALPHA_TURN_DETECTOR 必须是 livekit 或 jev")
    detector: MultilingualModel | JevTurnDetector = MultilingualModel()
    if turn_mode == "jev":
        if jev_router is None or not jev_router.api_key:
            raise ValueError("Jev 轮次检测需要已配置的 Jev API key")
        detector = JevTurnDetector(jev_router, fallback=detector)
    max_delay = float(os.getenv("ALPHA_TURN_MAX_DELAY", "5" if turn_mode == "jev" else "3"))
    if not math.isfinite(max_delay) or max_delay < 0.5:
        raise ValueError("ALPHA_TURN_MAX_DELAY 必须是不小于 0.5 的有限数字")
    return AgentSession(
        vad=silero.VAD.load(min_speech_duration=0.1),
        stt=openai.STT(
            model="gpt-4o-mini-transcribe", language="zh",
            use_realtime=os.getenv("ALPHA_STT_REALTIME", "1") != "0",
            turn_detection={"type": "server_vad", "threshold": 0.5,
                            "prefix_padding_ms": 300, "silence_duration_ms": 350},
        ),
        llm=None,
        tts=tts.StreamAdapter(tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice=os.getenv("ALPHA_TTS_VOICE", "coral"),
            speed=float(os.getenv("ALPHA_TTS_SPEED", "1.15")),
            response_format="pcm",  # 省去默认 MP3 的解码和压缩帧缓冲。
            instructions="用自然、清晰的普通话朗读，语气轻松。只朗读原文，不增加内容。",
        ), sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(
            # 默认最短句子 20 字、上下文 10 字，会把短中文句子继续攒着。
            min_sentence_len=4, stream_context_len=2, retain_format=True,
        )),
        turn_handling={
            "turn_detection": detector,
            "endpointing": {"min_delay": 0.5, "max_delay": max_delay},
            # 大脑的请求会执行真实操作，绝不能在用户轮次确认之前提前生成。
            "preemptive_generation": {"enabled": False},
            "interruption": {
                "mode": "vad",
                # 声音只降音量，不让框架提前取消。stt_node 确认有效文字后手动打断。
                "enabled": False,
                "discard_audio_if_uninterruptible": False,
            },
        },
    )


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("请在 voice_pipeline/.env 或 alpha/.env 设置 OPENAI_API_KEY")

    # trust_env=False：本机大脑请求不经过系统 HTTP 代理。客户端随任务退出关闭，
    # 不能用一个在 session.start() 返回后立刻关闭的 async with 包住它。
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0), trust_env=False)
    ctx.add_shutdown_callback(http_client.aclose)
    health = await http_client.get("http://127.0.0.1:8765/health", timeout=5.0)
    health.raise_for_status()

    agent = AlphaVoiceAgent(http_client)
    session = create_session(jev_router=agent.search_router)
    session.on("user_input_transcribed", log_transcript)
    session.on("conversation_item_added", log_conversation)
    session.on("metrics_collected", log_metrics)
    await session.start(agent=agent, room=ctx.room)
    logger.info("[就绪] 中文语音 ｜ Silero + %s 轮次检测 ｜ OpenAI STT/TTS ｜ AlphaBrain",
                os.getenv("ALPHA_TURN_DETECTOR", "livekit"))


if __name__ == "__main__":
    cli.run_app(server)
