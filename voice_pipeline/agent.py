"""本机语音入口：LiveKit 负责听和说，brain.server 负责理解和执行工具。

运行：.venv/bin/python agent.py console
首次运行先下载轮次检测模型：.venv/bin/python agent.py download-files
API 按 LiveKit Agents 1.8.8 编写。
"""

from __future__ import annotations

import logging
import json
import os
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
)
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

if __package__:
    from .interruptions import SpeechGate, is_meaningful
    from .sounds import confirmation_tone
else:
    from interruptions import SpeechGate, is_meaningful
    from sounds import confirmation_tone


DIRECTORY = Path(__file__).resolve().parent
# 独立的语音配置优先；没有配置的项沿用 Alpha 的 .env，不依赖启动目录。
load_dotenv(DIRECTORY / ".env")
load_dotenv(DIRECTORY.parent / ".env")

logger = logging.getLogger("alpha.voice_pipeline")
BRAIN_URL = "http://127.0.0.1:8765/ask/stream"
server = AgentServer()


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
    def __init__(self, http_client: httpx.AsyncClient) -> None:
        super().__init__(
            instructions="你是 Alpha 的语音接口。原样朗读大脑返回的回答。",
            llm=BrainNodeLLM(),
        )
        self.http_client = http_client
        self.interruptions = SpeechGate()

    async def on_enter(self) -> None:
        self.interruptions.attach(self.session)

    async def on_exit(self) -> None:
        self.interruptions.close()

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
                    logger.info("[忽略] 非语言转写 %r，恢复音量，继续汇报", text)
                    continue  # 必须在进入 LiveKit 的取消逻辑之前丢弃。
                if self.session.agent_state in {"speaking", "thinking"}:
                    logger.info("[确认打断] 有效转写 %r，停止旧回答", text)
                # 状态可能仍显示 listening，但队列中已有回答；也要取消这些旧任务。
                await self.session.interrupt(force=True)
            yield event

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
        started = time.perf_counter()
        answer_parts: list[str] = []
        completed = False
        try:
            async with self.http_client.stream("POST", BRAIN_URL, json={"text": text}) as response:
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
                            if not answer_parts:
                                logger.info("[大脑] 首段文字 %.2fs", time.perf_counter() - started)
                            answer_parts.append(delta)
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


def create_session() -> AgentSession:
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
            "turn_detection": MultilingualModel(),
            "endpointing": {"min_delay": 0.5, "max_delay": 3.0},
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

    session = create_session()
    session.on("user_input_transcribed", log_transcript)
    session.on("conversation_item_added", log_conversation)
    session.on("metrics_collected", log_metrics)
    await session.start(agent=AlphaVoiceAgent(http_client), room=ctx.room)
    logger.info("[就绪] 中文语音 ｜ Silero + 多语言轮次检测 ｜ OpenAI STT/TTS ｜ AlphaBrain")


if __name__ == "__main__":
    cli.run_app(server)
