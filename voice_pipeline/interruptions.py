"""声音只触发降音量，最终转写决定是否打断；不调用另一个模型。"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from array import array
from collections.abc import Callable

from livekit import rtc
from livekit.agents import AgentSession
from livekit.agents.voice import io
from livekit.agents.voice.events import UserStateChangedEvent

logger = logging.getLogger("alpha.voice_pipeline")


def is_meaningful(text: str) -> bool:
    """只拒绝空内容和明确的非语言标记；不按字数丢弃正常短句。"""
    normalized = "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                         if not c.isspace() and not unicodedata.category(c).startswith(("P", "S")))
    noises = {"咳嗽", "咳嗽声", "咳嗽聲", "笑声", "笑聲", "呼吸声", "呼吸聲",
              "噪音", "背景噪音", "静音", "靜音", "听不清", "聽不清",
              "cough", "coughing", "laughter", "laughing", "breathing", "noise",
              "backgroundnoise", "silence", "inaudible"}
    return bool(normalized) and normalized not in noises and re.fullmatch("咳+", normalized) is None


class DuckingOutput(io.AudioOutput):
    """在实际输出链上缩放 PCM，保留时间戳、缓冲区和播放完成事件。"""

    def __init__(self, sink: io.AudioOutput, gain: Callable[[], float]):
        super().__init__(label="alpha-ducking", next_in_chain=sink,
                         sample_rate=sink.sample_rate,
                         capabilities=io.AudioOutputCapabilities(pause=sink.can_pause))
        self._gain = gain

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        gain = self._gain()
        if gain != 1.0:
            # AudioFrame.data 是 int16 memoryview；不修改原音频，避免污染重用的帧。
            samples = array("h", (int(sample * gain) for sample in frame.data))
            frame = rtc.AudioFrame(data=samples.tobytes(), sample_rate=frame.sample_rate,
                                   num_channels=frame.num_channels,
                                   samples_per_channel=frame.samples_per_channel)
        assert self.next_in_chain is not None
        await self.next_in_chain.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        assert self.next_in_chain is not None
        self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        assert self.next_in_chain is not None
        self.next_in_chain.clear_buffer()


class SpeechGate:
    def __init__(self, *, timeout_s: float = 1.0):
        self.gain = 1.0
        self.timeout_s = timeout_s
        self._timer: asyncio.TimerHandle | None = None
        self._session: AgentSession | None = None
        self._output: DuckingOutput | None = None
        self._original: io.AudioOutput | None = None

    def attach(self, session: AgentSession) -> None:
        self._session = session
        self._original = session.output.audio
        if self._original is not None:
            self._output = DuckingOutput(self._original, lambda: self.gain)
            session.output.audio = self._output
        session.on("user_state_changed", self._user_state_changed)

    def _user_state_changed(self, event: UserStateChangedEvent) -> None:
        if event.new_state == "speaking":
            if self._session is not None and self._session.agent_state == "speaking":
                self.speech_started()
        elif event.old_state == "speaking":
            self.speech_stopped()

    def speech_started(self) -> None:
        self._cancel_timer()
        self.gain = 0.2
        logger.info("[疑似打断] 降低音量到 20%，保留汇报，等待最终转写")

    def speech_stopped(self) -> None:
        if self.gain < 1.0:
            self._cancel_timer()
            self._timer = asyncio.get_running_loop().call_later(self.timeout_s, self._timed_out)

    def _timed_out(self) -> None:
        self.restore()
        logger.info("[恢复] 未收到有效转写，恢复音量，继续汇报")

    def restore(self) -> None:
        self._cancel_timer()
        self.gain = 1.0

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def close(self) -> None:
        self.restore()
        if self._session is not None:
            self._session.off("user_state_changed", self._user_state_changed)
            if self._output is not None and self._session.output.audio is self._output:
                self._session.output.audio = self._original
        self._session = None
