"""本地提示音，直接走 LiveKit 播放队列，不调用 TTS。"""

import math
from array import array
from collections.abc import AsyncIterator

from livekit import rtc


async def confirmation_tone() -> AsyncIterator[rtc.AudioFrame]:
    rate, total, block = 24000, 2880, 480  # 120 ms，分成 20 ms 的帧。
    fade = rate // 100  # 淡入淡出，避免音头爆音。
    for start in range(0, total, block):
        samples = array("h", (
            int(0.15 * 32767 * min(1.0, index / fade, (total - 1 - index) / fade)
                * math.sin(2 * math.pi * 880 * index / rate))
            for index in range(start, min(start + block, total))
        ))
        yield rtc.AudioFrame(data=samples.tobytes(), sample_rate=rate,
                             num_channels=1, samples_per_channel=len(samples))
