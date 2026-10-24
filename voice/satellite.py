"""ESPHome 语音卫星（J9）：设备本地听唤醒词，唤醒后通过 API 连接把音频推给 Alpha。

设备（esp-alpha/esphome/alpha-xvf.yaml）听到唤醒词发 start 请求；这里回端口 0（音频走 API 连接），
设备就开始推 16 kHz、16 位、单声道 PCM，直接喂进现有的 AsrInput.process（Silero 断句 → ASR → Jev → 大脑）。
一轮在这些时候结束，结束时发 RUN_END，设备停麦、回到空闲并重新听唤醒词：
- 这句话提交给大脑（Alpha 播报时设备不推流，听不到自己的声音）；
- 唤醒后一直没开口，或说完后一直没有提交（例如转写为空），超过 ALPHA_SATELLITE_LISTEN_SECONDS 没有人声；
- 设备自己要求停止。
播报中再次唤醒等于打断：先停掉当前播报。

API 连接跨 ASR 重连保留：run() 由外层常驻，feed() 每条 ASR 连接调用一次。
"""

import asyncio
import logging
import time

from aioesphomeapi import APIClient, VoiceAssistantEventType

from voice.segmenter import RATE

logger = logging.getLogger('alpha.qwen_alpha')

PORT = 6053
# 一轮结束时补的静音：断句器要 600ms 静音才会结束一段，被设备中途停掉的话也能收尾，不留半截。
FLUSH = bytes(int(RATE * 0.7) * 2)


class _RunEnded:
    """队列里的分隔符：前面的音频属于已经结束的那一轮。"""


class SatelliteLink:
    def __init__(self, host, *, port=PORT, noise_psk=None, listen_s=8.0, client_factory=APIClient,
                 clock=time.monotonic):
        self.host, self.port, self.noise_psk = host, port, noise_psk
        self.listen_s = listen_s
        self.client_factory, self.clock = client_factory, clock
        self.client = None
        self.queue = asyncio.Queue()
        self.active = False
        self.started_at = self.last_voice = 0.0
        self.receiver = None  # feed() 设置；唤醒时用来打断正在进行的播报

    async def run(self):
        """保持和设备的 API 连接，断线 5 秒后重连；连接期间订阅语音助手。"""
        while True:
            stopped = asyncio.Event()

            async def on_stop(expected):
                stopped.set()

            client = self.client_factory(self.host, self.port, None, client_info='alpha',
                                         noise_psk=self.noise_psk)
            try:
                await client.connect(on_stop=on_stop, login=True)
                info = await client.device_info()
                self.client = client
                unsubscribe = client.subscribe_voice_assistant(
                    handle_start=self.handle_start, handle_stop=self.handle_stop, handle_audio=self.handle_audio)
                logger.info('[卫星] 已连接 %s（%s，ESPHome %s）', info.name, self.host, info.esphome_version)
                try:
                    await stopped.wait()
                finally:
                    unsubscribe()
                logger.warning('[卫星] 连接断开，5 秒后重连')
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning('[卫星] 连接 %s 失败：%s；5 秒后重试', self.host, type(exc).__name__)
            finally:
                self.client = None
                self._finish('disconnected', notify=False)
                await client.disconnect()
            await asyncio.sleep(5)

    async def handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        logger.info('[卫星] 唤醒：%s', wake_word_phrase or '（无唤醒词）')
        receiver = self.receiver
        if receiver is not None and (receiver.active or receiver.current_report is not None):
            # 播报中再次唤醒就是要打断；设备推流期间 Alpha 不播报，所以只会发生在播报开始之后。
            logger.info('[卫星] 唤醒打断当前播报')
            await receiver.interrupt()
        if self.active:
            self._finish('restarted', notify=False)
        self.active = True
        self.started_at = self.last_voice = self.clock()
        self._event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START)
        self._event(VoiceAssistantEventType.VOICE_ASSISTANT_STT_START)
        return 0  # 0：音频走 API 连接，不另开 UDP

    async def handle_audio(self, data, data2=None):
        # 不能有 await 在 put 之前：每块音频是一个后台任务，入队顺序就是到达顺序。
        if self.active:
            self.queue.put_nowait(data)

    async def handle_stop(self, server_side):
        if self.active:
            self.end_run('device')

    def end_run(self, reason):
        """结束这一轮：通知设备停麦并恢复唤醒词检测。提交给大脑时由外层调用。"""
        if self.active:
            self._finish(reason)

    def _finish(self, reason, *, notify=True):
        if not self.active:
            return
        self.active = False
        self.queue.put_nowait(_RunEnded())
        if notify:
            self._event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END)
        logger.info('[卫星] 本轮结束：%s（%.1f 秒）', reason, self.clock() - self.started_at)

    def _event(self, kind):
        if self.client is not None:
            self.client.send_voice_assistant_event(kind, None)

    async def feed(self, adapter):
        """把设备推来的音频交给这条 ASR 连接的断句器；太久没有人声就结束这一轮。"""
        self.receiver = adapter.receiver
        try:
            while True:
                # 不用 wait_for：3.11 里取消恰好和 get() 拿到数据同时发生时，wait_for 会吞掉取消，
                # 这个循环就再也停不下来（单元测试里实际卡住过）。asyncio.timeout 能区分超时和取消。
                try:
                    async with asyncio.timeout(0.5):
                        item = await self.queue.get()
                except TimeoutError:
                    item = None
                if isinstance(item, _RunEnded):
                    await adapter.process(FLUSH)
                elif item is not None:
                    await adapter.process(item)
                    if adapter.segmenter.active is not None:
                        self.last_voice = self.clock()
                if self.active and self.clock() - self.last_voice >= self.listen_s:
                    self.end_run('timeout')
        finally:
            self.receiver = None
