"""跟随 macOS 系统的默认音频设备。

PortAudio（sounddevice）只在初始化时读取一次设备列表，打开的流也一直绑定在当时的设备上：
在系统里切换输出、连上 AirPods 之后，Alpha 仍然用旧设备。要看到新设备必须重新初始化 PortAudio，
而重新初始化会关闭进程里所有的流；在流还开着时做这件事可能让进程崩溃。所以由 AudioRoute 协调：

1. watch() 每秒直接问 CoreAudio 当前的默认输入/输出设备（不经过 PortAudio）；
2. 发现变化 → changed 置位，扬声器和麦克风各自正常关闭自己的流；
3. 所有流都关闭后重新初始化 PortAudio，再让它们用新的默认设备重新打开。

.env 里用 REALTIME_INPUT_DEVICE / REALTIME_OUTPUT_DEVICE 指定了固定设备时，那一边不跟随系统。
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import ctypes
import ctypes.util
import logging
import sys
from typing import Callable, Iterator

logger = logging.getLogger('alpha.audio_devices')

CLOSE_TIMEOUT_S = 3.0


class _PropertyAddress(ctypes.Structure):
    _fields_ = [('selector', ctypes.c_uint32), ('scope', ctypes.c_uint32), ('element', ctypes.c_uint32)]


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode('ascii'), 'big')


class CoreAudioDefaults:
    """读取 macOS 当前的默认输入/输出设备名称；不是 macOS 或读取失败时返回 None。"""

    SYSTEM_OBJECT = 1

    def __init__(self) -> None:
        self._audio = self._foundation = None
        if sys.platform != 'darwin':
            return
        try:
            self._audio = ctypes.CDLL(ctypes.util.find_library('CoreAudio'))
            self._foundation = ctypes.CDLL(ctypes.util.find_library('CoreFoundation'))
            self._foundation.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                            ctypes.c_long, ctypes.c_uint32]
            self._foundation.CFRelease.argtypes = [ctypes.c_void_p]
        except (OSError, TypeError, AttributeError) as exc:
            logger.warning('[音频设备] 无法读取系统默认设备：%s', type(exc).__name__)
            self._audio = self._foundation = None

    def _get(self, target: int, selector: str, value: ctypes._SimpleCData) -> bool:
        address = _PropertyAddress(_fourcc(selector), _fourcc('glob'), 0)
        size = ctypes.c_uint32(ctypes.sizeof(value))
        return self._audio.AudioObjectGetPropertyData(  # type: ignore[union-attr]
            ctypes.c_uint32(target), ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(value)) == 0

    def _device_name(self, selector: str) -> str | None:
        device = ctypes.c_uint32()
        if not self._get(self.SYSTEM_OBJECT, selector, device) or device.value == 0:
            return None
        name = ctypes.c_void_p()
        if not self._get(device.value, 'lnam', name) or not name.value:
            return None
        buffer = ctypes.create_string_buffer(512)
        try:
            ok = self._foundation.CFStringGetCString(name, buffer, len(buffer), 0x08000100)  # type: ignore[union-attr]
        finally:
            self._foundation.CFRelease(name)  # type: ignore[union-attr]
        return buffer.value.decode('utf-8', 'replace') if ok else None

    def __call__(self) -> tuple[str | None, str | None]:
        if self._audio is None:
            return None, None
        return self._device_name('dIn '), self._device_name('dOut')


def reinitialize_portaudio() -> None:
    """重新读取设备列表。调用时不能有任何打开的 PortAudio 流。"""
    import sounddevice as sd  # type: ignore[import-untyped]
    sd._terminate()
    sd._initialize()


class AudioRoute:
    """同一进程里所有 PortAudio 流的协调者。

    流的使用方：打开前 await wait_ready()；打开期间放在 with route.stream() 里；
    循环中看到 route.changed 置位就正常关闭流，然后回到开头重新打开。
    """

    def __init__(self, *, follow_input: bool, follow_output: bool,
                 probe: Callable[[], tuple[str | None, str | None]] | None = None,
                 reinitialize: Callable[[], None] = reinitialize_portaudio,
                 interval_s: float = 1.0, close_timeout_s: float = CLOSE_TIMEOUT_S):
        self.follow_input, self.follow_output = follow_input, follow_output
        self._probe = probe
        self.reinitialize = reinitialize
        self.interval_s, self.close_timeout_s = interval_s, close_timeout_s
        self.changed = asyncio.Event()      # 置位：所有流都应关闭
        self._idle = asyncio.Event()        # 置位：当前没有打开的流
        self._idle.set()
        self._open = 0

    def probe(self) -> tuple[str | None, str | None]:
        if self._probe is None:
            self._probe = CoreAudioDefaults()  # 只在真正跟随时才加载 CoreAudio
        return self._probe()

    @property
    def enabled(self) -> bool:
        return self.follow_input or self.follow_output

    async def wait_ready(self) -> None:
        """切换进行中时等它完成，避免在重新初始化前后打开流。"""
        while self.changed.is_set():
            await asyncio.sleep(0.05)

    @contextmanager
    def stream(self) -> Iterator[None]:
        self._open += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._open -= 1
            if self._open == 0:
                self._idle.set()

    def _followed(self, devices: tuple[str | None, str | None]) -> tuple[str | None, str | None]:
        input_name, output_name = devices
        return (input_name if self.follow_input else None, output_name if self.follow_output else None)

    async def switch(self, reason: str) -> bool:
        """让所有流关闭 → 重新初始化 PortAudio → 允许重新打开。返回是否完成了重新初始化。"""
        self.changed.set()
        try:
            try:
                await asyncio.wait_for(self._idle.wait(), self.close_timeout_s)
            except TimeoutError:
                # 有流没有配合关闭（例如旧的 Omni 收音）：流还开着时重新初始化可能崩溃，这次放弃。
                logger.warning('[音频设备] %s，但仍有 %d 个流没有关闭，这次不切换', reason, self._open)
                return False
            self.reinitialize()
            logger.info('[音频设备] %s，已切换', reason)
            return True
        finally:
            self.changed.clear()

    async def watch(self) -> None:
        if not self.enabled:
            return
        # 在线程里读：第一次调用要初始化 CoreAudio（约 50ms），系统切换设备时也可能变慢，不能卡住事件循环。
        last = self._followed(await asyncio.to_thread(self.probe))
        while True:
            await asyncio.sleep(self.interval_s)
            current = self._followed(await asyncio.to_thread(self.probe))
            if current == last or None in [name for name, followed in
                                           zip(current, (self.follow_input, self.follow_output)) if followed]:
                # 没变，或者正在切换中暂时读不到默认设备：下一秒再看。
                continue
            labels = [f'{kind} {old} → {new}' for kind, old, new in
                      zip(('输入', '输出'), last, current) if old != new]
            # 放弃的切换也不再重试：否则配合关闭的流会每秒被关掉、重开一次。
            await self.switch('系统默认设备变化：' + '；'.join(labels))
            last = current
