"""启动器就绪顺序和进程归属测试，不启动真实服务或麦克风。"""
import asyncio
import signal
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from apps import run_qwen as launcher


class Process:
    def __init__(self, code=None):
        self.returncode = code
        self.signals = []
        self.finished = asyncio.Event()
        if code is not None:
            self.finished.set()

    async def wait(self):
        await self.finished.wait()
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        self.returncode = 0
        self.finished.set()


class LauncherTests(IsolatedAsyncioTestCase):
    async def test_waits_for_brain_before_voice_and_ctrl_c_stops_both_owned_processes(self):
        brain, voice = Process(), Process()
        stopping = asyncio.Event()
        order = []
        async def spawn(module, env):
            order.append(module)
            if module == 'voice.qwen_alpha':
                stopping.set()
                return voice
            return brain
        health = AsyncMock(side_effect=[False, False, True])
        with patch.object(launcher, 'healthy', health), patch.object(launcher, 'spawn', side_effect=spawn):
            result = await launcher.run(None, url=launcher.DEFAULT_URL, env={}, stopping=stopping)
        self.assertEqual(result, 0)
        self.assertEqual(order, ['brain.server', 'voice.qwen_alpha'])
        self.assertEqual(health.await_count, 3)
        self.assertEqual(brain.signals, [signal.SIGINT])
        self.assertEqual(voice.signals, [signal.SIGINT])

    async def test_existing_brain_is_reused_and_never_stopped(self):
        voice = Process(0)
        spawn = AsyncMock(return_value=voice)
        with patch.object(launcher, 'healthy', AsyncMock(return_value=True)), patch.object(launcher, 'spawn', spawn):
            result = await launcher.run(None, url=launcher.DEFAULT_URL, env={}, stopping=asyncio.Event())
        self.assertEqual(result, 0)
        spawn.assert_awaited_once_with('voice.qwen_alpha', {})

    async def test_brain_startup_failure_never_starts_voice(self):
        spawn = AsyncMock(return_value=Process(2))
        with patch.object(launcher, 'healthy', AsyncMock(return_value=False)), patch.object(launcher, 'spawn', spawn):
            with self.assertRaisesRegex(RuntimeError, '启动失败'):
                await launcher.run(None, url=launcher.DEFAULT_URL, env={}, stopping=asyncio.Event())
        spawn.assert_awaited_once_with('brain.server', {})

    async def test_startup_timeout_cleans_owned_brain(self):
        brain = Process()
        with patch.object(launcher, 'healthy', AsyncMock(return_value=False)), \
                patch.object(launcher, 'spawn', AsyncMock(return_value=brain)):
            with self.assertRaisesRegex(RuntimeError, '超时'):
                await launcher.run(None, url=launcher.DEFAULT_URL, env={}, stopping=asyncio.Event(), startup_timeout=0)
        self.assertEqual(brain.signals, [signal.SIGINT])

    async def test_brain_crash_stops_voice(self):
        brain, voice = Process(1), Process()
        with patch.object(launcher, 'healthy', AsyncMock(side_effect=[False, True])), \
                patch.object(launcher, 'spawn', AsyncMock(side_effect=[brain, voice])):
            with self.assertRaisesRegex(RuntimeError, '已退出'):
                await launcher.run(None, url=launcher.DEFAULT_URL, env={}, stopping=asyncio.Event())
        self.assertEqual(voice.signals, [signal.SIGINT])
