"""一键启动本机 AlphaBrain 和 Qwen，等待健康检查，并负责清理自己启动的进程。"""
import argparse
import asyncio
import os
import signal
import time
from urllib.parse import urlsplit

import httpx2
from dotenv import load_dotenv

from core.paths import ENV_FILE, PROJECT_ROOT, VENV_PYTHON

DEFAULT_URL = 'http://127.0.0.1:8765'


async def healthy(client, url):
    try:
        response = await client.get(url + '/health', timeout=.7)
        return response.status_code == 200 and response.json().get('ok') is True
    except (httpx2.HTTPError, ValueError, TypeError, AttributeError):
        return False


async def spawn(module, env):
    return await asyncio.create_subprocess_exec(
        str(VENV_PYTHON), '-m', module, cwd=str(PROJECT_ROOT), env=env,
        start_new_session=True,
    )


async def stop_process(process):
    if process is None or process.returncode is not None:
        return
    # 第一次只通知父进程，让大脑正常关闭 worker 并保存操作状态。
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 15)
        return
    except TimeoutError:
        print('[启动器] 清理仍未结束，停止该进程组。', flush=True)
    for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 3)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout)
            return
        except TimeoutError:
            pass


async def run(client, *, url, env, stopping, startup_timeout=60):
    brain = voice = None
    monitors = []
    try:
        if await healthy(client, url):
            print('[启动器] AlphaBrain 已在运行，复用现有服务；退出时不会关闭它。', flush=True)
        else:
            parsed = urlsplit(url)
            if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'}
                    or parsed.port != 8765 or parsed.path not in {'', '/'}):
                raise RuntimeError('配置的大脑地址不可用；自动启动只支持 http://127.0.0.1:8765')
            print('[启动器] 正在启动 AlphaBrain…', flush=True)
            brain = await spawn('brain.server', env)
            deadline = time.monotonic() + startup_timeout
            while not await healthy(client, url):
                if stopping.is_set():
                    return 0
                if brain.returncode is not None:
                    raise RuntimeError(f'AlphaBrain 启动失败，退出码 {brain.returncode}，请查看上面的日志')
                if time.monotonic() >= deadline:
                    raise RuntimeError('等待 AlphaBrain 就绪超时，请查看上面的日志')
                try:
                    await asyncio.wait_for(stopping.wait(), .2)
                except TimeoutError:
                    pass
        if stopping.is_set():
            return 0
        print('[启动器] 大脑已就绪，启动 Qwen；按一次 Ctrl+C 退出。', flush=True)
        voice = await spawn('voice.qwen_alpha', env)
        stopped = asyncio.create_task(stopping.wait())
        voice_exit = asyncio.create_task(voice.wait())
        monitors = [stopped, voice_exit]
        brain_exit = asyncio.create_task(brain.wait()) if brain is not None else None
        if brain_exit is not None:
            monitors.append(brain_exit)
        done, _ = await asyncio.wait(monitors, return_when=asyncio.FIRST_COMPLETED)
        if stopped in done:
            return 0
        if brain_exit is not None and brain_exit in done:
            raise RuntimeError(f'AlphaBrain 已退出（{brain.returncode}），同时关闭语音进程')
        return voice_exit.result()
    finally:
        for task in monitors:
            task.cancel()
        await asyncio.gather(*monitors, return_exceptions=True)
        if brain is not None or voice is not None:
            print('[启动器] 正在关闭本次启动的进程，请稍等…', flush=True)
        await stop_process(voice)
        await stop_process(brain)


async def main():
    parser = argparse.ArgumentParser(description='同时启动 AlphaBrain 和 Qwen 语音；Ctrl+C 一起退出')
    parser.add_argument('--startup-timeout', type=float, default=60, help='大脑就绪超时秒数，默认 60')
    args = parser.parse_args()
    if not 0 < args.startup_timeout < float('inf'):
        parser.error('--startup-timeout 必须为正的有限数字')
    load_dotenv(ENV_FILE)
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    url = env.get('ALPHA_BRAIN_URL', DEFAULT_URL).rstrip('/')
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopping.set)
    try:
        async with httpx2.AsyncClient() as client:
            return await run(client, url=url, env=env, stopping=stopping,
                             startup_timeout=args.startup_timeout)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'[启动器] {exc}', flush=True)
        return 1
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
