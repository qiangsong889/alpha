"""实时把 ESPHome 卫星的日志逐行写进文件（不经过 esphome CLI 的管道缓冲；断线自动重连）。

在 alpha 目录运行：.venv/bin/python -m devtools.satellite_logs [秒数，默认 300] [输出文件，默认 data/satellite_device.log]
读取 .env 的 ALPHA_SATELLITE_HOST（默认 192.168.1.115）和 ALPHA_SATELLITE_KEY。
"""
import asyncio
import os
import re
import sys
import time

from aioesphomeapi import APIClient, LogLevel
from dotenv import load_dotenv

from core.paths import ENV_FILE

KEEP = re.compile(r'micro_wake_word|voice_assistant|唤醒|诊断|\[E\]|\[W\]')
ANSI = re.compile(r'\x1b\[[0-9;]*m')


async def main():
    load_dotenv(ENV_FILE)
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    path = sys.argv[2] if len(sys.argv) > 2 else 'data/satellite_device.log'
    host = os.environ.get('ALPHA_SATELLITE_HOST') or '192.168.1.115'
    deadline = time.monotonic() + seconds
    with open(path, 'w', buffering=1) as out:
        while time.monotonic() < deadline:
            client = APIClient(host, 6053, None, client_info='alpha-logs',
                               noise_psk=os.environ.get('ALPHA_SATELLITE_KEY') or None)
            stopped = asyncio.Event()

            async def on_stop(expected):
                stopped.set()

            def on_log(msg):
                line = ANSI.sub('', msg.message.decode(errors='replace'))
                if KEEP.search(line) and 'waiting for client' not in line:
                    out.write(time.strftime('%H:%M:%S ') + line + '\n')

            try:
                await client.connect(on_stop=on_stop, login=True)
                client.subscribe_logs(on_log, log_level=LogLevel.LOG_LEVEL_DEBUG)
                out.write(time.strftime('%H:%M:%S ') + '--- 已连接 ---\n')
                try:
                    async with asyncio.timeout(max(0.1, deadline - time.monotonic())):
                        await stopped.wait()
                except TimeoutError:
                    pass
            except Exception as exc:
                out.write(time.strftime('%H:%M:%S ') + f'--- 连接失败：{type(exc).__name__} ---\n')
                await asyncio.sleep(2)
            finally:
                await client.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
