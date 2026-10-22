"""启动控制台：python -m apps.console [--port 8770]

只监听 127.0.0.1；每次启动生成新的令牌，终端打印带令牌的链接。控制台只读，不调用任何模型。
"""

from __future__ import annotations

import argparse
import secrets
import sys

import httpx2
import uvicorn

from console.data import ConsoleData
from console.server import create_app
from core.paths import ARCHIVE_DB, COSTS_DB, ENV_FILE, OPERATIONS_DB, PROJECT_ROOT
from core.runtime import VOICE_STATUS, ProcessStatus

HOST = '127.0.0.1'
DEFAULT_PORT = 8770


async def fetch_brain_status(url: str) -> ProcessStatus:
    """大脑的 /status。连不上不等于"已停止"：也可能卡住了，所以叫 unreachable。"""
    try:
        async with httpx2.AsyncClient() as client:
            response = await client.get(url.rstrip('/') + '/status', timeout=1.0)
            response.raise_for_status()
            data = response.json()
    except (httpx2.HTTPError, ValueError) as exc:
        return ProcessStatus('unreachable', detail=f'连不上大脑（{type(exc).__name__}）：没有启动，或者卡住了')
    if not isinstance(data, dict):
        return ProcessStatus('unreadable', detail='大脑返回的状态格式不认识')
    state = data.get('state')
    if state == 'running':
        return ProcessStatus('running', data, 0.0)
    return ProcessStatus(state if state == 'starting' else 'unreadable', data,
                         detail='大脑正在启动' if state == 'starting' else '大脑返回的状态格式不认识')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Alpha 控制台（只读）')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    token = secrets.token_urlsafe(32)
    data = ConsoleData(env_file=ENV_FILE, archive_db=ARCHIVE_DB, costs_db=COSTS_DB,
                       operations_db=OPERATIONS_DB, voice_status=VOICE_STATUS)
    app = create_app(token=token, data=data, brain_status=fetch_brain_status,
                     web_dir=PROJECT_ROOT / 'console' / 'web')
    print(f'\nAlpha 控制台：http://{HOST}:{args.port}/login?token={token}\n'
          '（令牌每次启动都会重新生成；不要把这个链接发给别人）\n', file=sys.stderr, flush=True)
    uvicorn.run(app, host=HOST, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
