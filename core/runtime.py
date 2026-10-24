"""运行状态：进程实际在用的配置、心跳，以及"已保存"和"正在生效"的比较。

- 大脑通过 HTTP `/status` 提供快照；语音没有 HTTP 服务，定时把快照写进 data/run/voice.json。
- 快照里只有非密钥参数的值；密钥只说明是否已设置。
- 读取方（控制台）必须允许"不知道"：没有文件是未启动，心跳太旧是状态过期，不能当成正在运行。
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Mapping

from core.paths import PROJECT_ROOT
from core.settings import REGISTRY, SettingsReport, check_settings

logger = logging.getLogger('alpha.runtime')

RUN_DIR = PROJECT_ROOT / 'data' / 'run'
VOICE_STATUS = RUN_DIR / 'voice.json'
VOICE_LOCK = RUN_DIR / 'voice.lock'
HEARTBEAT_S = 5.0
STALE_AFTER_S = 3 * HEARTBEAT_S
SCHEMA = 1


def effective_settings(report: SettingsReport) -> dict[str, Any]:
    """这个进程会读、并且启动时实际生效的参数；密钥只给出是否已设置。"""
    return {name: (bool(value) if REGISTRY[name].secret else value)
            for name, value in report.values.items() if report.process in REGISTRY[name].readers}


def snapshot(report: SettingsReport, *, started_at: float, state: str = 'running',
             now: float | None = None, **extra: Any) -> dict[str, Any]:
    now = time.time() if now is None else now
    return {
        'schema': SCHEMA, 'process': report.process, 'pid': os.getpid(), 'state': state,
        'started_at': started_at, 'heartbeat_at': now,
        'settings_fingerprint': report.fingerprint, 'settings': effective_settings(report),
        'warnings': list(report.warnings), **extra,
    }


def write_status(path: Path, data: Mapping[str, Any]) -> None:
    """先写临时文件再改名：读取方永远不会读到写了一半的 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, default=str)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


class AlreadyRunning(RuntimeError):
    """已经有另一个进程持有这把锁。"""


@contextmanager
def single_instance(path: Path) -> Iterator[None]:
    """同一时间只允许一个进程运行（例如语音层：两个进程会同时听、同时回答）。

    用 flock：进程不管正常退出还是被强制结束，操作系统都会释放锁，不会留下"死锁文件"。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, 'a+', encoding='utf-8')
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            raise AlreadyRunning(handle.read().strip() or '未知') from None
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        handle.close()


class StatusWriter:
    """后台定时写心跳；退出时写一次 stopped。写入失败只记日志，不能影响语音本身。"""

    def __init__(self, path: Path, report: SettingsReport, extra: Callable[[], Mapping[str, Any]],
                 *, interval_s: float = HEARTBEAT_S, clock: Callable[[], float] = time.time):
        self.path, self.report, self.extra = path, report, extra
        self.interval_s, self.clock = interval_s, clock
        self.started_at = clock()

    def write(self, state: str = 'running') -> None:
        try:
            write_status(self.path, snapshot(self.report, started_at=self.started_at, state=state,
                                             now=self.clock(), **dict(self.extra())))
        except Exception as exc:
            logger.warning('[运行状态] 写入失败：%s', type(exc).__name__)

    async def run(self) -> None:
        try:
            while True:
                self.write()
                await asyncio.sleep(self.interval_s)
        finally:
            self.write('stopped')


@dataclass(frozen=True)
class ProcessStatus:
    """读取方看到的状态。state 取值：running / stale / stopped / not_started / unreadable / unavailable。"""
    state: str
    data: dict[str, Any] | None = None
    age_s: float | None = None
    detail: str = ''


def read_status_file(path: Path, *, now: float | None = None,
                     stale_after_s: float = STALE_AFTER_S) -> ProcessStatus:
    now = time.time() if now is None else now
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return ProcessStatus('not_started', detail='没有状态文件：进程没有启动过，或者从未成功写入')
    except (OSError, ValueError) as exc:
        return ProcessStatus('unreadable', detail=f'状态文件无法读取：{type(exc).__name__}')
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        return ProcessStatus('unreadable', detail='状态文件格式不认识')
    heartbeat = data.get('heartbeat_at')
    age = now - heartbeat if isinstance(heartbeat, (int, float)) else None
    if data.get('state') == 'stopped':
        return ProcessStatus('stopped', data, age, '进程正常退出')
    if age is None or age > stale_after_s:
        # 进程可能崩溃、被强制结束，或者卡住了：不能当成仍在运行。
        return ProcessStatus('stale', data, age, f'心跳已经 {age:.0f} 秒没有更新' if age is not None else '没有心跳时间')
    return ProcessStatus('running', data, age)


def saved_report(process: str, env_file: Path) -> SettingsReport:
    """只按 .env 文件（加上默认值）计算配置，不受读取方自己进程环境变量的影响。"""
    from dotenv import dotenv_values
    values = {key: value for key, value in dotenv_values(env_file).items() if value is not None}
    return check_settings(process, values)


@dataclass(frozen=True)
class SettingComparison:
    name: str
    saved: Any
    effective: Any
    state: str  # applied / pending_restart / unknown


def compare_settings(saved: SettingsReport, status: ProcessStatus) -> list[SettingComparison]:
    """逐个比较已保存和正在生效的值。进程不在运行时，正在生效的值是"不知道"，不能猜。"""
    effective = (status.data or {}).get('settings') if status.state == 'running' else None
    rows = []
    for name, value in effective_settings(saved).items():
        if not isinstance(effective, dict) or name not in effective:
            rows.append(SettingComparison(name, value, None, 'unknown'))
            continue
        live = effective[name]
        rows.append(SettingComparison(name, value, live, 'applied' if _same(value, live) else 'pending_restart'))
    return rows


def _same(saved: Any, live: Any) -> bool:
    # JSON 往返后 30.0 和 30 应当视为相同；其余按值比较。
    if isinstance(saved, (int, float)) and isinstance(live, (int, float)) \
            and not isinstance(saved, bool) and not isinstance(live, bool):
        return float(saved) == float(live)
    return saved == live
