"""控制台的数据来源：全部只读。

- SQLite 用 `mode=ro` 打开：控制台从结构上就写不了存档、费用和操作记录；文件不存在也不会被创建。
- 每次请求新开连接、用完就关：不和大脑、语音的写入抢锁，也没有跨线程共享连接的问题。
- 返回给页面的值里没有任何密钥：密钥只说明是否已设置。
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable

from core.runtime import ProcessStatus, compare_settings, read_status_file, saved_report
from core.settings import BRAIN, SETTINGS, VOICE, SettingsError, setting, unrecognized_keys

ESTIMATE_REASON = 'embedded_asr_billing_unverified'  # 旧 Omni 内置 ASR 的参考估算，不计入已知小计
PROCESSES = (BRAIN, VOICE)


class Unavailable(Exception):
    """数据库文件不存在或打不开。页面显示"没有数据"，不能当成零。"""


def open_readonly(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise Unavailable(f'{path.name} 不存在')
    try:
        db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=2)
    except sqlite3.Error as exc:
        raise Unavailable(f'{path.name} 打不开：{exc}') from exc
    db.row_factory = sqlite3.Row
    return db


def start_of_today(now: float) -> float:
    """本机时区的今天零点。"""
    return datetime.fromtimestamp(now).astimezone().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _money(value: Decimal) -> str:
    return f'{value:.6f}'


def public_status(status: ProcessStatus) -> dict[str, Any]:
    """给总览用的进程状态：去掉逐项参数（设置页单独比较），其余字段原样给出。"""
    data = {key: value for key, value in (status.data or {}).items() if key != 'settings'}
    return {'state': status.state, 'detail': status.detail, 'age_s': status.age_s, **data}


@dataclass
class ConsoleData:
    env_file: Path
    archive_db: Path
    costs_db: Path
    operations_db: Path
    voice_status: Path
    clock: Callable[[], float] = field(default=time.time)

    # ── 进程状态 ──

    def voice(self) -> ProcessStatus:
        return read_status_file(self.voice_status, now=self.clock())

    def brain_url(self) -> str:
        """大脑地址按已保存的 .env 计算；地址本身写错时用默认值，不让控制台启动失败。"""
        try:
            return setting('ALPHA_BRAIN_URL', self._saved_env())
        except SettingsError:
            return setting('ALPHA_BRAIN_URL', {})

    def _saved_env(self) -> dict[str, str]:
        from dotenv import dotenv_values
        if not self.env_file.exists():
            return {}
        return {key: value for key, value in dotenv_values(self.env_file).items() if value is not None}

    # ── 总览 ──

    def overview(self, brain: ProcessStatus) -> dict[str, Any]:
        now = self.clock()
        since = start_of_today(now)
        voice = self.voice()
        today = self._today_turns(since)
        costs = self._today_costs(since)
        if today.get('available') and costs.get('available') and today['turns']:
            costs['known_per_turn_usd'] = _money(Decimal(costs['known_usd']) / today['turns'])
        settings = self.settings(brain, voice)
        return {
            'now': now, 'since': since,
            'processes': {BRAIN: public_status(brain), VOICE: public_status(voice)},
            'today': today, 'costs': costs,
            'attention': {
                'unknown_operations': self._unknown_operations(),
                'pending_restart': [row['name'] for row in settings['items'] if row['state'] == 'pending_restart'],
                'unrecognized_keys': settings['unrecognized_keys'],
                'saved_problems': settings['saved_problems'],
            },
        }

    def _today_turns(self, since: float) -> dict[str, Any]:
        try:
            with closing(open_readonly(self.archive_db)) as db:
                row = db.execute("""
                    SELECT COUNT(*) AS turns, COUNT(DISTINCT session_id) AS sessions,
                           SUM(status = 'error') AS errors, AVG(brain_ms) AS avg_brain_ms
                    FROM turns WHERE created_at >= ?""", (since,)).fetchone()
                delivery = {r['delivery']: r['n'] for r in db.execute(
                    'SELECT delivery, COUNT(*) AS n FROM turns WHERE created_at >= ? GROUP BY delivery', (since,))}
        except (Unavailable, sqlite3.Error) as exc:
            return {'available': False, 'detail': str(exc)}
        avg = row['avg_brain_ms']
        return {'available': True, 'turns': row['turns'], 'sessions': row['sessions'], 'errors': row['errors'] or 0,
                'avg_brain_ms': round(avg) if avg is not None else None, 'delivery': delivery}

    def _today_costs(self, since: float) -> dict[str, Any]:
        """三类分开：已知（服务端用量或接口费用）、参考估算、待核对（缺少最终用量或价格，不能当成零）。"""
        try:
            with closing(open_readonly(self.costs_db)) as db:
                rows = db.execute('SELECT component, usd, reason FROM cost_events WHERE updated_at >= ?',
                                  (since,)).fetchall()
        except (Unavailable, sqlite3.Error) as exc:
            return {'available': False, 'detail': str(exc)}
        known, estimate, to_verify = Decimal(0), Decimal(0), 0
        components: dict[str, dict[str, Any]] = {}
        for row in rows:
            usd, reason = Decimal(row['usd']), row['reason']
            item = components.setdefault(row['component'], {'known': Decimal(0), 'estimate': Decimal(0),
                                                            'to_verify': 0, 'events': 0})
            item['events'] += 1
            if reason == ESTIMATE_REASON:
                estimate += usd
                item['estimate'] += usd
                continue
            known += usd
            item['known'] += usd
            if reason:
                to_verify += 1
                item['to_verify'] += 1
        return {
            'available': True, 'known_usd': _money(known), 'estimate_usd': _money(estimate),
            'to_verify': to_verify, 'events': len(rows),
            'components': {name: {**item, 'known': _money(item['known']), 'estimate': _money(item['estimate'])}
                           for name, item in sorted(components.items())},
        }

    def _unknown_operations(self) -> dict[str, Any]:
        """结果未知的操作：可能已经执行，也可能没有，需要人去确认。不限于今天。"""
        try:
            with closing(open_readonly(self.operations_db)) as db:
                count = db.execute("SELECT COUNT(*) FROM operations WHERE status = 'unknown'").fetchone()[0]
        except (Unavailable, sqlite3.Error) as exc:
            return {'available': False, 'detail': str(exc)}
        return {'available': True, 'count': count}

    # ── 设置 ──

    def settings(self, brain: ProcessStatus, voice: ProcessStatus | None = None) -> dict[str, Any]:
        """每个参数：已保存的值（.env 或默认）、各进程正在生效的值、合并后的状态。"""
        voice = self.voice() if voice is None else voice
        saved_env = self._saved_env()
        statuses = {BRAIN: brain, VOICE: voice}
        saved = {process: saved_report(process, self.env_file) if self.env_file.exists() else None
                 for process in PROCESSES}
        live = {process: {row.name: row for row in compare_settings(saved[process], statuses[process])}
                for process in PROCESSES if saved[process] is not None}
        items = []
        for item in SETTINGS:
            if item.legacy:
                continue
            raw = saved_env.get(item.name)
            source = 'env' if raw is not None and raw.strip() else 'default'
            invalid = None
            try:
                value = setting(item.name, saved_env)
            except SettingsError as exc:
                value, invalid = None, str(exc)
            processes = []
            for process in PROCESSES:
                if process in item.readers and item.name in live.get(process, {}):
                    row = live[process][item.name]
                    processes.append({'process': process, 'state': row.state,
                                      'effective': _public(item, row.effective) if row.state != 'unknown' else None})
            items.append({
                'name': item.name, 'group': item.group, 'description': item.description, 'kind': item.kind,
                'default': None if item.secret else item.default, 'choices': list(item.choices),
                'minimum': item.minimum, 'maximum': item.maximum, 'readers': sorted(item.readers),
                'editable': item.editable, 'secret': item.secret,
                'saved': _public(item, value), 'source': source, 'invalid': invalid,
                'processes': processes, 'state': _combined(processes),
            })
        return {
            'processes': {process: {'state': statuses[process].state, 'detail': statuses[process].detail,
                                    'running_fingerprint': (statuses[process].data or {}).get('settings_fingerprint'),
                                    'saved_fingerprint': saved[process].fingerprint if saved[process] else None}
                          for process in PROCESSES},
            'items': items,
            'groups': list(dict.fromkeys(row['group'] for row in items)),
            'unrecognized_keys': unrecognized_keys(self.env_file) if self.env_file.exists() else [],
            # 按已保存的 .env 重新启动时会遇到的问题：错误会让那个进程拒绝启动。
            'saved_problems': {process: {'errors': report.errors, 'warnings': report.warnings}
                               for process, report in saved.items() if report is not None},
        }


def _public(item, value: Any) -> Any:
    if item.secret:
        return bool(value)
    return value


def _combined(processes: list[dict[str, Any]]) -> str:
    """有一个进程需要重启就是 pending_restart；全部已生效才是 applied；没有常驻进程读取的是 next_start。"""
    if not processes:
        return 'next_start'
    states = {row['state'] for row in processes}
    if 'pending_restart' in states:
        return 'pending_restart'
    if states == {'applied'}:
        return 'applied'
    return 'unknown'
