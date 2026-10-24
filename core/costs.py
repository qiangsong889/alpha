"""USD 标准价估算账本；只存用量，不存语音、提示词或密钥。

两个进程共用 SQLite。缺失用量是 unknown，不能当作零；可用后续相同 event_id 补全。
价格核对日期 2026-10-17，来源和限制见 docs/COSTS.md。
"""
import json
import logging
import os
import sqlite3
import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from core.trace import CURRENT_SESSION, CURRENT_TURN

logger = logging.getLogger('alpha.costs')


class CumulativeUsage:
    """每条持久连接一个实例，把累计 JSON 用量换成本次用量。

    缺失字段不输出，也不清除此前的计数基线；返回值和基线均不引用调用方的字典。
    bool 是状态而非计数；列表等非计数字段原样复制。
    """
    def __init__(self):
        self.last = None

    def reset(self):
        self.last = None

    def delta(self, usage: dict) -> dict:
        def numeric(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        def subtract(current, previous, path):
            if isinstance(current, dict):
                old = previous if isinstance(previous, dict) else {}
                return {key: subtract(value, old.get(key), f'{path}.{key}' if path else key)
                        for key, value in current.items()}
            if numeric(current) and numeric(previous):
                difference = current - previous
                if difference < 0:
                    logger.warning('[累计用量重置] %s：%s → %s，使用本次原始值', path, previous, current)
                    return current
                return difference
            return deepcopy(current)

        def merge(previous, current):
            if isinstance(previous, dict) and isinstance(current, dict):
                result = deepcopy(previous)
                for key, value in current.items():
                    result[key] = merge(previous.get(key), value)
                return result
            return deepcopy(current)

        result = subtract(usage, self.last, '')
        self.last = merge(self.last, usage)
        return result


MILLION = Decimal(1000000)
# input/output/cache read per million; cache write uses 1.25x/2x input.
CLAUDE = {
    'claude-sonnet-5': ('2', '10', '.2'),
    'claude-sonnet-5-5': ('2', '10', '.1'),
    'claude-sonnet-4-6': ('3', '15', '.3'),
    'claude-sonnet-4-5': ('3', '15', '.3'),
    'claude-haiku-5-5': ('.1', '.5', '.01'),
    'claude-haiku-4-5': ('1', '5', '.1'),
}
OMNI = {'singapore': ('.23', '.93', '.7', '1.87'),
        'beijing': ('.212', '.848', '.636', '1.696')}


def data(value):
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    return value if isinstance(value, dict) else {}


def count(value):
    return value if type(value) is int and value >= 0 else None


def quote(component, model, usage, region='singapore'):
    """返回 (已知小计, 未计价原因)；数值必须来自服务端或明确的本地测量。"""
    u = data(usage)
    try:
        if component in {'Claude', 'Memory', 'Context'}:
            rates = next((v for k, v in sorted(CLAUDE.items(), key=lambda p: -len(p[0]))
                          if model == k or model.startswith(k + '-20')), None)
            if rates is None:
                return Decimal(0), 'unknown_model_price'
            i, o, r = map(Decimal, rates)
            values = [count(u.get('input_tokens')), count(u.get('output_tokens')),
                      count(u.get('cache_read_input_tokens') or 0),
                      count(u.get('cache_creation_input_tokens') or 0)]
            if None in values:
                return Decimal(0), 'missing_token_usage'
            inp, out, read, write = values
            cache = data(u.get('cache_creation'))
            hour = count(cache.get('ephemeral_1h_input_tokens', 0))
            five = count(cache.get('ephemeral_5m_input_tokens', write - (hour or 0)))
            if hour is None or five is None or hour + five != write:
                return Decimal(0), 'invalid_cache_usage'
            total_input = inp + read + write
            if model.startswith('claude-haiku-5-5') and total_input > 100000:
                i, o, r = Decimal('.5'), Decimal('2.5'), Decimal('.05')
            cost = (inp*i + out*o + read*r + five*i*Decimal('1.25') + hour*i*2) / MILLION
            searches = data(u.get('server_tool_use')).get('web_search_requests', 0)
            if count(searches) is None:
                return cost, 'missing_search_usage'
            return cost + Decimal(searches)*Decimal('.01'), ''
        if component == 'Jev':
            if 'gateway_usd' in u:
                exact = Decimal(str(u['gateway_usd']))
                if not exact.is_finite() or exact < 0:
                    raise ValueError('invalid gateway cost')
                return exact, ''
            if 'jev' not in model:
                return Decimal(0), 'unknown_model_price'
            n = count(u.get('input_tokens'))
            if n is None:
                return Decimal(0), 'missing_token_usage'
            # Gateway reports exact cost when available; otherwise use published rate.
            from core.settings import setting  # 延迟导入：settings 的规则里也会读取价格表
            rate = Decimal(str(setting('JEV_INPUT_USD_PER_MTOK')))
            if not rate.is_finite() or rate < 0:
                raise ValueError('invalid rate')
            return Decimal(n)*rate/MILLION, ''
        if component == 'TTS':
            if model == 'qwen-audio-3.1-tts-flash' and region == 'singapore':
                # 按 token 计价（2026-10 新加坡价目表）：输入 $0.23/M，输出 $1.87/M。
                tokens = count(u.get('input_tokens')), count(u.get('output_tokens'))
                if None in tokens:
                    return Decimal(0), 'missing_token_usage'
                return (Decimal(tokens[0]) * Decimal('.23') + Decimal(tokens[1]) * Decimal('1.87')) / MILLION, ''
            if model == 'qwen-audio-3.1-tts-plus' and region == 'singapore':
                n = count(u.get('characters'))
                if n is None:
                    return Decimal(0), 'missing_billed_characters'
                return Decimal(n) * Decimal('.3') / 10000, ''
            if not (model == 'qwen3-tts-flash-realtime' or model.startswith('qwen3-tts-flash-realtime-20')):
                return Decimal(0), 'unknown_model_price'
            n = count(u.get('characters'))
            if n is None:
                return Decimal(0), 'missing_billed_characters'
            rate = {'singapore': '.13', 'beijing': '.143353'}[region]
            return Decimal(n)*Decimal(rate)/10000, ''
        if component == 'Omni':
            inputs, outputs = data(u.get('input_tokens_details')), data(u.get('output_tokens_details'))
            values = [count(inputs.get('text_tokens')), count(inputs.get('audio_tokens')),
                      count(outputs.get('text_tokens')), count(outputs.get('audio_tokens'))]
            if None in values:
                return Decimal(0), 'missing_modality_usage'
            return sum((Decimal(n)*Decimal(rate) for n, rate in zip(values, OMNI[region])), Decimal(0))/MILLION, ''
        if component == 'ASR':
            # 独立 ASR 的 completed.usage.duration 是服务端报告的秒数。
            # Omni 内嵌转写仍只能按上传时长参考估算，两者不能混为一谈。
            reported = 'duration' in u
            seconds = Decimal(str(u['duration'] if reported else u['sent_audio_seconds']))
            if not seconds.is_finite() or seconds < 0:
                raise ValueError('invalid duration')
            rate = {'singapore': '.00009', 'beijing': '.000047'}[region]
            return seconds*Decimal(rate), '' if reported else 'embedded_asr_billing_unverified'
    except (KeyError, ValueError, TypeError, ArithmeticError):
        return Decimal(0), 'invalid_usage_or_price'
    return Decimal(0), 'unknown_component'


class CostLedger:
    def __init__(self, path):
        self.path = str(path)
        if self.path != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=5)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS cost_events (
            id TEXT PRIMARY KEY, component TEXT NOT NULL, model TEXT NOT NULL,
            region TEXT NOT NULL, usage TEXT NOT NULL, usd TEXT NOT NULL,
            reason TEXT NOT NULL, updated_at REAL NOT NULL)''')
        # 旧账本没有归属列；补列后旧记录保持 NULL，不猜测它们属于哪一轮。
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(cost_events)')}
        for column in ('session_id', 'turn_id'):
            if column not in columns:
                self.db.execute(f'ALTER TABLE cost_events ADD COLUMN {column} TEXT')
        self.db.execute('CREATE INDEX IF NOT EXISTS cost_events_turn ON cost_events (turn_id)')
        self.db.commit()

    def record(self, event_id, component, model, usage=None, *, region='singapore', reason=None,
               session_id=None, turn_id=None):
        usage = data(usage)
        cost, missing = quote(component, model, usage, region)
        missing = reason if reason is not None else missing
        with self.db:
            # 开始记录和最终用量用同一个 id 两次写入；归属以先写入的非空值为准，不被后来的空值覆盖。
            self.db.execute('''INSERT INTO cost_events
                (id, component, model, region, usage, usd, reason, updated_at, session_id, turn_id)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET usage=excluded.usage, usd=excluded.usd,
                reason=excluded.reason, updated_at=excluded.updated_at,
                session_id=COALESCE(cost_events.session_id, excluded.session_id),
                turn_id=COALESCE(cost_events.turn_id, excluded.turn_id)
                WHERE length(cost_events.reason) > 0 OR length(excluded.reason) = 0''',
                (event_id, component, model, region, json.dumps(usage), str(cost), missing, time.time(),
                 session_id, turn_id))
        return cost, missing

    def summary(self):
        totals = {}
        for component, usd, reason in self.db.execute('SELECT component,usd,reason FROM cost_events'):
            row = totals.setdefault(component, {'usd': Decimal(0), 'unpriced': 0, 'events': 0})
            row['usd'] += Decimal(usd)
            row['unpriced'] += bool(reason)
            row['events'] += 1
        return {k: {**v, 'usd': str(v['usd'])} for k, v in totals.items()}

    def close(self):
        self.db.close()


_ledger: CostLedger | None = None


def configure_costs(path=None):
    global _ledger
    if _ledger is not None:
        _ledger.close()
    from core.paths import COSTS_DB
    _ledger = CostLedger(path or COSTS_DB)
    return _ledger


_FROM_CONTEXT = object()


def record_cost(component, model, usage=None, *, event_id=None, region='singapore', reason=None,
                session_id=_FROM_CONTEXT, turn_id=_FROM_CONTEXT):
    """旁路计费不得影响语音或工具执行；未启用时不写入（例如单元测试）。

    归属默认取当前上下文（大脑的一轮对话）；其他进程（例如语音层的 TTS）可以显式传入。
    """
    ident = event_id or uuid4().hex
    if session_id is _FROM_CONTEXT:
        session_id = CURRENT_SESSION.get()
    if turn_id is _FROM_CONTEXT:
        turn_id = CURRENT_TURN.get()
    if _ledger is not None:
        try:
            usd, missing = _ledger.record(ident, component, model, usage, region=region, reason=reason,
                                          session_id=session_id, turn_id=turn_id)
            if missing and not usd:
                logger.info('[费用][%s] 未计价：%s', component, missing)
            else:
                logger.info('[费用][%s] %s $%.8f%s', component,
                            '参考估算' if missing == 'embedded_asr_billing_unverified' else '估算/接口费用', usd,
                            ' ｜ 待核对：' + missing if missing else '')
        except Exception:
            logger.exception('[费用] 账本写入失败，统计不完整')
    return ident


def cost_summary():
    if _ledger is None:
        return {}
    try:
        return _ledger.summary()
    except Exception:
        logger.exception('[费用] 账本读取失败')
        return {}


def log_summary():
    totals = cost_summary()
    if not totals:
        return
    # 包含独立 ASR 的服务端用量费用，排除旧 Omni 的 ASR 参考估算。
    known = sum((Decimal(usd) for usd, reason in _ledger.db.execute(
        'SELECT usd, reason FROM cost_events') if reason != 'embedded_asr_billing_unverified'), Decimal(0))
    unknown = sum(v['unpriced'] for v in totals.values())
    breakdown = ' ｜ '.join(f"{k} ${Decimal(v['usd']):.6f}" for k, v in sorted(totals.items()))
    logger.info('[费用账本累计] %s ｜ 已知小计 $%.6f（不含 ASR参考估算）｜ 待核对 %d 笔；非实际扣费',
                breakdown, known, unknown)


async def costed_call(awaitable, *, component, model):
    ident = record_cost(component, model, reason='request_started_no_final_usage')
    try:
        response = await awaitable
        usage = data(getattr(response, 'usage', None)).copy()
        search_seen = any(getattr(block, 'type', None) == 'web_search_tool_result'
                          or (getattr(block, 'type', None) == 'server_tool_use'
                              and getattr(block, 'name', None) == 'web_search')
                          for block in getattr(response, 'content', []))
        if search_seen and 'web_search_requests' not in data(usage.get('server_tool_use')):
            usage['server_tool_use'] = {'web_search_requests': None}
        record_cost(component, getattr(response, 'model', None) or model, usage, event_id=ident)
        return response
    finally:
        # 异常/取消时保留待核对记录；API 可能已处理，不能假设没有费用。
        log_summary()


class AudioCostMeter:
    """只计成功 send 的 PCM16 单声道 16kHz 字节；含静音，不以墙钟/人声长度代替。"""
    def __init__(self, session_id, region):
        self.ident = 'asr:' + session_id
        self.region = region
        self.bytes = 0
        self.flushed = 0

    def sent(self, chunk):
        self.bytes += len(chunk)
        if self.bytes - self.flushed >= 320000:  # 每 10 秒落盘一次，退出再补尾段。
            self.flush()

    def flush(self):
        if self.bytes == self.flushed:
            return
        record_cost('ASR', 'qwen3-asr-flash-realtime',
                    {'sent_audio_seconds': self.bytes / 32000},
                    event_id=self.ident, region=self.region)
        self.flushed = self.bytes
        log_summary()


if __name__ == '__main__':
    configure_costs()
    print(json.dumps({'currency': 'USD', 'basis': 'standard_price_estimate',
                      'components': cost_summary()}, ensure_ascii=False, indent=2))
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    log_summary()
