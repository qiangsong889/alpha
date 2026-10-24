"""Alpha 所有可配置参数的唯一来源：名字、类型、默认值、范围、分组、谁读取、改了要重启谁。

两种用法：
- setting("名字")：替代 os.getenv。每次调用都按清单转换类型并校验，测试里 patch 环境变量照常生效。
- check_settings(进程)：进程启动时校验整份清单和参数之间的关系，一次报告全部问题，并给出配置指纹。

运行 `python -m core.settings` 查看当前值、错误、警告和 .env 里的未识别参数（密钥只显示是否已设置）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

BRAIN, VOICE, PERSONAL, CLI, LEGACY = 'brain', 'voice', 'personal', 'cli', 'legacy'


class SettingsError(ValueError):
    """一个或多个参数不合法；problems 里是每一条具体原因。"""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__('；'.join(problems))


@dataclass(frozen=True)
class Setting:
    name: str
    group: str
    description: str
    kind: str = 'str'  # str / int / float / bool / choice / timezone / secret
    default: Any = None
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    readers: frozenset[str] = frozenset()
    editable: bool = True  # False：只显示，后端拒绝从页面修改（安全边界、旧路径、连接地址等）

    @property
    def secret(self) -> bool:
        return self.kind == 'secret'

    @property
    def legacy(self) -> bool:
        return self.readers == frozenset({LEGACY})


def _s(name, group, description, kind='str', default=None, *, readers, choices=(), minimum=None,
       maximum=None, editable=True) -> Setting:
    return Setting(name, group, description, kind, default, tuple(choices), minimum, maximum,
                   frozenset(readers), editable and kind != 'secret')


def _legacy(name: str, description: str = '已移除入口的旧参数，仅保留识别；当前运行不读取') -> Setting:
    return _s(name, '旧路径', description, readers={LEGACY}, editable=False)


SETTINGS: tuple[Setting, ...] = (
    # ── 服务连接 ──
    _s('HA_URL', '服务连接', 'Home Assistant 地址（不含 /api/...）', readers={BRAIN, PERSONAL}, editable=False),
    _s('HA_TOKEN', '密钥', 'Home Assistant 长期访问令牌', 'secret', readers={BRAIN, PERSONAL}),
    _s('ANTHROPIC_API_KEY', '密钥', 'Claude API 密钥（SDK 自动读取）', 'secret', readers={BRAIN}),
    _s('DASHSCOPE_API_KEY', '密钥', '阿里云 DashScope 密钥（ASR / TTS / Omni）', 'secret', readers={VOICE}),
    _s('TYPESAFE_API_KEY', '密钥', 'Jev（TypeSafe 直连）密钥', 'secret', readers={VOICE}),
    _s('AI_GATEWAY_API_KEY', '密钥', 'Jev（Vercel 网关）密钥', 'secret', readers={VOICE}),
    _s('OPENAI_API_KEY', '密钥', '已移除的 LiveKit / OpenAI 实时入口密钥（仅识别旧配置）', 'secret', readers={LEGACY}),
    _s('QWEN_WORKSPACE_ID', '服务连接', 'DashScope 工作空间 ID', readers={VOICE}, editable=False),
    _s('QWEN_REGION', '服务连接', 'DashScope 区域', 'choice', 'singapore',
       choices=('singapore', 'beijing'), readers={VOICE}),
    _s('ALPHA_BRAIN_URL', '服务连接', '语音层连接的大脑地址', default='http://127.0.0.1:8765',
       readers={VOICE}, editable=False),
    # ── 大脑 ──
    _s('ANTHROPIC_MODEL', '大脑', '大脑使用的 Claude 模型', default='claude-sonnet-5', readers={BRAIN}),
    _s('ALPHA_AUTO_MEMORY', '大脑', '自动记忆（后台提取对话中的偏好）', 'bool', False, readers={BRAIN}),
    _s('REMINDER_TIMEZONE', '位置与时区', '提醒、日程和时间注入使用的时区', 'timezone',
       'America/Los_Angeles', readers={BRAIN, PERSONAL}),
    _s('HOME_CITY', '位置与时区', '网页搜索的城市（留空则不带位置）', default='', readers={BRAIN}),
    _s('HOME_REGION', '位置与时区', '网页搜索的州/省', default='', readers={BRAIN}),
    _s('HOME_COUNTRY', '位置与时区', '网页搜索的国家代码', default='', readers={BRAIN}),
    # ── 上下文 ──
    _s('ALPHA_CONTEXT_IDLE_MIN', '上下文', '空闲多少分钟后开始新的对话', 'float', 30.0,
       minimum=0.01, maximum=1440, readers={BRAIN}),
    _s('ALPHA_CONTEXT_COMPACT_TOKENS', '上下文', '对话历史超过多少 token 后台压缩', 'int', 15000,
       minimum=1, readers={BRAIN}),
    _s('ALPHA_CONTEXT_HARD_TOKENS', '上下文', '对话历史的硬上限（请求前同步压缩）', 'int', 40000,
       minimum=2, readers={BRAIN}),
    # ── 收音 ──
    _s('QWEN_ALPHA_INPUT', '收音', '收音方式：asr（Silero + 独立 ASR）或 omni（旧路径，可能卡住）', 'choice',
       'asr', choices=('asr', 'omni'), readers={VOICE}),
    _s('SILERO_MODEL_PATH', '收音', 'Silero VAD 模型文件（留空用内置路径）', default='', readers={VOICE},
       editable=False),
    _s('REALTIME_INPUT_DEVICE', '收音', '麦克风设备（编号或名字；留空用系统默认，并跟随系统切换）', default='', readers={VOICE}),
    _s('QWEN_ASR_LANGUAGE', '收音', '语音识别的语言（留空自动判断，中英文都能识别；填 zh 固定中文）', 'choice', '',
       choices=('zh', 'en', 'yue', 'ja', 'ko'), readers={VOICE}),
    _s('REALTIME_INPUT_CHANNEL', '收音', '用麦克风设备的第几个声道（从 1 开始；XVF3800 的语音识别波束是 2）', 'int', 1,
       minimum=1, maximum=16, readers={VOICE}),
    _s('QWEN_VAD', '收音', 'Omni 路径的服务端 VAD', 'choice', 'server_vad',
       choices=('server_vad', 'semantic_vad'), readers={VOICE}),
    _s('QWEN_SILENCE_MS', '收音', 'Omni 路径：静音多少毫秒算说完', 'int', 800, minimum=200, maximum=6000,
       readers={VOICE}),
    _s('QWEN_ALPHA_LOCAL_VAD', '收音', 'Omni 路径：本地 VAD 把关上传', 'bool', False, readers={VOICE}),
    _s('QWEN_ALPHA_VAD_PRE_MS', '收音', 'Omni 路径：开口前缓冲（毫秒）', 'int', 400, minimum=0, maximum=2000,
       readers={VOICE}),
    _s('QWEN_ALPHA_VAD_TAIL_MS', '收音', 'Omni 路径：尾部静音（毫秒，留空按静音时长自动计算）', 'int', None,
       minimum=0, maximum=10000, readers={VOICE}),
    _s('QWEN_ALPHA_VAD_MODE', '收音', 'Omni 路径：WebRTC VAD 激进程度', 'int', 3, minimum=0, maximum=3,
       readers={VOICE}),
    _s('QWEN_ALPHA_VAD_ONSET_MS', '收音', 'Omni 路径：连续人声多少毫秒才开始上传', 'int', 80,
       minimum=0, maximum=2000, readers={VOICE}),
    _s('QWEN_ALPHA_VAD_MIN_RMS', '收音', 'Omni 路径：最低音量门槛', 'float', 0.003, minimum=0, maximum=1,
       readers={VOICE}),
    _s('QWEN_ALPHA_VAD_DEBUG', '收音', 'Omni 路径：打印 VAD 诊断', 'bool', False, readers={VOICE}),
    # ── 提交与打断 ──
    _s('QWEN_ALPHA_SUBMIT_MODE', '提交与打断', '判断说完：jev（完整即提交）或 window（固定窗口）', 'choice',
       'jev', choices=('jev', 'window'), readers={VOICE}),
    _s('QWEN_ALPHA_COLLECTION_SECONDS', '提交与打断', '安静多少秒提交（window 模式和 Jev 不确定时）', 'float',
       2.0, minimum=0, maximum=60, readers={VOICE}),
    _s('QWEN_ALPHA_CONTINUATION_SECONDS', '提交与打断', '句子没说完时最多等多少秒再追问', 'float', 5.0,
       minimum=0, maximum=120, readers={VOICE}),
    _s('QWEN_ALPHA_JEV_INTERRUPTIONS', '提交与打断', '播报时插话先用 Jev 判断是不是在对 Alpha 说', 'bool',
       True, readers={VOICE}),
    _s('QWEN_ALPHA_INTERRUPT_MODE', '提交与打断',
       '播报期间怎样打断：call（只接受“Alpha……”、“停”或提问后的简短回答）或 jev（由 Jev 判断是否对 Alpha 说话）',
       'choice', 'call', choices=('call', 'jev'), readers={VOICE}),
    _s('QWEN_ALPHA_UNCERTAIN_INTERRUPTION', '提交与打断', 'Jev 判断不了时：hold（继续播报）或 accept（打断）',
       'choice', 'hold', choices=('hold', 'accept'), readers={VOICE}),
    _s('JEV_PROVIDER', '提交与打断', 'Jev 服务提供方', 'choice', 'typesafe', choices=('typesafe', 'vercel'),
       readers={VOICE}),
    _s('JEV_MODEL', '提交与打断', 'Jev 模型（留空用默认）', default='', readers={VOICE}),
    _s('JEV_TIMEOUT_MS', '提交与打断', 'Jev 请求超时（毫秒）', 'float', 300.0, minimum=1, maximum=10000,
       readers={VOICE}),
    _s('JEV_MIN_CONFIDENCE', '提交与打断', 'Jev 判断的最低置信度', 'float', 0.8, minimum=0, maximum=1,
       readers={VOICE}),
    # ── 播报 ──
    _s('QWEN_ALPHA_SPEAKER', '播报', '播报方式：tts（原样朗读）或 omni（可能改写原文）', 'choice', 'tts',
       choices=('tts', 'omni'), readers={VOICE}),
    _s('QWEN_TTS_VOICE', '播报', 'Qwen TTS 音色', default='Ethan', readers={VOICE}),
    _s('QWEN_VOICE', '播报', 'Omni 路径的音色', default='Tina', readers={VOICE}),
    _s('REALTIME_OUTPUT_DEVICE', '播报', '扬声器设备（编号或名字；留空用系统默认，并跟随系统切换）', default='', readers={VOICE}),
    _s('QWEN_SESSION_SECONDS', '播报', 'Qwen 会话轮换周期（秒）', 'float', 1800.0, minimum=60, maximum=1800,
       readers={VOICE}, editable=False),
    # ── 费用 ──
    _s('JEV_INPUT_USD_PER_MTOK', '费用', 'Jev 每百万输入 token 的估算价格（美元）', 'float', 0.042,
       minimum=0, readers={BRAIN, VOICE}, editable=False),
    # ── 文字 CLI ──
    _s('ALPHA_VOICE', '文字 CLI', 'CLI 语音模式下 macOS say 的声音', default='Meijia', readers={CLI}),
    # ── 已退役参数：兼容识别现有.env，不再代表运行入口，不在页面开放 ──
    _legacy('INTERRUPT_MS'), _legacy('COLLECTION_WINDOW_S'), _legacy('COLLECTION_INCOMPLETE_S'),
    _legacy('OPENAI_REALTIME_MODEL'), _legacy('OPENAI_REALTIME_VOICE'), _legacy('RECAP_TTS_VOICE'),
    _legacy('SILENCE_MS', '旧的 Whisper 按键说话（voice/listen.py）读取'),
    _legacy('ALPHA_TURN_DETECTOR'), _legacy('ALPHA_TURN_MAX_DELAY'), _legacy('ALPHA_TTS_VOICE'),
    _legacy('ALPHA_TTS_SPEED'), _legacy('ALPHA_STT_REALTIME'), _legacy('ALPHA_SEARCH_ROUTER'),
    _legacy('ALPHA_JEV_INTERRUPTIONS'), _legacy('ALPHA_CONTINUATION_SECONDS'),
    _legacy('ALPHA_COLLECTION_SECONDS'), _legacy('ALPHA_COLLECTION_ENABLED'),
    _legacy('ALPHA_ANSWER_WINDOW_SECONDS'),
)
REGISTRY: dict[str, Setting] = {item.name: item for item in SETTINGS}
assert len(REGISTRY) == len(SETTINGS), '参数清单里有重复的名字'


def _convert(item: Setting, raw: str) -> Any:
    value = raw.strip()
    if item.kind in ('str', 'secret'):
        return value
    if item.kind == 'bool':
        if value not in ('0', '1'):
            raise ValueError('必须是 0 或 1')
        return value == '1'
    if item.kind == 'choice':
        value = value.lower()
        if value not in item.choices:
            raise ValueError('必须是 ' + ' / '.join(item.choices) + ' 之一')
        return value
    if item.kind == 'timezone':
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError('不是有效的 IANA 时区') from exc
        return value
    if item.kind == 'int':
        try:
            number: float = int(value)
        except ValueError as exc:
            raise ValueError('必须是整数') from exc
    elif item.kind == 'float':
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError('必须是数字') from exc
        if not math.isfinite(number):
            raise ValueError('必须是有限数字')
    else:
        raise ValueError(f'未知的参数类型 {item.kind}')
    if item.minimum is not None and number < item.minimum:
        raise ValueError(f'不能小于 {item.minimum:g}')
    if item.maximum is not None and number > item.maximum:
        raise ValueError(f'不能大于 {item.maximum:g}')
    return number


def setting(name: str, env: Mapping[str, str] | None = None) -> Any:
    """读取一个已登记的参数：没设置或为空时用清单里的默认值；不合法时抛出 SettingsError。"""
    item = REGISTRY.get(name)
    if item is None:
        raise KeyError(f'参数 {name} 不在 core/settings.py 的清单里')
    raw = (os.environ if env is None else env).get(name)
    if raw is None or not raw.strip():
        return item.default
    try:
        return _convert(item, raw)
    except ValueError as exc:
        raise SettingsError([f'{name}={raw!r} 不合法：{exc}']) from None


# ── 参数之间的关系 ──
# 每条规则返回 (级别, 说明)；级别 error 会阻止启动，warning 只提示。
Rule = Callable[[Callable[[str], Any], str], Iterable[tuple[str, str]]]


def _context_limits(get, process):
    if process == BRAIN and get('ALPHA_CONTEXT_COMPACT_TOKENS') >= get('ALPHA_CONTEXT_HARD_TOKENS'):
        yield 'error', 'ALPHA_CONTEXT_COMPACT_TOKENS 必须小于 ALPHA_CONTEXT_HARD_TOKENS'


def _required_credentials(get, process):
    needed = {BRAIN: ('HA_URL', 'HA_TOKEN', 'ANTHROPIC_API_KEY'),
              VOICE: ('DASHSCOPE_API_KEY', 'QWEN_WORKSPACE_ID'),
              PERSONAL: ('HA_URL', 'HA_TOKEN')}.get(process, ())
    for name in needed:
        if not get(name):
            yield 'error', f'{name} 未设置（{process} 必需）'


def _jev_credentials(get, process):
    if process == VOICE:
        key = 'AI_GATEWAY_API_KEY' if get('JEV_PROVIDER') == 'vercel' else 'TYPESAFE_API_KEY'
        if not get(key):
            yield 'warning', f'{key} 未设置：Jev 不可用，判断说完会回退到本地规则和安静窗口'


def _omni_upload_gate(get, process):
    if (process == VOICE and get('QWEN_ALPHA_INPUT') == 'omni' and get('QWEN_ALPHA_LOCAL_VAD')
            and get('QWEN_VAD') != 'server_vad'):
        yield 'error', 'QWEN_ALPHA_LOCAL_VAD=1 需要 QWEN_VAD=server_vad'


def _input_channel(get, process):
    if process == VOICE and get('REALTIME_INPUT_CHANNEL') > 1:
        if not get('REALTIME_INPUT_DEVICE'):
            yield 'warning', ('REALTIME_INPUT_CHANNEL 大于 1 但没有固定 REALTIME_INPUT_DEVICE：'
                              '跟随系统默认设备时，换成单声道麦克风会打不开')
        if get('QWEN_ALPHA_INPUT') == 'omni':
            yield 'warning', 'REALTIME_INPUT_CHANNEL 只对 asr 收音生效，omni 路径仍用第 1 个声道'


def _legacy_paths(get, process):
    if process == VOICE:
        if get('QWEN_ALPHA_INPUT') == 'omni':
            yield 'warning', 'QWEN_ALPHA_INPUT=omni 是旧路径：服务端 VAD 可能等不到说完而卡住'
        if get('QWEN_ALPHA_SPEAKER') == 'omni':
            yield 'warning', 'QWEN_ALPHA_SPEAKER=omni 是旧路径：Omni 朗读可能改写原文'


def _model_price(get, process):
    if process == BRAIN:
        from core.costs import CLAUDE  # 延迟导入，避免 costs 和 settings 互相导入
        model = get('ANTHROPIC_MODEL')
        if not any(model == known or model.startswith(known + '-20') for known in CLAUDE):
            yield 'warning', f'ANTHROPIC_MODEL={model} 不在价格表里，费用会记为未计价'


class _InvalidInput(Exception):
    """规则用到了本身不合法的参数；这条规则这次不下结论。"""


RULES: tuple[Rule, ...] = (_context_limits, _required_credentials, _jev_credentials,
                           _omni_upload_gate, _input_channel, _legacy_paths, _model_price)


@dataclass(frozen=True)
class SettingsReport:
    process: str
    values: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fingerprint: str = ''

    def public_values(self) -> dict[str, Any]:
        """可以显示或发给页面的值：密钥只说明是否已设置。"""
        return {name: (bool(value) if REGISTRY[name].secret else value) for name, value in self.values.items()}


def check_settings(process: str, env: Mapping[str, str] | None = None) -> SettingsReport:
    """校验整份清单（只读取，不修改环境）。process 决定检查哪些必需参数和关系规则。"""
    env = os.environ if env is None else env
    values: dict[str, Any] = {}
    errors: list[str] = []
    invalid: set[str] = set()
    for item in SETTINGS:
        try:
            values[item.name] = setting(item.name, env)
        except SettingsError as exc:
            # 只有这个进程会读的参数写错了才阻止启动；别的进程的参数由它自己检查。
            if process in item.readers:
                errors.extend(exc.problems)
            invalid.add(item.name)
            values[item.name] = item.default

    def get(name: str) -> Any:
        if name in invalid:
            raise _InvalidInput(name)
        return values[name]

    warnings: list[str] = []
    for rule in RULES:
        # 每条规则都运行，才能一次列出全部问题；只跳过用到了不合法参数的规则，
        # 因为拿默认值去判断参数之间的关系，结论可能是错的。
        try:
            results = list(rule(get, process))
        except _InvalidInput:
            continue
        for level, message in results:
            (errors if level == 'error' else warnings).append(message)
    # 指纹只包含这个进程会读的参数：改了语音的参数，不会让大脑也显示"需要重启"。
    mine = {name: (bool(value) if REGISTRY[name].secret else value) for name, value in values.items()
            if process in REGISTRY[name].readers}
    fingerprint = sha256(json.dumps(mine, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    return SettingsReport(process, values, errors, warnings, fingerprint)


def require_settings(process: str, logger=None) -> SettingsReport:
    """进程启动时调用：有错误就抛出 SettingsError（列出全部问题），警告写进日志。"""
    report = check_settings(process)
    if logger is not None:
        for message in report.warnings:
            logger.warning('[配置] %s', message)
        logger.info('[配置] %s 进程配置指纹 %s', process, report.fingerprint)
    if report.errors:
        raise SettingsError(report.errors)
    return report


def unrecognized_keys(env_file: str | Path) -> list[str]:
    """.env 里出现、但清单里没有的参数。它们可能被 SDK、子进程或脚本读取，所以只叫"未识别"。"""
    from dotenv import dotenv_values
    return sorted(key for key in dotenv_values(env_file) if key not in REGISTRY)


def _report() -> int:
    from dotenv import dotenv_values, load_dotenv
    from core.paths import ENV_FILE
    saved = dotenv_values(ENV_FILE)
    load_dotenv(ENV_FILE)
    status = 0
    for process in (BRAIN, VOICE):
        report = check_settings(process)
        print(f'== {process} ｜ 指纹 {report.fingerprint}')
        for message in report.errors:
            print('  ✗', message)
            status = 1
        for message in report.warnings:
            print('  !', message)
    print()
    groups: dict[str, list[Setting]] = {}
    for item in SETTINGS:
        if not item.legacy:
            groups.setdefault(item.group, []).append(item)
    for group, items in groups.items():
        print(f'[{group}]')
        for item in items:
            _print_setting(item, saved)
    unknown = unrecognized_keys(ENV_FILE)
    if unknown:
        print('\n.env 里的未识别参数（可能已不再使用，也可能被 SDK / 脚本读取，请自行确认）：')
        for key in unknown:
            print('  ?', key)
    return status


def _print_setting(item: Setting, saved: Mapping[str, Any]) -> None:
    try:
        value = setting(item.name)
    except SettingsError:
        value = '（不合法）'
    shown = ('已设置' if value else '未设置') if item.secret else value
    raw = os.environ.get(item.name)
    source = ('默认' if raw is None or not raw.strip() else
              '.env' if saved.get(item.name) == raw else '启动时的环境变量（覆盖了 .env）')
    lock = '' if item.editable else ' 🔒'
    print(f'  {item.name:<36} {shown!s:<28} {source}{lock}')


if __name__ == '__main__':
    raise SystemExit(_report())
