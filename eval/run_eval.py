"""真实 Claude + 隔离工具的行为 eval。入口：python -m eval.run_eval。"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4
from zoneinfo import ZoneInfo

# 同时支持 python -m eval.run_eval 和 python eval/run_eval.py。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from mcp.types import CallToolResult, TextContent

import brain.agent as agent
import core.memory as memory_module
import integrations.operations as operations
from brain.email_confirmation import EmailConfirmation
from brain.auto_memory import AutoMemory
from brain.memory_extractor import MemoryExtractor
from core.rules import ALPHA_RULES
from integrations.mcp_client import load_tools
from integrations.tool_executor import HA_DEVICE_TOOLS, make_executor
from integrations.tool_gateway import ToolGateway, ToolNotSent
from integrations.tool_policy import TOOL_POLICY, ToolPolicy

DEFAULT_NOW = '2026-10-16T09:01:00-07:00'
SEARCH_TOOL = {
    'name': 'web_search',
    'description': '联网搜索网页，查找最新信息或附近餐馆。查询应结合对话上下文和配置的默认地点。',
    'input_schema': {'type': 'object', 'properties': {'query': {'type': 'string', 'minLength': 1}},
                     'required': ['query'], 'additionalProperties': False},
}
INTERNAL_OUTPUTS = {agent.VOICE_REPLY}


class FrozenClock:
    """只替换业务模块引用的时间，不修改 asyncio 或 SDK 的全局时钟。"""

    def __init__(self, now: str):
        value = datetime.fromisoformat(now)
        if value.utcoffset() is None:
            raise ValueError('context.now 必须带时区')
        self.epoch = value.timestamp()
        self.elapsed = 0.0

    def time(self) -> float:
        return self.epoch + self.elapsed

    def monotonic(self) -> float:
        return self.elapsed

    def datetime_type(self):
        clock = self

        class FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls.fromtimestamp(clock.time(), tz)

        return FixedDatetime


def plain(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    return deepcopy(value)


def tool_records_from_messages(messages):
    """保留本轮模型实际收到的完整结果，包括网关拒绝；不截断、不让模型撰写摘要。"""
    records = []
    for message in messages:
        content = message.get('content', [])
        if not isinstance(content, list):
            continue
        for raw in content:
            block = plain(raw)
            if block.get('type') == 'tool_use' and message['role'] == 'assistant':
                records.append({'sequence': len(records) + 1, 'call_id': block['id'],
                                'name': block['name'], 'arguments': deepcopy(block['input']),
                                'result_received': False, 'is_error': True, 'result': None})
            elif block.get('type') == 'tool_result' and message['role'] == 'user':
                for record in reversed(records):
                    if record['call_id'] == block['tool_use_id'] and not record['result_received']:
                        record.update(result_received=True, is_error=block.get('is_error', False),
                                      result=deepcopy(block.get('content', '')))
                        break
    return records


class RecordingMessages:
    """包装 SDK，不改大脑；记录每次完整响应的工具调用、用量与 stop_reason。"""

    def __init__(self, messages):
        self.delegate = messages
        self.calls: list[dict] = []
        self.responses: list[dict] = []
        self.schemas: dict[str, dict] = {}
        self.search_location: dict | None = None
        self.request_locations: list[dict | None] = []

    def request(self, kwargs):
        kwargs = dict(kwargs)
        # 大脑把搜索单独附加一次，而 eval 为路由还需把它放入 self.tools。
        # 只在测试包装器去掉这一个重复声明；绝不发送真正的服务端搜索定义。
        unique = {}
        for tool in kwargs['tools']:
            if tool.get('type', 'custom') != 'custom':
                raise ValueError('eval 禁止真实服务端工具')
            if tool['name'] in unique and unique[tool['name']] != tool:
                raise ValueError('同名工具声明不一致')
            unique[tool['name']] = deepcopy(tool)
        kwargs['tools'] = list(unique.values())
        self.schemas = {name: tool['input_schema'] for name, tool in unique.items()}
        self.request_locations.append(deepcopy(self.search_location))
        return kwargs

    def record(self, response):
        request_index = len(self.responses) + 1
        self.responses.append({'stop_reason': response.stop_reason, 'usage': plain(response.usage),
                               'content_types': [block.type for block in response.content],
                               'visible_text_chars': sum(len(block.text.strip()) for block in response.content
                                                         if block.type == 'text')})
        for block in response.content:
            if block.type in {'tool_use', 'server_tool_use'}:
                self.calls.append({'name': block.name, 'arguments': deepcopy(block.input),
                                   'call_id': block.id, 'request_index': request_index,
                                   'type': block.type})
        return response

    async def create(self, **kwargs):
        response = await self.delegate.create(**self.request(kwargs))
        return self.record(response)

    @asynccontextmanager
    async def stream(self, **kwargs):
        async with self.delegate.stream(**self.request(kwargs)) as stream:
            recorder = self

            class RecordedStream:
                recorded = False

                def __aiter__(self):
                    return stream.__aiter__()

                async def get_final_message(self):
                    response = await stream.get_final_message()
                    if not self.recorded:
                        recorder.record(response)
                        self.recorded = True
                    return response

            yield RecordedStream()


class RecordingExtractor(MemoryExtractor):
    """仅增加可观测记录；提取、校验、保存均使用生产逻辑。"""
    def __init__(self, client, records):
        super().__init__(client)
        self.records = records

    async def extract(self, *args, **kwargs):
        record = {}
        try:
            candidates = await super().extract(*args, **kwargs)
            record['candidates'] = [candidate.to_dict() for candidate in candidates]
            return candidates
        except Exception as exc:
            record['error'] = type(exc).__name__
            raise
        finally:
            record.update(model_calls=self.model_calls, usage=self.last_usage,
                          dropped=dict(self.dropped))
            self.records.append(record)


async def drain_auto_memory(brain):
    # ask 已等待并行提取和核对；兼容尚有清理任务的实现。
    if brain.auto_memory is not None:
        while brain.auto_memory.tasks:
            await asyncio.gather(*list(brain.auto_memory.tasks))


async def personal_catalog():
    # 只读取已注册函数的 schema 和 docstring，不启动 MCP 子进程、不调用 Google。
    # import 时 personal_server 会 load_dotenv；阻止它覆盖用例的虚构位置。
    with patch('dotenv.load_dotenv', return_value=False):
        from servers.personal_server import mcp

    class MetadataSession:
        async def list_tools(self, **kwargs):
            return SimpleNamespace(tools=await mcp.list_tools(), next_cursor=None)

    return await load_tools(MetadataSession())


def ha_catalog(devices):
    target = {key: {'type': 'string'} for key in ('name', 'area', 'floor')}
    target['domain'] = {'type': 'array', 'items': {'type': 'string'}}
    tools = []
    for name, description in (
        ('intent__HassTurnOn', '打开指定的设备。'),
        ('intent__HassTurnOff', '关闭指定的设备。'),
        ('light__HassLightSet', '调整灯光；brightness 为百分之零到一百。'),
        ('homeassistant__GetLiveContext', '查询家居设备的当前状态。'),
    ):
        props = {} if name.endswith('GetLiveContext') else deepcopy(target)
        if name == 'light__HassLightSet':
            props.update(brightness={'type': 'number', 'minimum': 0, 'maximum': 100},
                         color={'type': 'string'}, temperature={'type': 'integer'})
        tools.append({'name': name, 'description': description,
                      'input_schema': {'type': 'object', 'properties': props, 'additionalProperties': False}})
    # 静态目录只说明有哪些设备；实时状态只能通过查询工具获得。
    catalog_devices = [{k: v for k, v in device.items() if k in {'name', 'domain', 'area', 'floor'}}
                       for device in devices]
    return tools, [{'type': 'text', 'text': ALPHA_RULES + '\n可用设备：\n' +
                    json.dumps(catalog_devices, ensure_ascii=False)}]


class FakeServices:
    """唯一外部执行端。使用本地数据；没有任何真实网络的回退路径。"""

    def __init__(self, context, clock):
        self.closed = False
        self.clock = clock
        self.devices = deepcopy(context.get('devices', [
            {'name': '客厅灯', 'domain': 'light', 'area': '客厅'},
            {'name': '厨房灯', 'domain': 'light', 'area': '厨房'}]))
        for device in self.devices:
            device.setdefault('state', 'off')
        self.items = [{'id': 'task-milk', 'title': '牛奶', 'status': 'needsAction'},
                      {'id': 'task-eggs', 'title': '鸡蛋', 'status': 'needsAction'}]
        self.documents: dict[str, str] = {}
        self.reminders: list[dict] = []
        self.drafts: dict[str, dict] = {}
        self.results: dict = {}
        self.effects: Counter = Counter()
        self.executions: list[dict] = []
        self.errors: list[str] = []

    def matching_devices(self, args):
        """返回目标条件的交集；结果描述和状态更新必须共用这份匹配逻辑。"""
        return [device for device in self.devices
                if all(not args.get(key) or device.get(key) == args[key]
                       for key in ('name', 'area', 'floor'))
                and (not args.get('domain') or device.get('domain') in args['domain'])]

    @staticmethod
    def result(data):
        return CallToolResult(content=[TextContent(type='text', text=data.get('text', ''))],
                              structured_content=data.get('structured_content'),
                              is_error=data.get('is_error', False))

    def default(self, name, args):
        if name == 'prepare_email':
            tool, values = args['tool_name'], args['arguments']
            if tool != 'send_email':
                raise ToolNotSent('eval 尚未实现回复邮件的草稿模拟')
            token = uuid4().hex
            preview = {key: values[key] for key in ('to', 'subject', 'body')}
            self.drafts[token] = {'arguments': deepcopy(values), 'used': False, 'tool': tool}
            return {'text': '草稿已准备，尚未发送', 'structured_content': {
                'preview': preview, 'confirmation_token': token}}
        if name in {'send_email', 'reply_email'}:
            draft = self.drafts.get(args.get('confirmation_token'))
            values = {k: v for k, v in args.items() if k not in {'confirmed', 'confirmation_token'}}
            if (not draft or draft['used'] or args.get('confirmed') is not True
                    or draft['tool'] != name or draft['arguments'] != values):
                raise ToolNotSent('假发送端拒绝：没有匹配且未使用的确认快照')
            draft['used'] = True
            return {'text': '邮件已发送', 'structured_content': {'message_id': 'sent-eval-01'}}
        if name == 'web_search':
            return {'text': '虚构搜索结果：Springfield 的示例中餐馆，仅用于测试工具选择。'}
        if name in HA_DEVICE_TOOLS:
            if name.endswith('GetLiveContext'):
                return {'text': json.dumps({'devices': self.devices}, ensure_ascii=False)}
            matches = self.matching_devices(args)
            if not matches:
                return {'text': '找不到设备', 'is_error': True}
            return {'text': json.dumps({'response_type': 'action_done',
                    'data': {'success': matches, 'failed': []}}, ensure_ascii=False)}
        if name == 'add_items':
            return {'text': '已添加到 ' + args['list_name'] + '：' + '、'.join(args['items'])}
        if name == 'list_items':
            return {'text': args['list_name'] + '：\n' + '\n'.join(
                f"- {item['title']}：{'已完成' if item['status'] == 'completed' else '未完成'}"
                for item in self.items), 'structured_content': {'items': deepcopy(self.items)}}
        if name == 'complete_items':
            return {'text': '已标为完成：' + '、'.join(args['item_titles'])}
        if name == 'create_reminder':
            local = datetime.fromisoformat(args['local_time'])
            if local.utcoffset() is not None:
                raise ToolNotSent('local_time 必须不带时区')
            zone = ZoneInfo(os.environ['REMINDER_TIMEZONE'])
            aware = local.replace(tzinfo=zone)
            if aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != local:
                raise ToolNotSent('本地时间不存在')
            return {'text': '已创建提醒：' + args['title'], 'structured_content': {
                'id': f'reminder-{len(self.reminders)+1}', 'start': aware.isoformat()}}
        if name == 'list_reminders':
            return {'text': '暂无提醒' if not self.reminders else json.dumps(self.reminders, ensure_ascii=False)}
        if name == 'get_weather':
            day = '明天' if args.get('day', 'today') == 'tomorrow' else '今天'
            return {'text': f'{day}多云，12 到 19 度，降雨概率 30%' +
                    ('，现在 15 度，体感 13 度' if day == '今天' else '')}
        if name == 'create_document':
            return {'text': '已创建文档：' + args['title'], 'structured_content': {
                'document_id': f'doc-eval-{len(self.documents)+1:02}', 'title': args['title']}}
        if name == 'append_document':
            if args['document_id'] not in self.documents:
                raise ToolNotSent('文档 ID 尚未由创建工具返回')
            return {'text': '文档已追加'}
        if name == 'read_email':
            return {'text': '发件人：sender@example.com\n主题：会议\n正文：明天九点开会。'}
        raise ToolNotSent(f'eval 未实现此工具的默认模拟：{name}')

    async def call(self, name, arguments):
        args = deepcopy(arguments)
        public_args = {k: v for k, v in args.items() if k != 'confirmation_token'}
        self.executions.append({'name': name, 'arguments': public_args})
        self.effects[f'{name}_executions'] += 1
        try:
            # 确认资格和文档依赖不能被 tool_results 的成功文本绕过。
            if name in {'send_email', 'reply_email', 'append_document'}:
                data = self.default(name, args)
                data.update(deepcopy(self.results.get(name, {})))
            elif name in self.results:
                data = deepcopy(self.results[name])
            else:
                data = self.default(name, args)
            result = self.result(data)
            if result.is_error:
                return result
            if name == 'add_items':
                for title in args['items']:
                    self.items.append({'id': f'task-{uuid4().hex}', 'title': title, 'status': 'needsAction'})
                self.effects['items_created'] += len(args['items'])
            elif name == 'complete_items':
                for item in self.items:
                    if item['title'] in args['item_titles']:
                        item['status'] = 'completed'
            elif name == 'create_document':
                self.documents[data['structured_content']['document_id']] = ''
            elif name == 'append_document':
                self.documents[args['document_id']] += args['text']
            elif name == 'create_reminder':
                self.reminders.append(deepcopy(data))
            elif name in {'send_email', 'reply_email'}:
                self.effects['emails_sent'] += 1
            elif name in HA_DEVICE_TOOLS:
                for device in self.matching_devices(args):
                    if name.endswith('HassTurnOff'):
                        device['state'] = 'off'
                    elif name.endswith('HassTurnOn'):
                        device['state'] = 'on'
                    elif name.endswith('HassLightSet'):
                        device.update(state='on', brightness=args.get('brightness'))
            return result
        except (KeyError, TypeError, ValueError, ToolNotSent) as exc:
            self.errors.append(f'{name}: {exc}')
            raise ToolNotSent(str(exc)) from exc

    async def call_ha(self, name, arguments, **kwargs):
        return await self.call(name, arguments)


class EvalBrain(agent.AlphaBrain):
    """只替换启动依赖；ask 和完整 agent 循环保持原样。"""

    def __init__(self, context, client, services, clock, model, *, db_path=':memory:', auto_memory=False):
        super().__init__(model=model, style='text')
        self.db_path = db_path
        self.context = context
        self.test_client = client
        self.services = services
        self.clock = clock
        self.enable_auto_memory = auto_memory
        # 后台 Haiku 必须与主模型的工具调用记录隔离。
        self.extraction_recorder = RecordingMessages(client.messages.delegate)
        self.extraction_records = []

    async def start(self):
        self.model = self._model_override
        self.reminder_tz = ZoneInfo(self.context.get('timezone', 'America/Los_Angeles'))
        self.personal_server = SimpleNamespace()  # ask 的已启动检查，不产生子进程。
        self.http_client = SimpleNamespace(is_closed=False)
        self.personal_tools = await personal_catalog()
        self.ha_catalog = ha_catalog(self.services.devices)
        self.store = operations.OperationStore(self.db_path)
        self.memory_store = agent.MemoryStore(self.db_path)
        self.worker = self.services
        self.ha_url = 'https://ha.invalid/api/mcp'
        mcp_executor = make_executor(ha_url=self.ha_url, http_client=self.http_client, worker=self.worker, ha_tools=self.ha_tools)

        # ExecutorResult 从实际网关模块导入，避免依赖大脑模块的内部导出。
        from integrations.tool_gateway import ExecutorResult

        async def execute(name, args):
            if name == 'web_search':
                result = await self.services.call(name, args)
                return ExecutorResult(not result.is_error, result.content[0].text)
            return await mcp_executor(name, args)

        self.gateway = ToolGateway(self.store, execute, ha_tools=self.ha_tools)
        self.email_confirmation = EmailConfirmation(self.gateway, clock=self.clock.monotonic)
        self._prepare_catalog()
        # 保存真正配置构造器的地点用于评分，再换成普通 query 工具。
        self.test_client.messages.search_location = deepcopy(self.web_search.get('user_location'))
        self.web_search = deepcopy(SEARCH_TOOL)
        self.tools.append(deepcopy(SEARCH_TOOL))
        self.tool_owners['web_search'] = 'eval_search'
        self.client = self.test_client
        if self.enable_auto_memory:
            extraction_client = SimpleNamespace(messages=self.extraction_recorder)
            self.auto_memory = AutoMemory(self.memory_store, extraction_client, self._lock,
                clock=self.clock.monotonic,
                extractor_factory=lambda client: RecordingExtractor(client, self.extraction_records))
            self._refresh_memory_system()
        self.messages = deepcopy(self.context.get('history', []))
        if contacts := self.context.get('known_contacts'):
            self.system[0]['text'] += '\n已知联系人：' + json.dumps(contacts, ensure_ascii=False)
        self._started = True

    async def close(self):
        store = self.store
        memory_store = self.memory_store
        try:
            await super().close()
        finally:
            if store is not None:
                store.close()
            if memory_store is not None:
                memory_store.close()


def load_cases(path: Path) -> list[dict]:
    rows, ids = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get('id'), str) or row['id'] in ids:
            raise ValueError(f'第 {line_number} 行 ID 缺失或重复')
        if ('input' in row) == ('turns' in row):
            raise ValueError(f"{row['id']}: input / turns 必须二选一")
        memories = row.get('context', {}).get('memories', [])
        if not isinstance(memories, list) or any(
            not isinstance(m, dict) or m.get('kind') not in ('preference', 'fact')
            or not isinstance(m.get('content'), str) or not m['content'].strip() for m in memories
        ):
            raise ValueError(f"{row['id']}: context.memories 必须含合法 kind 和非空 content")
        if 'requires_auto_memory' in row and type(row['requires_auto_memory']) is not bool:
            raise ValueError(f"{row['id']}: requires_auto_memory 必须是布尔值")
        if 'turns' in row and not row['turns']:
            raise ValueError(f"{row['id']}: turns 不能为空")
        for turn in row.get('turns', [row]):
            if not isinstance(turn.get('input'), str) or not turn['input'].strip():
                raise ValueError(f"{row['id']}: input 不能为空")
            if 'new_session' in turn and type(turn['new_session']) is not bool:
                raise ValueError(f"{row['id']}: new_session 必须是布尔值")
            memory_effects = turn.get('expect_effects', {})
            if 'memories_count' in memory_effects and (type(memory_effects['memories_count']) is not int or memory_effects['memories_count'] < 0):
                raise ValueError(f"{row['id']}: memories_count 必须为非负整数")
            for source_count in ('auto_memories_count', 'explicit_memories_count', 'pending_memories_count', 'auto_extraction_calls'):
                if source_count in memory_effects and (type(memory_effects[source_count]) is not int or memory_effects[source_count] < 0):
                    raise ValueError(f"{row['id']}: {source_count} 必须为非负整数")
            for flag in ('expect_auto_notice', 'expect_auto_question'):
                if flag in turn and type(turn[flag]) is not bool:
                    raise ValueError(f"{row['id']}: {flag} 必须是布尔值")
            if 'memory_contains' in memory_effects and (not isinstance(memory_effects['memory_contains'], list) or not all(isinstance(v, str) and v for v in memory_effects['memory_contains'])):
                raise ValueError(f"{row['id']}: memory_contains 必须为非空字符串的列表")
            required, allowed, forbidden = (set(turn.get(k, [])) for k in ('expect_tools', 'allow_tools', 'must_not'))
            if 'expect_tools' not in turn or (required | allowed) & forbidden or required & allowed:
                raise ValueError(f"{row['id']}: 工具规则缺失或冲突")
            if 'reply_rubric' in turn and (not isinstance(turn['reply_rubric'], str)
                                          or not turn['reply_rubric'].strip()):
                raise ValueError(f"{row['id']}: reply_rubric 必须是非空字符串")
            supported = {'expect_tools', 'expect_args', 'expect_args_by_tool',
                         'expect_reply_contains', 'expect_reply_contains_any',
                         'expect_reply_not_contains', 'expect_effects', 'expect_auto_notice', 'expect_auto_question'}
            unknown = {key for key in turn if key.startswith('expect_')} - supported
            if unknown:
                raise ValueError(f"{row['id']}: 未实现的评分字段：{sorted(unknown)}")
            for key in turn.get('expect_effects', {}):
                if key not in {'emails_sent', 'confirmation_required', 'items_created', 'memories_count', 'memory_contains', 'auto_memories_count', 'explicit_memories_count', 'pending_memories_count', 'auto_extraction_calls'} and not (
                    key.endswith('_executions') and key.removesuffix('_executions') in {*TOOL_POLICY, 'web_search'}
                ):
                    raise ValueError(f"{row['id']}: 未实现的副作用计数：{key}")
        ids.add(row['id'])
        rows.append(row)
    return rows


def arguments_match(expected, actual, schema):
    if not isinstance(actual, dict):
        return False
    actual = deepcopy(actual)
    for key, prop in schema.get('properties', {}).items():
        if key not in actual and 'default' in prop:
            actual[key] = prop['default']
    for key, value in expected.items():
        if key == 'text_contains':
            # 列表形式兼容文档 text；字典形式可指定邮件 body 等自由文本字段。
            fields = value if isinstance(value, dict) else {'text': value}
            for field, snippets in fields.items():
                if (not isinstance(actual.get(field), str)
                        or not all(s in actual[field] for s in snippets)):
                    return False
        elif key not in actual or type(actual[key]) is not type(value) or actual[key] != value:
            # int/float 的 JSON 数字相等；布尔值不能冒充 0/1。
            if not (type(value) in (int, float) and type(actual.get(key)) in (int, float)
                    and actual[key] == value):
                return False
    return True


def score_turn(expected, observed, schemas, context):
    if observed.get('error'):
        # API/超时等运行错误不混成“模型没有选对工具”。
        return ['运行错误：' + observed['error']]
    errors = []
    calls = [c for c in observed['tool_calls'] if c['name'] not in INTERNAL_OUTPUTS]
    names = {c['name'] for c in calls}
    required, allowed = set(expected['expect_tools']), set(expected.get('allow_tools', []))
    for name in sorted(required - names):
        errors.append(f'缺少必需工具：{name}')
    for name in sorted(names - required - allowed):
        errors.append(f'调用了未允许工具：{name}')
    for name in sorted(names & set(expected.get('must_not', []))):
        errors.append(f'调用了明确禁止工具：{name}')
    arg_rules = dict(expected.get('expect_args_by_tool', {}))
    if 'expect_args' in expected:
        if len(required) != 1:
            errors.append('expect_args 仅适用于一个必需工具')
        else:
            arg_rules[next(iter(required))] = expected['expect_args']
    for call in calls:
        schema = schemas.get(call['name'])
        if schema is None:
            errors.append(f"没有工具 schema：{call['name']}")
            continue
        for problem in Draft202012Validator(schema).iter_errors(call['arguments']):
            errors.append(f"{call['name']} 参数不符合 schema：{problem.message}")
        if call['name'] in arg_rules and not arguments_match(arg_rules[call['name']], call['arguments'], schema):
            errors.append(f"{call['name']} 参数不匹配：{json.dumps(call['arguments'], ensure_ascii=False)}")
        if call['type'] == 'server_tool_use':
            errors.append('eval 意外调用了真正的服务端工具')
    if expected.get('reply_rubric') and observed.get('reply_scoring') == 'judge':
        if observed.get('judge_error'):
            errors.append('评判器运行错误：' + observed['judge_error'])
        elif not observed['reply_judge']['pass']:
            errors.append('回答语义不符合标准：' + observed['reply_judge']['reason'])
    else:
        answer = observed['answer']
        for word in expected.get('expect_reply_contains', []):
            if word not in answer:
                errors.append(f'回答缺少：{word}')
        any_words = expected.get('expect_reply_contains_any')
        if any_words and not any(word in answer for word in any_words):
            errors.append('回答未命中任一可接受表述：' + ' / '.join(any_words))
        for word in expected.get('expect_reply_not_contains', []):
            if word in answer:
                errors.append(f'回答包含禁止表述：{word}')
    for flag, observed_key in (('expect_auto_notice', 'auto_notice_in_answer'),
                               ('expect_auto_question', 'auto_question_at_end')):
        if flag in expected and observed.get(observed_key, False) is not expected[flag]:
            errors.append(f'{flag}：预期 {expected[flag]}，实际 {observed.get(observed_key, False)}')
    for key, value in expected.get('expect_effects', {}).items():
        actual = observed['effects'].get(key, [] if key == 'memory_contains' else 0)
        if key == 'memory_contains':
            for snippet in value:
                if not any(snippet in content for content in actual):
                    errors.append(f'记忆缺少：{snippet}')
        elif actual != value:
            errors.append(f'副作用 {key}：预期 {value}，实际 {actual}')
    if context.get('home_city'):
        location = {'type': 'approximate', 'city': context['home_city'],
                    'timezone': context.get('timezone', 'America/Los_Angeles')}
        location.update({k: context['home_' + k] for k in ('region', 'country') if context.get('home_' + k)})
        if not observed['search_locations'] or any(x != location for x in observed['search_locations']):
            errors.append('搜索地点配置不符合 context')
    errors.extend(observed.get('harness_errors', []))
    return errors


def seed_operations(store, context):
    for row in context.get('operations', []):
        op = store.create('seed-' + uuid4().hex, row['name'], row['arguments'])
        status = row['status']
        path = {'pending': [], 'approved': ['approved'], 'cancelled': ['cancelled'],
                'executing': ['approved', 'executing'],
                'succeeded': ['approved', 'executing', 'succeeded'],
                'failed': ['approved', 'executing', 'failed'],
                'unknown': ['approved', 'executing', 'unknown']}[status]
        previous = 'pending'
        for current in path:
            assert store.transition(op.id, previous, current, result=row.get('result'))
            previous = current
        updated = datetime.fromisoformat(row['updated_at']).timestamp()
        # 仅测试库，公开 API 不支持指定历史时间；不改生产 store 的实现。
        store._db.execute('UPDATE operations SET created_at=?, updated_at=? WHERE id=?',
                          (updated, updated, op.id))
        store._db.commit()


async def run_case(case, sdk_client, *, model: str, timeout_s: float = 90, judge=None, auto_memory=False) -> dict:
    context = deepcopy(case.get('context', {}))
    clock = FrozenClock(context.get('now', DEFAULT_NOW))
    services = FakeServices(context, clock)
    recorder = RecordingMessages(sdk_client.messages)
    env = {'HOME_CITY': context.get('home_city', ''), 'HOME_REGION': context.get('home_region', ''),
           'HOME_COUNTRY': context.get('home_country', ''),
           'REMINDER_TIMEZONE': context.get('timezone', 'America/Los_Angeles'),
           'ALPHA_AUTO_MEMORY': '1' if auto_memory else '0'}
    output = {'id': case['id'], 'model': model, 'turns': [], 'manual_checks': [], 'auto_memory': auto_memory}
    async with AsyncExitStack() as stack:
        directory = stack.enter_context(TemporaryDirectory(prefix='alpha-eval-'))
        db_path = Path(directory) / 'operations.db'
        brain = EvalBrain(context, SimpleNamespace(messages=recorder), services, clock, model, db_path=db_path, auto_memory=auto_memory)
        stack.enter_context(patch.dict(os.environ, env))
        stack.enter_context(patch.dict(TOOL_POLICY, {'web_search': ToolPolicy(False, True, 'immediate')}))
        stack.enter_context(patch.object(agent, 'datetime', clock.datetime_type()))
        stack.enter_context(patch.object(operations, 'time', SimpleNamespace(time=clock.time)))
        stack.enter_context(patch.object(memory_module, 'time', SimpleNamespace(time=clock.time)))
        stack.enter_context(patch('integrations.tool_executor.call_ha_tool', side_effect=services.call_ha))
        # 完整生命期均不允许真实发现连接或启动 worker。
        stack.enter_context(patch.object(agent, 'load_ha_catalog', side_effect=RuntimeError('eval 禁止 HA 发现')))
        stack.enter_context(patch.object(agent, 'load_personal_catalog', side_effect=RuntimeError('eval 禁止 MCP 发现')))
        stack.push_async_callback(brain.close)
        await brain.start()
        assert brain.email_confirmation is not None
        confirmation = brain.email_confirmation
        seed_operations(brain.store, context)
        assert brain.memory_store is not None
        for memory in context.get('memories', []):
            brain.memory_store.add(memory['content'], memory['kind'], 'eval-seed')
        if draft := context.get('pending_email'):
            confirmation.begin_turn('seed-email')
            decision = await confirmation.handle(draft['name'], draft['arguments'])
            if not decision.preview:
                raise ValueError('无法注入 pending_email')
            assert confirmation.draft is not None
            confirmation.draft.created_at -= draft.get('age_s', 0)
            brain.messages.append({'role': 'assistant', 'content': decision.text})
        for index, turn in enumerate(case.get('turns', [case]), 1):
            if turn.get('new_session') and index > 1:
                await brain.close()
                fresh_context = {**context, 'history': []}
                brain = EvalBrain(fresh_context, SimpleNamespace(messages=recorder), services,
                                  clock, model, db_path=db_path, auto_memory=auto_memory)
                stack.push_async_callback(brain.close)
                await brain.start()
                assert brain.email_confirmation is not None
                confirmation = brain.email_confirmation
            services.results = {**case.get('tool_results', {}), **turn.get('tool_results', {})}
            start_calls, start_responses = len(recorder.calls), len(recorder.responses)
            start_exec, start_errors = len(services.executions), len(services.errors)
            start_history = len(brain.messages)
            start_locations = len(recorder.request_locations)
            before = services.effects.copy()
            start_extraction = len(brain.extraction_records)
            start_extraction_requests = len(brain.extraction_recorder.responses)
            assert brain.memory_store is not None
            memory_before_answer = brain.memory_store.list()
            notice = False
            question = brain.auto_memory.question(mark=False) if brain.auto_memory is not None else ''
            started = time.perf_counter()
            answer, exception = '', None
            main_answer_s = 0.0
            try:
                async with asyncio.timeout(timeout_s):
                    answer = await brain.ask(turn['input'])
                    main_answer_s = time.perf_counter() - started
                    assert brain.memory_store is not None
                    memory_after_answer = brain.memory_store.list()
                    newly_auto = [m for m in memory_after_answer if m.source == 'auto' and m not in memory_before_answer]
                    notice = bool(newly_auto and agent.claims_memory_saved(answer)
                                  and all(m.content in answer for m in newly_auto))
                    if brain.auto_memory is not None:
                        question = next((f'要我记住“{p.candidate.content}”吗？' for p in brain.auto_memory.pending
                                         if p.asked and p.asked_turn == brain.auto_memory.turn), '')
                    await drain_auto_memory(brain)
                    if any(record.get('error') for record in brain.extraction_records[start_extraction:]):
                        raise RuntimeError('后台记忆提取失败，不能以没有候选作为通过依据')
                    if auto_memory and not case.get('requires_auto_memory'):
                        assert brain.auto_memory is not None
                        if newly_auto:
                            services.errors.append('原有业务用例产生了非预期的自动记忆变更')
                        if brain.auto_memory.pending:
                            services.errors.append('原有业务用例产生了非预期的敏感记忆候选')
                        if notice or question or '对了，我记下了' in answer:
                            services.errors.append('原有业务用例多出了自动记忆通知或提问')
            except Exception as exc:
                exception = f'{type(exc).__name__}: {exc}'
            effects: dict[str, object] = {k: services.effects[k] - before[k] for k in services.effects.keys() | before.keys()}
            draft = confirmation.draft
            waiting = bool(draft and not draft.used)
            effects['confirmation_required'] = waiting
            assert brain.memory_store is not None
            memories = [vars(memory) for memory in brain.memory_store.list()]
            effects['memories_count'] = len(memories)
            effects['memory_contains'] = [memory['content'] for memory in memories]
            effects['auto_memories_count'] = sum(m['source'] == 'auto' for m in memories)
            effects['explicit_memories_count'] = sum(m['source'] == 'explicit' for m in memories)
            effects['pending_memories_count'] = len(brain.auto_memory.pending) if brain.auto_memory is not None else 0
            effects['auto_extraction_calls'] = sum(r['model_calls'] for r in brain.extraction_records[start_extraction:])
            if brain.last_response_fallback:
                services.errors.append('模型没有生成可见回答，最终使用了程序兜底提示')
            semantic_judge = judge is not None and bool(turn.get('reply_rubric'))
            if waiting and not semantic_judge and not any(s in answer for s in ('确认', '发送吗', '要发吗', '发出吗')):
                services.errors.append('存在待确认草稿，但回答没有询问用户确认')
            observed = {'index': index, 'input': turn['input'], 'answer': answer,
                        'new_session': turn.get('new_session', False), 'memories': memories,
                        'auto_notice_in_answer': bool(notice),
                        'auto_question_at_end': bool(question and answer.endswith(question)),
                        'auto_memory': {'extractions': deepcopy(brain.extraction_records[start_extraction:]),
                            'responses': deepcopy(brain.extraction_recorder.responses[start_extraction_requests:]),
                            'pending': ([p.candidate.to_dict() for p in brain.auto_memory.pending]
                                        if brain.auto_memory is not None else []),
                            'main_answer_s': main_answer_s,
                            'drain_s': round(time.perf_counter() - started - main_answer_s, 3)},
                        'tool_calls': deepcopy(recorder.calls[start_calls:]),
                        'tool_records': tool_records_from_messages(brain.messages[start_history:]),
                        'executions': deepcopy(services.executions[start_exec:]),
                        'effects': effects, 'responses': deepcopy(recorder.responses[start_responses:]),
                        'search_locations': deepcopy(recorder.request_locations[start_locations:]),
                        'harness_errors': services.errors[start_errors:], 'error': exception,
                        'response_fallback': brain.last_response_fallback,
                        'elapsed_s': round(time.perf_counter() - started, 3)}
            observed['reply_scoring'] = 'judge' if semantic_judge else 'literal'
            if semantic_judge and exception is None:
                try:
                    observed['reply_judge'] = await judge.evaluate(
                        answer, turn['reply_rubric'], tool_records=observed['tool_records'])
                    observed['judge_model'] = judge.model
                    observed['judge_usage'] = judge.last_usage
                except Exception as exc:
                    observed['judge_error'] = f'{type(exc).__name__}: {exc}'
            observed['failures'] = score_turn(turn, observed, recorder.schemas, context)
            observed['passed'] = not observed['failures']
            output['turns'].append(observed)
            if turn.get('notes'):
                output['manual_checks'].append(f'第 {index} 轮：' + turn['notes'])
            if exception:
                # 历史可能不完整；不把下一轮在坏状态上继续跑出的错误当成独立结果。
                output['aborted'] = True
                break
    if case.get('notes') and case.get('turns'):
        output['manual_checks'].append(case['notes'])
    elif case.get('notes') and not output['manual_checks']:
        output['manual_checks'].append(case['notes'])
    output['passed'] = len(output['turns']) == len(case.get('turns', [case])) and all(t['passed'] for t in output['turns'])
    output['status'] = ('error' if any(t['error'] or t.get('judge_error') for t in output['turns'])
                        else 'pass' if output['passed'] else 'fail')
    return output


def summarize_results(results):
    """以完整用例为计数单位；多轮中任何一轮失败，这次尝试就不通过。"""
    grouped = {}
    for result in results:
        grouped.setdefault(result['id'], []).append(result)
    cases = []
    for ident, attempts in grouped.items():
        passed = sum(result['passed'] for result in attempts)
        cases.append({'id': ident, 'passed': passed, 'total': len(attempts),
                      'stable': len(attempts) >= 3 and passed == len(attempts),
                      'failed_attempts': [result['attempt'] for result in attempts if not result['passed']]})
    total = len(results)
    passed = sum(result['passed'] for result in results)
    return {'passed': passed, 'total': total, 'pass_rate': passed / total if total else 0,
            'failed': sum(r['status'] == 'fail' for r in results),
            'errors': sum(r['status'] == 'error' for r in results), 'cases': cases}


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('cases.jsonl'))
    parser.add_argument('--case', action='append', dest='case_ids', help='只运行指定 ID，可重复指定')
    parser.add_argument('--model', help='默认取 ANTHROPIC_MODEL')
    parser.add_argument('--output', type=Path, help='报告 JSONL 路径，默认 eval/results/时间戳.jsonl')
    parser.add_argument('--timeout', type=float, default=90, help='每轮最长秒数')
    parser.add_argument('--repeat', type=int, default=1, help='每条用例独立重复次数，默认 1')
    parser.add_argument('--auto-memory', action='store_true', help='启用真实后台提取，逐轮等待完成并检查持久化效果')
    parser.add_argument('--judge', action='store_true', help='先校准评判模型，再对 reply_rubric 用例做语义评分')
    parser.add_argument('--judge-model', default='claude-haiku-5-5')
    parser.add_argument('--list', action='store_true', help='只检查数据并列出 ID，不调用模型')
    args = parser.parse_args(argv)
    cases = load_cases(args.cases)
    if args.case_ids:
        missing = set(args.case_ids) - {c['id'] for c in cases}
        if missing:
            parser.error('不存在的用例：' + ', '.join(sorted(missing)))
        cases = [c for c in cases if c['id'] in args.case_ids]
    if args.timeout <= 0:
        parser.error('--timeout 必须大于零')
    if args.repeat < 1:
        parser.error('--repeat 必须是正整数')
    if args.list:
        for case in cases:
            print(case['id'])
        return 0
    if not args.auto_memory:
        unavailable = [c['id'] for c in cases if c.get('requires_auto_memory')]
        if unavailable and args.case_ids:
            parser.error('这些用例需要 --auto-memory：' + ', '.join(unavailable))
        cases = [c for c in cases if not c.get('requires_auto_memory')]
    load_dotenv(ROOT / '.env')
    if not os.getenv('ANTHROPIC_API_KEY', '').strip():
        parser.error('请在 alpha/.env 设置 ANTHROPIC_API_KEY')
    model = args.model or os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-5')
    output = args.output or ROOT / 'eval/results' / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:6] + '.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_suffix('.summary.json')
    calibration_path = output.with_suffix('.calibration.json')
    if output.exists() or summary_path.exists() or (args.judge and calibration_path.exists()):
        parser.error('报告或汇总文件已存在，请换一个 --output 路径')
    logging.basicConfig(level=logging.WARNING)
    results = []
    # 使用官方 SDK；禁用自动重试以便成本和失败可见，不使用 .env 自定义代理端点。
    async with AsyncAnthropic(base_url='https://api.anthropic.com', max_retries=0) as client:
        judge = None
        if args.judge:
            from eval.judge import ReplyJudge
            from eval.calibrate_judge import calibrate, save_report
            judge = ReplyJudge(client, model=args.judge_model)
            calibration = await calibrate(judge)
            save_report(calibration, calibration_path)
            print(f"评判器校准：{calibration['correct']}/{calibration['total']}；报告：{calibration_path}", flush=True)
            if not calibration['passed']:
                print('校准未通过，停止运行，不使用该评判器评分。')
                return 2
        with output.open('x', encoding='utf-8') as report:
            for case in cases:
                for attempt in range(1, args.repeat + 1):
                    # 每次都完整新建大脑、数据库、工具状态和历史，不只重放一次 ask。
                    try:
                        result = await run_case(case, client, model=model, timeout_s=args.timeout, judge=judge, auto_memory=args.auto_memory)
                    except Exception as exc:
                        result = {'id': case['id'], 'passed': False, 'status': 'error',
                                  'error': f'{type(exc).__name__}: {exc}', 'turns': []}
                    result.update(attempt=attempt, repeat=args.repeat)
                    results.append(result)
                    report.write(json.dumps(result, ensure_ascii=False) + '\n')
                    report.flush()
                    print(f"{result['status'].upper()} {case['id']} [{attempt}/{args.repeat}]", flush=True)
                    if result.get('error'):
                        print('  ' + result['error'], flush=True)
                    for turn in result['turns']:
                        for reason in turn['failures']:
                            print(f"  第{turn['index']}轮：{reason}", flush=True)
    summary = summarize_results(results)
    summary.update(model=model, repeat=args.repeat, report=str(output), auto_memory=args.auto_memory,
                   reply_scoring='judge' if args.judge else 'literal',
                   judge_model=args.judge_model if args.judge else None)
    with summary_path.open('x', encoding='utf-8') as saved:
        json.dump(summary, saved, ensure_ascii=False, indent=2)
        saved.write('\n')
    print('\n按用例汇总：')
    for row in summary['cases']:
        label = '本次稳定' if row['stable'] else '未全通过' if row['failed_attempts'] else '样本不足 3 次'
        failures = f"；失败尝试 {row['failed_attempts']}" if row['failed_attempts'] else ''
        print(f"{row['id']:<12} {row['passed']}/{row['total']}  {label}{failures}")
    print(f"总计：{summary['passed']}/{summary['total']}（{summary['pass_rate']:.1%}）"
          f" ｜ 行为失败 {summary['failed']} ｜ 运行错误 {summary['errors']}")
    print(f'详细记录：{output}\n汇总：{summary_path}')
    print('notes 中的语义要求仍需人工复查；搜索是普通工具替身，不评价真实搜索质量。')
    return 2 if summary['errors'] else 1 if summary['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
