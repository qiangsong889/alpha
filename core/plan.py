"""待确认计划的纯逻辑：不执行工具、不修改操作记录，也不管理播放或用户确认。"""

from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import Any, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from integrations.operations import Operation
from integrations.tool_policy import get_policy


PlanStatus = Literal["collecting", "awaiting_confirmation", "executing", "done", "cancelled"]


def action_count(op: Operation) -> int:
    """统计请求中的条目数；同名任务实际会匹配几条，要执行时才知道。"""
    if get_policy(op.name) is None:
        raise ValueError(f"工具不在策略白名单中：{op.name}")
    field = {"add_items": "items", "complete_items": "item_titles"}.get(op.name)
    if field is None:
        return 1
    items = op.arguments.get(field)
    if (not isinstance(items, list) or not items
            or any(not isinstance(item, str) or not item.strip() for item in items)):
        raise ValueError(f"{op.name} 的 {field} 必须是非空标题列表")
    # 不排序或去重：重复元素也是请求中的条目。
    return len(items)


def _validate_pending(op: Operation) -> None:
    policy = get_policy(op.name)
    if policy is None:
        raise ValueError(f"工具不在策略白名单中：{op.name}")
    if op.status != "pending" or not policy.mutates or policy.commit == "immediate":
        raise ValueError("计划只接受 pending 的 deferred/confirm 写操作")


def validate_arguments(op: Operation) -> None:
    """在保存计划项之前校验模板需要的参数，不把坏数据留到复述阶段。"""
    action_count(op)
    args = op.arguments
    required_text = {
        "add_items": ("list_name",),
        "complete_items": ("list_name",),
        "create_reminder": ("title", "local_time"),
        "delete_task": ("list_id", "task_id"),
        "delete_reminder": ("event_id",),
    }.get(op.name, ())
    for field in required_text:
        value = args.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{op.name} 的 {field} 必须是非空字符串")
    if op.name == "create_reminder":
        minutes = args.get("minutes_before", 0)
        if type(minutes) is not int or not 0 <= minutes <= 40320:
            raise ValueError("minutes_before 必须是 0 到 40320 的整数")
        text = args["local_time"]
        try:
            local = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("local_time 格式无效，例如 2026-10-16T08:00:00") from exc
        if local.tzinfo is not None:
            raise ValueError("local_time 必须是不带时区的本地时间")
        # 与个人 server 的格式约定一致，不能只接受日期而悄悄复述成凌晨。
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?", text):
            raise ValueError("local_time 必须包含日期和时分秒，例如 2026-10-16T08:00:00")


def needs_plan(ops: Iterable[Operation]) -> bool:
    """只判断尚未提交的写操作；先校验所有项，避免短路时漏掉未知工具。"""
    total, requires_confirmation = 0, False
    for op in ops:
        _validate_pending(op)
        total += action_count(op)
        policy = get_policy(op.name)
        requires_confirmation |= policy is not None and policy.commit == "confirm"
    return requires_confirmation or total >= 2


@dataclass(frozen=True)
class PlanItem:
    item_id: int
    op_id: str
    name: str
    arguments: dict[str, Any]


def _key(item: PlanItem) -> str:
    # 忽略对象键顺序；列表顺序、重复元素和字符串内容保持原样。
    return json.dumps([item.name, item.arguments], sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


class Plan:
    def __init__(self, *, timezone_name: str = "America/Los_Angeles"):
        self.id = uuid4().hex
        self.version = 0
        self.status: PlanStatus = "collecting"
        # 5b 由调用方传入与个人 server 相同的配置；这里不读取环境变量。
        self._timezone = ZoneInfo(timezone_name)
        self._items: list[PlanItem] = []
        self._next_item_id = 1

    @property
    def items(self) -> tuple[PlanItem, ...]:
        """返回独立快照，外部修改嵌套参数也不会绕过版本变化。"""
        return tuple(deepcopy(self._items))

    @property
    def has_duplicates(self) -> bool:
        """留给 5b 判断是否还需澄清次数；不把一句“对”当作数量选择。"""
        keys = [_key(item) for item in self._items]
        return len(set(keys)) != len(keys)

    def _editable(self) -> bool:
        return self.status in {"collecting", "awaiting_confirmation"}

    @staticmethod
    def _snapshot(op: Operation, item_id: int) -> PlanItem:
        _validate_pending(op)
        validate_arguments(op)
        item = PlanItem(item_id, op.id, op.name, deepcopy(op.arguments))
        _key(item)  # 先确认能作为标准 JSON 比较；失败时不修改计划。
        return item

    def _changed(self) -> None:
        self.version += 1
        self.status = "collecting"

    def add(self, op: Operation) -> bool:
        if not self._editable() or any(item.op_id == op.id for item in self._items):
            return False
        self._items.append(self._snapshot(op, self._next_item_id))
        self._next_item_id += 1
        self._changed()
        return True

    def replace(self, index: int, new_op: Operation) -> bool:
        if not self._valid_index(index):
            return False
        if any(item.op_id == new_op.id for i, item in enumerate(self._items) if i != index):
            return False
        self._items[index] = self._snapshot(new_op, self._items[index].item_id)
        self._changed()
        return True

    def remove(self, index: int) -> bool:
        if not self._valid_index(index):
            return False
        del self._items[index]
        self._changed()
        return True

    def _valid_index(self, index: int) -> bool:
        # Python 接口从 0 开始；不接受 -1，也不把 True 当成第 2 项。
        return self._editable() and type(index) is int and 0 <= index < len(self._items)

    def recap(self, now: datetime) -> str:
        """只生成文字；5b 负责播放、改变状态和确认资格。"""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now 必须带时区信息")
        if not self._items:
            return "当前没有待执行的计划。"
        local_now = now.astimezone(self._timezone)
        keys = [_key(item) for item in self._items]
        counts = Counter(keys)
        last = {key: i for i, key in enumerate(keys)}
        parts = []
        for i, item in enumerate(self._items):
            # 圈号对应固定 item_id；删掉 #1 后，#2 仍复述为 ②，不会移位或复用编号。
            number = chr(ord("①") + item.item_id - 1) if item.item_id <= 20 else f"（{item.item_id}）"
            text = f"{number} {self.describe_item(item, local_now)}"
            key = keys[i]
            if counts[key] > 1 and last[key] == i:
                text += f"（这一项说了 {counts[key]} 次，要做 {counts[key]} 次还是 1 次？）"
            parts.append(text)
        return "我确认一下：" + "；".join(parts) + "。对吗？"

    def describe_item(self, item: PlanItem, now: datetime) -> str:
        args = item.arguments
        if item.name == "add_items":
            return f"把{'、'.join(args['items'])}加到{args['list_name']}"
        if item.name == "complete_items":
            return f"把{args['list_name']}里的{'、'.join(args['item_titles'])}标为已完成"
        if item.name == "create_reminder":
            local = datetime.fromisoformat(args["local_time"])
            days = (local.date() - now.date()).days
            full_date = f"{local.year:04d}年{local.month:02d}月{local.day:02d}日"
            date = {0: "今天", 1: "明天", 2: "后天"}.get(days, full_date)
            hour = local.hour
            period = ("凌晨" if hour < 6 else "早上" if hour < 12 else
                      "中午" if hour == 12 else "下午" if hour < 18 else "晚上")
            display_hour = hour if hour <= 12 else hour - 12
            clock = f"{display_hour}:{local.minute:02d}"
            if local.second or local.microsecond:
                clock += f":{local.second:02d}"
                if local.microsecond:
                    clock += f".{local.microsecond:06d}".rstrip("0")
            minutes = args.get("minutes_before", 0)
            advance = f"，提前 {minutes} 分钟提醒" if minutes else ""
            return f"{date}{period} {clock} 提醒你{args['title']}{advance}"
        return f"{item.name} " + json.dumps(args, ensure_ascii=False, sort_keys=True, allow_nan=False)
