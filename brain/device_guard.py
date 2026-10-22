"""同一轮里的设备操作防护：不让模型来回开关同一个设备。

模型可能在一轮里反复"操作 → 查状态 → 再操作"（例如刚开的灯状态还没上报，它看到旧状态又去改），
每一次都是真实的设备动作。规则只看本轮已经发出的调用，不管结果成功与否：
- 开/关类工具：同一个目标本轮只能改一次，开了再关、关了再开都拒绝；
- 其他写工具（调亮度、窗帘位置等）：同一工具 + 同一目标本轮只能调用一次；
- 开关灯类工具必须指定目标；不带任何目标就是"所有设备"；
- 本轮设备写操作达到上限后全部拒绝，让模型直接汇报。
被拒绝的调用不会发给 Home Assistant，也不登记操作。
"""

from __future__ import annotations

from typing import Any

POWER_TOOLS = frozenset({"intent__HassTurnOn", "intent__HassTurnOff"})
TARGETED_TOOLS = POWER_TOOLS | {"light__HassLightSet"}
TARGET_FIELDS = ("name", "area", "floor", "domain", "device_class")
MAX_DEVICE_WRITES = 8


class DeviceRefused(Exception):
    """防护拒绝了这次调用：没有发给 Home Assistant，本次没有执行。"""


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, (list, tuple)):
        return tuple(sorted(str(_normalize(item)) for item in value))
    return value


def device_target(args: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """调用的目标：名称、区域、楼层、类型；忽略大小写和顺序。"""
    return tuple((field, _normalize(args[field])) for field in TARGET_FIELDS
                 if args.get(field) not in (None, "", [], ()))


class DeviceTurnGuard:
    """每一轮新建一个；check 通过后、发出请求前调用 record。"""

    def __init__(self, limit: int = MAX_DEVICE_WRITES):
        self.limit = limit
        self.writes = 0
        self._done: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()

    @staticmethod
    def _key(tool: str, args: dict[str, Any]) -> tuple[str, tuple[tuple[str, Any], ...]]:
        # 开和关算同一类：开了再关就是来回切换。
        return ("power" if tool in POWER_TOOLS else tool, device_target(args))

    def check(self, tool: str, args: dict[str, Any]) -> str | None:
        """允许就返回 None；拒绝就返回给模型的说明。"""
        if tool in TARGETED_TOOLS and not device_target(args):
            return ("没有指定设备目标，本次没有执行。不带目标会作用到所有设备；"
                    "要操作全部灯请用 domain 指定 light，否则说明具体名称或区域。")
        if self.writes >= self.limit:
            return (f"本轮设备操作已达 {self.limit} 次上限，本次没有执行。"
                    "不要继续操作，直接根据已有结果向用户汇报。")
        if self._key(tool, args) in self._done:
            return ("本轮已经对这个目标操作过，本次没有执行。同一轮不要来回切换或重复操作同一设备，"
                    "也不要为了核对而改动；直接根据已有结果向用户汇报。")
        return None

    def record(self, tool: str, args: dict[str, Any]) -> None:
        self.writes += 1
        self._done.add(self._key(tool, args))
