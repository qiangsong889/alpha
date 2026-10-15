"""工具的本地执行策略；表内名称同时构成新网关的白名单。"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ToolPolicy:
    mutates: bool
    retry_safe: bool
    commit: Literal["immediate", "deferred", "confirm"]


TOOL_POLICY: dict[str, ToolPolicy] = {
    "list_items": ToolPolicy(mutates=False, retry_safe=True, commit="immediate"),
    "homeassistant__GetLiveContext": ToolPolicy(mutates=False, retry_safe=True, commit="immediate"),
    "intent__HassTurnOn": ToolPolicy(mutates=True, retry_safe=True, commit="immediate"),
    "intent__HassTurnOff": ToolPolicy(mutates=True, retry_safe=True, commit="immediate"),
    "light__HassLightSet": ToolPolicy(mutates=True, retry_safe=True, commit="immediate"),
    # 当前按标题重新查找所有同名任务；重试时可能匹配后来新增的任务，因此不保证重试安全。
    # 当前没有取消完成工具，提交前保留延迟窗口，允许用户继续补充或取消。
    "complete_items": ToolPolicy(mutates=True, retry_safe=False, commit="deferred"),
    "add_items": ToolPolicy(mutates=True, retry_safe=False, commit="deferred"),
    "create_reminder": ToolPolicy(mutates=True, retry_safe=False, commit="deferred"),
}


def get_policy(name: str) -> ToolPolicy | None:
    """未登记的工具不提供默认策略；未来网关看到 None 时必须拒绝。"""
    return TOOL_POLICY.get(name)
