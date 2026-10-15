"""Qwen 可见的本地计划控制工具，不注册到 MCP 或 TOOL_POLICY。"""

from typing import Any

from core.plan_coordinator import CoordinatorResult, PlanCoordinator
from integrations.tool_executor import format_for_model


def _tool(description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False,
    }}


_PLAN_ID = {"type": "string", "description": "工具结果中当前计划的 id"}
_ITEM_ID = {"type": "integer", "minimum": 1, "description": "固定项编号 item_id，如 #2 传 2；不是列表下标"}
CONTROL_TOOLS = {
    "confirm_plan": _tool(
        "只有用户明确同意最近一次已播放完的复述时才调用。用户提问、要求修改或意思含糊时不能调用。"
        "version 使用最新计划版本。只有用户明确表示相同的两项都要时，duplicates_intended 才传 true。",
        {"plan_id": _PLAN_ID, "version": {"type": "integer"},
         "duplicates_intended": {"type": "boolean", "default": False}}, ["plan_id", "version"],
    ),
    "update_plan_item": _tool(
        "按固定 item_id 修改计划项，arguments 是这个工具完整的新参数。可以在同一句话里修改多项。",
        {"plan_id": _PLAN_ID, "item_id": _ITEM_ID, "arguments": {"type": "object"}},
        ["plan_id", "item_id", "arguments"],
    ),
    "remove_plan_item": _tool(
        "按固定 item_id 删除计划项。删除后其他编号不变。",
        {"plan_id": _PLAN_ID, "item_id": _ITEM_ID}, ["plan_id", "item_id"],
    ),
    "cancel_plan": _tool("用户明确取消整个计划时调用。", {"plan_id": _PLAN_ID}, ["plan_id"]),
}


async def call_control(coordinator: PlanCoordinator, name: str, arguments: dict) -> CoordinatorResult:
    schema = CONTROL_TOOLS[name]["parameters"]
    if (set(arguments) - set(schema["properties"])
            or set(schema["required"]) - set(arguments)):
        raise ValueError("计划控制工具的参数缺失或包含未知字段")
    types = {"string": str, "integer": int, "boolean": bool, "object": dict}
    for key, value in arguments.items():
        if type(value) is not types[schema["properties"][key]["type"]]:
            raise ValueError(f"计划控制参数类型错误：{key}")
    if name == "confirm_plan":
        return await coordinator.confirm(**arguments)
    if name == "update_plan_item":
        return await coordinator.update(**arguments)
    if name == "remove_plan_item":
        return coordinator.remove(**arguments)
    return coordinator.cancel_plan(**arguments)


def control_output(result: CoordinatorResult) -> dict:
    operations = [{"op_id": op.id, "name": op.name, "status": op.status, **format_for_model(op)}
                  for op in result.operations]
    return {
        "is_error": not result.ok, "reason": result.reason, "version": result.version,
        "snapshot": result.snapshot, "operations": operations,
        "text": ("\n".join(f"{op['name']} [{op['status']}]：{op['text']}" for op in operations)
                 or result.reason),
    }
