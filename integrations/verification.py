"""只读核实未知的 Google 写入；查不到创建结果不等于创建失败。"""

from datetime import datetime
from zoneinfo import ZoneInfo

from integrations.operations import Operation
from integrations.tool_gateway import ToolGateway


async def verify_operation(gateway: ToolGateway, op: Operation, timezone_name: str) -> Operation:
    if op.status != "unknown":
        return op
    if op.name == "add_items":
        name, args = "find_created_items", dict(op.arguments, operation_id=op.id)
    elif op.name == "create_reminder":
        name, args = "inspect_reminder", {"operation_id": op.id}
    elif op.name == "delete_task":
        name, args = "inspect_task", op.arguments
    elif op.name == "delete_reminder":
        name, args = "inspect_reminder", op.arguments
    else:
        return op
    query = await gateway.submit("verify:" + op.id, name, args)
    data = query.result.get("structured_content", {}) if isinstance(query.result, dict) else {}
    success = False
    if query.status == "succeeded":
        if op.name == "add_items":
            success = data.get("status") == "succeeded"
        elif op.name == "create_reminder" and data.get("present") is True:
            try:
                local = datetime.fromisoformat(data.get("start", {}).get("dateTime", "")).astimezone(
                    ZoneInfo(timezone_name)).replace(tzinfo=None)
                success = (local == datetime.fromisoformat(op.arguments["local_time"])
                           and data.get("title") == op.arguments["title"].strip())
            except (ValueError, TypeError):
                pass
        elif op.name.startswith("delete_"):
            success = data.get("present") is False
    if success and op.name == "create_reminder":
        data = dict(data, version=1, status="succeeded")
    return gateway.store.record_verification(op.id, {
        "text": "查询已证实操作生效" if success else "查询尚不能确定原操作结果，不要重试",
        "structured_content": data, "previous_result": op.result,
    }, succeeded=success)
