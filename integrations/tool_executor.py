"""把 Gateway 的调用路由到实际 MCP 服务，并如实格式化操作状态。"""

from datetime import datetime
import json
import os
from typing import TypedDict
from zoneinfo import ZoneInfo

import httpx2
from mcp.types import CallToolResult, TextContent

from integrations.mcp_client import call_ha_tool, device_action_succeeded
from integrations.operations import Operation
from integrations.personal_worker import PersonalWorker
from integrations.tool_gateway import CURRENT_OPERATION_ID, EMAIL_CONFIRMATION, Executor, ExecutorResult, ToolNotSent, ToolOutcomeUnknown
from integrations.tool_policy import GOOGLE_TOOLS, INTERNAL_TOOLS, get_policy


HA_ACTION_TOOLS = {"intent__HassTurnOn", "intent__HassTurnOff", "light__HassLightSet"}
HA_DEVICE_TOOLS = HA_ACTION_TOOLS | {"homeassistant__GetLiveContext"}
PERSONAL_TOOLS = {"list_items", "list_reminders", "get_weather", "add_items", "complete_items", "create_reminder"} | GOOGLE_TOOLS


def make_executor(
    *, ha_url: str, http_client: httpx2.AsyncClient | None, worker: PersonalWorker | None,
) -> Executor:
    async def execute(name: str, arguments: dict) -> ExecutorResult:
        if get_policy(name) is None:
            raise ToolNotSent(f"工具不在策略白名单中：{name}")
        if name in HA_DEVICE_TOOLS:
            if not ha_url or http_client is None or http_client.is_closed:
                raise ToolNotSent("HA 地址或 HTTP 客户端不可用，工具请求尚未发出")
            # 只有连接/握手阶段失败标为 ToolNotSent，工具请求之后的异常原样交给 Gateway。
            result = await call_ha_tool(
                name, arguments, url=ha_url, http_client=http_client, classify_not_sent=True,
            )
        elif name in PERSONAL_TOOLS | INTERNAL_TOOLS:
            if worker is None or worker.closed:
                raise ToolNotSent("个人服务未启动或已关闭，工具请求尚未发出")
            # 即使这里超时，也可能已经入队或执行，不能把它包装成 ToolNotSent。
            sent_arguments = dict(arguments)
            if name in {"send_email", "reply_email"}:
                token = EMAIL_CONFIRMATION.get()
                if not token:
                    raise ToolNotSent("邮件尚未经过用户确认，本次没有发送")
                sent_arguments.update(confirmed=True, confirmation_token=token)
            if name in {"add_items", "create_reminder", "create_event"} and CURRENT_OPERATION_ID.get():
                sent_arguments["operation_id"] = CURRENT_OPERATION_ID.get()
            result = await worker.call(name, sent_arguments)
        else:
            raise ToolNotSent(f"工具没有对应的服务器：{name}")

        if not isinstance(result, CallToolResult):
            raise TypeError("MCP 工具未返回完整结果")
        text = "\n".join(item.text for item in result.content if isinstance(item, TextContent))
        if not text and result.structured_content is not None:
            text = json.dumps(result.structured_content, ensure_ascii=False)
        ok = device_action_succeeded(name, result) if name in HA_ACTION_TOOLS else not result.is_error
        if name in PERSONAL_TOOLS | INTERNAL_TOOLS and result.structured_content is not None:
            payload = {"text": text or "工具未返回文字", "structured_content": result.structured_content}
            if (policy := get_policy(name)) is not None and policy.mutates and result.structured_content.get("status") == "unknown":
                raise ToolOutcomeUnknown(payload)
            return ExecutorResult(ok=ok, result=payload)
        return ExecutorResult(ok=ok, result=text or "工具未返回文字")

    return execute


class ModelToolResult(TypedDict):
    is_error: bool
    text: str


def _result_text(result: object | None) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and isinstance(result.get("text"), str):
        return result["text"]
    if isinstance(result, dict) and isinstance(result.get("error"), str):
        return result["error"]
    return json.dumps(result, ensure_ascii=False) if result is not None else ""


def format_for_model(op: Operation, duplicate_of: Operation | None = None) -> ModelToolResult:
    """重复提示说明的是本次未执行，不借用旧记录假称这次已经完成。"""
    if duplicate_of is not None:
        if op.status not in {"pending", "approved", "cancelled"} or duplicate_of.status != "succeeded":
            raise ValueError("重复提示必须对应未执行的本次操作和成功的旧操作")
        tz = ZoneInfo(os.environ.get("REMINDER_TIMEZONE", "America/Los_Angeles"))
        executed_at = datetime.fromtimestamp(duplicate_of.updated_at, tz).strftime("%H:%M")
        return {"is_error": False, "text": (
            f"本次没有执行。相同操作在 {executed_at} 已经成功执行过（那是之前的那次）。"
            "如果用户确实要再做一次，先向用户确认。"
        )}
    text = _result_text(op.result)
    if op.status == "succeeded":
        return {"is_error": False, "text": text or "操作已成功执行，工具未返回文字"}
    if op.status == "failed":
        return {"is_error": True, "text": text or "操作失败，工具未返回原因"}
    if op.status == "unknown":
        return {"is_error": True, "text": "结果未知，可能已经生效。不要重试，先查询确认。" + (f"原因：{text}" if text else "")}
    if op.status == "pending":
        return {"is_error": False, "text": f"已登记，尚未执行（操作 {op.id}）"}
    if op.status == "approved":
        return {"is_error": False, "text": "已批准，尚未执行"}
    if op.status == "executing":
        return {"is_error": False, "text": "正在执行，尚未拿到结果，不要重复提交"}
    return {"is_error": False, "text": "已取消，未执行"}
