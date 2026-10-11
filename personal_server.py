"""Google Tasks MCP Server。启动：alpha/.venv/bin/python alpha/personal_server.py。

使用 MCP SDK 2.3.0 的 MCPServer；请先运行 google_auth.py 完成授权。
"""
from datetime import datetime, timedelta
import re
from typing import Any

from google_auth import get_credentials
from googleapiclient.discovery import build  # type: ignore[import-untyped]
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer("Alpha Personal")


def _find_list_id(tasks: Any, list_name: str) -> str:
    """按名称查找清单；未找到时列出所有可用清单名称。"""
    page_token = None
    names: list[str] = []
    while True:
        response = tasks.tasklists().list(pageToken=page_token).execute()
        for task_list in response.get("items", []):
            names.append(task_list["title"])
            if task_list["title"] == list_name:
                return str(task_list["id"])
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    raise ToolError(
        f'找不到清单 "{list_name}"。现有清单：' + ("、".join(names) or "无")
    )


@mcp.tool()
def list_items(list_name: str) -> str:
    """读取指定 Google Tasks 清单的任务标题和完成状态，包括已完成任务。"""
    with build("tasks", "v1", credentials=get_credentials()) as tasks:
        list_id = _find_list_id(tasks, list_name)
        page_token = None
        lines: list[str] = []
        while True:
            response = tasks.tasks().list(
                tasklist=list_id, pageToken=page_token,
                showCompleted=True, showHidden=True, showDeleted=False,
            ).execute()
            for task in response.get("items", []):
                status = "已完成" if task.get("status") == "completed" else "未完成"
                lines.append(f"- {task.get('title') or '（无标题）'}：{status}")
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    return f"{list_name}：\n" + ("\n".join(lines) or "暂无任务")


@mcp.tool()
def add_items(list_name: str, items: list[str]) -> str:
    """向指定 Google Tasks 清单添加一个或多个任务，返回添加结果。"""
    if not isinstance(items, list) or not items or any(
        not isinstance(title, str) or not title.strip() for title in items
    ):
        raise ToolError("items 必须是至少包含一个非空任务标题的列表")

    added: list[str] = []
    with build("tasks", "v1", credentials=get_credentials()) as tasks:
        list_id = _find_list_id(tasks, list_name)
        for title in items:
            try:
                tasks.tasks().insert(tasklist=list_id, body={"title": title}).execute()
            except Exception as exc:
                completed = "、".join(added) or "无"
                raise ToolError(
                    f'添加 "{title}" 失败。已添加到 {list_name}：{completed}。'
                    "请只重试尚未添加的任务。"
                ) from exc
            added.append(title)
    return f"已添加到 {list_name}：" + "、".join(added)


@mcp.tool()
def create_reminder(title: str, when: str, minutes_before: int = 0) -> str:
    """在 Google 主日历创建 15 分钟日程并设置弹窗提醒。

    when 必须是带时区偏移的 ISO 8601 时间，例如 2026-10-11T08:00:00-07:00。
    相对时间需要先明确当前日期和时区，再转换为目标日期时间及正确的时区偏移。
    不能猜测时区；信息不足时需要用户补充。
    minutes_before 是提前提醒的分钟数，必须是 0 到 40320 的整数，0 表示准时提醒。
    """
    if not isinstance(title, str) or not title.strip():
        raise ToolError("提醒标题不能为空")
    if type(minutes_before) is not int or not 0 <= minutes_before <= 40320:
        raise ToolError("minutes_before 必须是 0 到 40320 的整数")
    if not isinstance(when, str):
        raise ToolError("when 必须是带时区偏移的 ISO 8601 时间字符串")
    try:
        start = datetime.fromisoformat(when)
    except ValueError as exc:
        raise ToolError(
            "时间格式无效，例如 2026-10-11T08:00:00-07:00"
        ) from exc
    if start.tzinfo is None or start.utcoffset() is None:
        raise ToolError("时间必须带时区偏移，例如 2026-10-11T08:00:00-07:00")
    # fromisoformat 接受的格式比 Calendar API 更宽；限制为 RFC3339 日期时间。
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})",
        when,
    ):
        raise ToolError("时间必须包含日期、时分秒和时区偏移，例如 2026-10-11T08:00:00-07:00")
    try:
        end = start + timedelta(minutes=15)
    except OverflowError as exc:
        raise ToolError("时间超出可支持的日期范围") from exc

    body = {
        "summary": title.strip(),
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": end.isoformat()},
        "reminders": {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": minutes_before}],
        },
    }
    with build("calendar", "v3", credentials=get_credentials()) as calendar:
        calendar.events().insert(calendarId="primary", body=body).execute()
    timing = "准时提醒" if minutes_before == 0 else f"提前 {minutes_before} 分钟提醒"
    return f"已创建提醒：{title.strip()}，时间：{start.isoformat()}，{timing}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
