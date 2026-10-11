"""Google Tasks MCP Server。启动：alpha/.venv/bin/python alpha/personal_server.py。

使用 MCP SDK 2.3.0 的 MCPServer；请先运行 google_auth.py 完成授权。
"""
from datetime import datetime, timedelta, timezone
import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import re
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

from google_auth import get_credentials
from googleapiclient.discovery import build  # type: ignore[import-untyped]
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

load_dotenv(Path(__file__).with_name(".env"))
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
    with build("tasks", "v1", credentials=get_credentials(), cache_discovery=False) as tasks:
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
    with build("tasks", "v1", credentials=get_credentials(), cache_discovery=False) as tasks:
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
def create_reminder(title: str, local_time: str, minutes_before: int = 0) -> str:
    """在 Google 主日历创建 15 分钟日程并设置弹窗提醒。

    local_time 必须是不带时区的本地 ISO 8601 时间，例如 2026-11-03T08:00:00。
    时区由服务端 REMINDER_TIMEZONE 配置决定（默认 America/Los_Angeles），
    服务端计算目标日期的夏令时偏移。相对时间需要先确定本地日期。
    夏令时切换导致不存在或重复的本地时间会被拒绝，需要选择明确的时间。
    minutes_before 必须是 0 到 40320 的整数，0 表示准时提醒。
    """
    if not isinstance(title, str) or not title.strip():
        raise ToolError("提醒标题不能为空")
    if type(minutes_before) is not int or not 0 <= minutes_before <= 40320:
        raise ToolError("minutes_before 必须是 0 到 40320 的整数")
    if not isinstance(local_time, str):
        raise ToolError("local_time 必须是不带时区的本地时间字符串")
    try:
        local = datetime.fromisoformat(local_time)
    except ValueError as exc:
        raise ToolError("时间格式无效，例如 2026-11-03T08:00:00") from exc
    if local.tzinfo is not None:
        raise ToolError("local_time 不要带时区或偏移；时区由服务端配置决定")
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?", local_time
    ):
        raise ToolError("时间必须包含日期和时分秒，例如 2026-11-03T08:00:00")
    tz_name = os.environ.get("REMINDER_TIMEZONE", "America/Los_Angeles")
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolError(f"REMINDER_TIMEZONE 不是可用的 IANA 时区：{tz_name}") from exc
    try:
        candidates = []
        for fold in (0, 1):
            candidate = local.replace(tzinfo=tz, fold=fold)
            instant = candidate.astimezone(timezone.utc)
            if instant.astimezone(tz).replace(tzinfo=None) == local:
                if instant not in candidates:
                    candidates.append(instant)
        if not candidates:
            raise ToolError(f"{local_time} 在 {tz_name} 因夏令时切换不存在，请选择其他时间")
        if len(candidates) > 1:
            raise ToolError(f"{local_time} 在 {tz_name} 因夏令时切换出现两次，请选择不重复的时间")
        start = candidates[0].astimezone(tz)
        # 用 UTC 加时长，保证跨切换时仍然持续 15 分钟。
        end = (candidates[0] + timedelta(minutes=15)).astimezone(tz)
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
    with build("calendar", "v3", credentials=get_credentials(), cache_discovery=False) as calendar:
        calendar.events().insert(calendarId="primary", body=body).execute()
    timing = "准时提醒" if minutes_before == 0 else f"提前 {minutes_before} 分钟提醒"
    return f"已创建提醒：{title.strip()}，时间：{start.isoformat()}（{tz_name}），{timing}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
