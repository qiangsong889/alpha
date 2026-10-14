"""Google Tasks MCP Server。启动：alpha/.venv/bin/python alpha/personal_server.py。

使用 MCP SDK 2.3.0 的 MCPServer；请先运行 google_auth.py 完成授权。
"""
from datetime import datetime, timedelta, timezone
import os
import sys
import time
from contextlib import contextmanager
from functools import cache
from threading import RLock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import re
from typing import Any
from dotenv import load_dotenv

from core.paths import ENV_FILE
from opencc import OpenCC

from servers.google_auth import get_credentials
from googleapiclient.discovery import build  # type: ignore[import-untyped]
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

load_dotenv(ENV_FILE)
mcp = MCPServer("Alpha Personal")


# MCP server 可能在不同线程处理同步工具；同一个 Tasks service 的 HTTP transport 需串行使用。
# RLock 允许同一线程重复拿锁；_LIST_IDS 则让清单名命中后不必每次再请求 Google。
_TASKS_LOCK = RLock()
_LIST_IDS: dict[str, str] = {}
_TO_SIMPLIFIED = OpenCC("t2s")


@contextmanager
def timed_step(name: str):
    # with 进入时开始计时；yield 把控制权交给 with 里的代码；离开时执行 finally。
    # 即使 with 内抛异常也打印耗时。stdio 的 stdout 用来传 MCP 消息，日志必须去 stderr。
    started = time.perf_counter()
    try:
        yield
    finally:
        print(f"[server 耗时] {name} {time.perf_counter() - started:.2f}s",
              file=sys.stderr, flush=True)


@cache
def get_tasks_service() -> Any:
    # 无参数缓存：第一次创建 service，以后复用同一对象和凭据，不在每次工具调用后关闭。
    # 调用方持有 _TASKS_LOCK：串行使用非线程安全的 HTTP transport，
    # 也确保首次并发请求不会重复创建 service。
    with timed_step("get_credentials"):
        creds = get_credentials()
    with timed_step("build"):
        return build("tasks", "v1", credentials=creds, cache_discovery=False)


def _find_list_id(tasks: Any, list_name: str) -> str:
    """名称统一转简体后查找；未找到时列出原始清单名称。"""
    # “購物清單”和“购物清单”按同一个简体名称查找；缓存也使用统一后的名字。
    normalized_name = _TO_SIMPLIFIED.convert(list_name)
    if normalized_name in _LIST_IDS:
        return _LIST_IDS[normalized_name]
    page_token = None
    names: list[str] = []
    while True:
        # 这是 tasklists（清单目录），不是 tasks（某个清单下面的条目）。
        response = tasks.tasklists().list(pageToken=page_token).execute()
        for task_list in response.get("items", []):
            names.append(task_list["title"])
            if _TO_SIMPLIFIED.convert(task_list["title"]) == normalized_name:
                _LIST_IDS[normalized_name] = str(task_list["id"])
                return _LIST_IDS[normalized_name]
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    raise ToolError(
        f'找不到清单 "{list_name}"。现有清单：' + ("、".join(names) or "无")
    )


@mcp.tool()
def list_items(list_name: str) -> str:
    """读取指定 Google Tasks 清单的任务标题和完成状态，包括已完成任务。"""
    # @mcp.tool 把此函数注册为模型可发现的工具；函数名、参数和 docstring 提供工具描述。
    # 普通 with 获取线程锁；Google Python API 是同步调用，因此这里不用 async with。
    with _TASKS_LOCK:
        tasks = get_tasks_service()
        list_id = _find_list_id(tasks, list_name)
        page_token = None
        lines: list[str] = []
        while True:
            response = tasks.tasks().list(
                tasklist=list_id, pageToken=page_token,
                showCompleted=False, showHidden=True, showDeleted=False,
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

    # Google 每次 insert 只创建一个任务；记录已成功部分，失败时告诉模型不要全部重做。
    added: list[str] = []
    with _TASKS_LOCK:
        tasks = get_tasks_service()
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
    # 先校验全部输入，再调用 Google，避免时间不明确时创建一个偏了几小时的提醒。
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
        # 夏令时切换时，本地时间可能不存在，或对应两个真实时刻；fold 分别检查两种解释。
        for fold in (0, 1):
            candidate = local.replace(tzinfo=tz, fold=fold)
            instant = candidate.astimezone(timezone.utc)
            # 先转 UTC 再转回本地：如果回不到原值，说明这是跳过的、不存在的本地时间。
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

    # 日历事件需要一个开始和结束的时间段；提醒只在开始时间减 minutes_before 时触发。
    body = {
        "summary": title.strip(),
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": end.isoformat()},
        "reminders": {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": minutes_before}],
        },
    }
    # 这个 Calendar service 只在当前工具里用，with 退出时关闭它；不是 Tasks 的共享缓存。
    with build("calendar", "v3", credentials=get_credentials(), cache_discovery=False) as calendar:
        calendar.events().insert(calendarId="primary", body=body).execute()
    timing = "准时提醒" if minutes_before == 0 else f"提前 {minutes_before} 分钟提醒"
    return f"已创建提醒：{title.strip()}，时间：{start.isoformat()}（{tz_name}），{timing}"


@mcp.tool()
def complete_items(list_name: str, item_titles: list[str]) -> str:
    """将指定清单中匹配标题的所有未完成任务标记完成。

    一个标题会匹配全部同名任务，无需重复传入；重复输入的标题会去重。
    找不到的标题不会阻止其他任务完成，返回结果分别报告已完成和没找到的条目。
    """
    if not item_titles or any(
        not isinstance(title, str) or not title.strip()
        for title in item_titles
    ):
        raise ToolError("item_titles 必须是至少包含一个非空任务标题的列表")

    # 字典键自动去重并保留顺序：输入两次“牛奶”也只按一个标题查找。
    titles = list(dict.fromkeys(item_titles))

    with _TASKS_LOCK:
        with timed_step("get_tasks_service"):
            tasks = get_tasks_service()
        with timed_step("_find_list_id"):
            list_id = _find_list_id(tasks, list_name)
        # 一个标题可能对应多个任务 id，因此 value 是列表；一次勾掉所有同名未完成任务。
        matches: dict[str, list[str]] = {title: [] for title in titles}
        page_token = None
        page_number = 0

        while True:
            page_number += 1
            with timed_step(f"tasks().list() 第{page_number}页"):
                response = tasks.tasks().list(
                    tasklist=list_id,
                    pageToken=page_token,
                    showCompleted=False,
                    showDeleted=False,
                ).execute()

            for task in response.get("items", []):
                title = task.get("title")
                if title in matches:
                    # dict 的 in 检查键；命中标题后，把 Google 的实际 task id 存进对应列表。
                    matches[title].append(task["id"])

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        # 找不到的标题独立汇报，仍继续完成其他找得到的任务，实现部分成功。
        missing = [title for title in titles if not matches[title]]
        completed: dict[str, int] = {}
        for title in titles:
            try:
                for task_id in matches[title]:
                    with timed_step(f"patch {title}（{task_id}）"):
                        tasks.tasks().patch(
                            # patch 只修改给定字段；以 task id 定位，标题本身不是 API 主键。
                            tasklist=list_id,
                            task=task_id,
                            body={"status": "completed"},
                        ).execute()
                    completed[title] = completed.get(title, 0) + 1
            except Exception as exc:
                confirmed = "、".join(
                    f"{name}（{count} 条）" for name, count in completed.items()
                ) or "无"
                raise ToolError(
                    f'标记 "{title}" 完成时失败。'
                    f'已确认完成：{confirmed}。'
                    f'没找到：{"、".join(missing) or "无"}。'
                    "失败的任务也可能已生效，请查询状态后再重试。"
                ) from exc

    confirmed = "、".join(
        f"{name}（{count} 条）" for name, count in completed.items()
    ) or "无"
    return f"{list_name}：已完成：{confirmed}；没找到：{'、'.join(missing) or '无'}"
        
    
    
if __name__ == "__main__":
    # 客户端启动此子进程后，经 stdin/stdout 收发 MCP 消息，不监听 HTTP 端口。
    mcp.run(transport="stdio")
