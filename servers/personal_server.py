"""Google 个人服务 MCP Server。启动：.venv/bin/python -m servers.personal_server。

使用 MCP SDK 2.3.0 的 MCPServer；请先运行 python -m servers.google_auth 完成授权。
"""
from datetime import datetime, timedelta, timezone
import os
import sys
import time
from contextlib import contextmanager
from threading import RLock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import re
import hashlib
import httpx2
from typing import Any, Literal
from dotenv import load_dotenv

from core.paths import ENV_FILE
from core.settings import setting
from opencc import OpenCC

from google.auth.exceptions import RefreshError
from servers.google_auth import GoogleAuthRequired, get_credentials
from servers import weather
from servers.google_tools import register_google_tools
from googleapiclient.discovery import build  # type: ignore[import-untyped]
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

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


# (创建时的 access token, Tasks service)；调用方持有 _TASKS_LOCK。
_TASKS_SERVICE: tuple[str | None, Any] | None = None


def get_tasks_service() -> Any:
    """复用同一个 Tasks service；每次先从磁盘取凭据（必要时刷新，失败抛 GoogleAuthRequired）。

    令牌换了（刷新或重新授权）才重建 service：重新授权后不用重启 Alpha，
    写请求发出前令牌也一定有效。调用方持有 _TASKS_LOCK，串行使用非线程安全的 HTTP transport。
    """
    global _TASKS_SERVICE
    with timed_step("get_credentials"):
        creds = get_credentials()
    if _TASKS_SERVICE is None or _TASKS_SERVICE[0] != creds.token:
        with timed_step("build"):
            _TASKS_SERVICE = (creds.token, build("tasks", "v1", credentials=creds, cache_discovery=False))
    return _TASKS_SERVICE[1]


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
def list_items(list_name: str) -> CallToolResult:
    """读取指定 Google Tasks 清单的任务标题和完成状态，包括已完成任务。"""
    # @mcp.tool 把此函数注册为模型可发现的工具；函数名、参数和 docstring 提供工具描述。
    # 普通 with 获取线程锁；Google Python API 是同步调用，因此这里不用 async with。
    with _TASKS_LOCK:
        tasks = get_tasks_service()
        list_id = _find_list_id(tasks, list_name)
        page_token = None
        lines: list[str] = []
        entries: list[dict[str, str]] = []
        while True:
            response = tasks.tasks().list(
                tasklist=list_id, pageToken=page_token,
                showCompleted=False, showHidden=True, showDeleted=False,
            ).execute()
            for task in response.get("items", []):
                status = "已完成" if task.get("status") == "completed" else "未完成"
                lines.append(f"- {task.get('title') or '（无标题）'}：{status}")
                entries.append({"task_id": task["id"], "title": task.get("title", ""),
                                "status": task.get("status", "needsAction")})
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    return _task_result(f"{list_name}：\n" + ("\n".join(lines) or "暂无任务"), {
        "list_id": list_id, "list_name": list_name, "items": entries,
    })


@mcp.tool()
def list_reminders(max_results: int = 10) -> CallToolResult:
    """读取主日历未来的日程及其事件 ID；ID 用于精确定位，不要向用户朗读。"""
    if type(max_results) is not int or not 1 <= max_results <= 20:
        raise ToolError("max_results 必须是 1 到 20 的整数")
    with build("calendar", "v3", credentials=get_credentials(), cache_discovery=False) as calendar:
        response = calendar.events().list(calendarId="primary", maxResults=max_results,
            timeMin=datetime.now(timezone.utc).isoformat(), singleEvents=True, orderBy="startTime").execute()
    entries = [{"event_id": item["id"], "title": item.get("summary", "（无标题）"),
                "start": item.get("start", {})} for item in response.get("items", [])]
    text = "\n".join(f"{item['title']}：{item['start'].get('dateTime', item['start'].get('date', ''))}"
                     for item in entries) or "暂无即将到来的日程"
    return _task_result(text, {"items": entries})


def _task_result(text: str, data: dict[str, Any], *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)],
                          structured_content=data, is_error=is_error)


@mcp.tool()
def get_weather(day: Literal["today", "tomorrow"] = "today") -> CallToolResult:
    """查询 HA 配置的家附近今天或明天的天气，摄氏度；不返回或朗读家的经纬度。

    这是家庭位置的预报，不代表用户当前定位，也不用于查询其他城市。
    """
    if day not in ("today", "tomorrow"):
        return _task_result("day 只能是 today 或 tomorrow", {}, is_error=True)
    with httpx2.Client(timeout=5.0) as client:
        try:
            lat, lon, tz = weather.home_location(client)
        except (httpx2.HTTPError, ValueError, KeyError, OverflowError) as exc:
            # 不返回异常原文：请求 URL 或服务响应可能包含家的坐标。
            reason = f"HTTP {exc.response.status_code}" if isinstance(exc, httpx2.HTTPStatusError) else "请检查 HA 连接、家庭经纬度和时区配置"
            return _task_result(f"读取家庭位置失败：{reason}", {}, is_error=True)
        try:
            data = weather.fetch_forecast(client, lat, lon, tz)
            return _task_result(weather.summarize(data, day), weather.forecast_details(data, day))
        except (httpx2.HTTPError, ValueError, KeyError, OverflowError) as exc:
            reason = f"HTTP {exc.response.status_code}" if isinstance(exc, httpx2.HTTPStatusError) else "连接超时、网络故障或返回数据不完整"
            return _task_result(f"天气服务暂时不可用：{reason}", {}, is_error=True)


@mcp.tool()
def add_items(list_name: str, items: list[str], operation_id: str = "") -> CallToolResult:
    """向指定 Google Tasks 清单添加任务，返回确认文字和本次新建条目的 ID。

    创建中断时返回已确认创建的条目与未确定结果，不要整批重试。
    """
    if not isinstance(list_name, str) or not list_name.strip():
        raise ToolError("list_name 必须是非空清单名称")
    if not isinstance(items, list) or not items or any(
        not isinstance(title, str) or not title.strip() for title in items
    ):
        raise ToolError("items 必须是至少包含一个非空任务标题的列表")

    # Google 每次 insert 只创建一个任务；记录已成功部分，失败时告诉模型不要全部重做。
    added: list[dict[str, str]] = []
    with _TASKS_LOCK:
        tasks = get_tasks_service()
        list_id = _find_list_id(tasks, list_name)
        for index, title in enumerate(items):
            try:
                body = {"title": title}
                if operation_id:
                    body["notes"] = f"alpha-operation:{operation_id}:{index}"
                created = tasks.tasks().insert(tasklist=list_id, body=body).execute()
                task_id = created.get("id") if isinstance(created, dict) else None
                if not isinstance(task_id, str) or not task_id.strip():
                    raise ValueError("Google 返回结果缺少新建任务 ID")
            # 授权失败也按未知处理：前面的条目可能已经添加，不能整体报"没有执行"。
            # get_tasks_service() 已在开始前刷新令牌，中途失效只会是授权恰好被撤销。
            except Exception as exc:
                completed = "、".join(item["title"] for item in added) or "无"
                text = (f'已确认添加到 {list_name}：{completed}。添加 "{title}" 的结果未知，'
                        "可能已经生效，不要重试，先查询核实；后续条目尚未尝试。")
                return _task_result(text, {
                    "version": 1, "status": "unknown", "list_id": list_id,
                    "list_name": list_name, "created": added,
                    "unconfirmed": {"title": title}, "unattempted": items[index + 1:],
                    "error": f"{type(exc).__name__}: {exc}",
                }, is_error=True)
            added.append({"task_id": task_id, "title": title})
    text = f"已添加到 {list_name}：" + "、".join(item["title"] for item in added)
    return _task_result(text, {
        "version": 1, "status": "succeeded", "list_id": list_id,
        "list_name": list_name, "created": added,
    })


@mcp.tool()
def delete_task(list_id: str, task_id: str) -> CallToolResult:
    """按精确的清单 ID 和任务 ID 删除一条任务，用于撤销本次新增。

    ID 必须来自已确认的创建记录，不能用标题代替，也不能猜测 ID。
    删除是一个新的写操作，不代表原来的添加没有发生。
    """
    for name, value in (("list_id", list_id), ("task_id", task_id)):
        if not isinstance(value, str) or not value.strip():
            raise ToolError(f"{name} 必须是非空 ID")
    with _TASKS_LOCK:
        tasks = get_tasks_service()
        try:
            tasks.tasks().delete(tasklist=list_id, task=task_id).execute()
        except RefreshError as exc:
            raise GoogleAuthRequired("已失效") from exc  # 请求没有发出，或已被 401 拒绝：都没有执行
        except Exception as exc:
            return _task_result("删除结果未知，可能已经生效。不要重试，先查询核实。", {
                "version": 1, "status": "unknown", "list_id": list_id,
                "task_id": task_id, "error": f"{type(exc).__name__}: {exc}",
            }, is_error=True)
    return _task_result("已删除指定任务", {
        "version": 1, "status": "succeeded", "list_id": list_id, "task_id": task_id,
    })


@mcp.tool()
def create_reminder(title: str, local_time: str, minutes_before: int = 0,
                    operation_id: str = "") -> CallToolResult:
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
    try:
        tz_name = setting("REMINDER_TIMEZONE")
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolError(f"REMINDER_TIMEZONE 不是可用的 IANA 时区：{exc}") from exc
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
    if operation_id:
        # Calendar 支持客户端指定事件 ID；哈希只含允许的 base32hex 字符。
        body["id"] = _event_id(operation_id)
    # 这个 Calendar service 只在当前工具里用，with 退出时关闭它；不是 Tasks 的共享缓存。
    with build("calendar", "v3", credentials=get_credentials(), cache_discovery=False) as calendar:
        try:
            created = calendar.events().insert(calendarId="primary", body=body).execute()
        except RefreshError as exc:
            raise GoogleAuthRequired("已失效") from exc  # 请求没有发出，或已被 401 拒绝：都没有执行
        except Exception as exc:
            return _task_result("创建提醒结果未知，可能已经生效，请先核实，不要重试。", {
                "version": 1, "status": "unknown", "event_id": body.get("id"),
                "error": f"{type(exc).__name__}: {exc}",
            }, is_error=True)
    timing = "准时提醒" if minutes_before == 0 else f"提前 {minutes_before} 分钟提醒"
    event_id = created.get("id") if isinstance(created, dict) else None
    if not isinstance(event_id, str) or not event_id:
        return _task_result("提醒可能已创建，但没有收到事件 ID，请先核实，不要重试。",
                            {"version": 1, "status": "unknown"}, is_error=True)
    return _task_result(f"已创建提醒：{title.strip()}，时间：{start.isoformat()}（{tz_name}），{timing}",
                        {"version": 1, "status": "succeeded", "event_id": event_id,
                         "title": title.strip(), "local_time": local_time})


def _event_id(operation_id: str) -> str:
    return hashlib.sha256(operation_id.encode()).hexdigest()


@mcp.tool()
def delete_reminder(event_id: str) -> CallToolResult:
    """按创建回执的事件 ID 删除主日历提醒。仅供授权后的撤销使用。"""
    if not event_id.strip():
        raise ToolError("event_id 不能为空")
    with build("calendar", "v3", credentials=get_credentials(), cache_discovery=False) as calendar:
        try:
            calendar.events().delete(calendarId="primary", eventId=event_id).execute()
        except RefreshError as exc:
            raise GoogleAuthRequired("已失效") from exc  # 请求没有发出，或已被 401 拒绝：都没有执行
        except Exception as exc:
            return _task_result("删除提醒结果未知，请先查询核实，不要重试。", {
                "version": 1, "status": "unknown", "event_id": event_id,
                "error": f"{type(exc).__name__}: {exc}",
            }, is_error=True)
    return _task_result("已删除指定提醒", {"version": 1, "status": "succeeded", "event_id": event_id})


@mcp.tool()
def inspect_reminder(event_id: str = "", operation_id: str = "") -> CallToolResult:
    """按保存的事件 ID 或添加操作编号查询，不修改日历。"""
    target = event_id or (_event_id(operation_id) if operation_id else "")
    if not target:
        raise ToolError("必须提供事件 ID 或操作编号")
    with build("calendar", "v3", credentials=get_credentials(), cache_discovery=False) as calendar:
        try:
            event = calendar.events().get(calendarId="primary", eventId=target).execute()
        except HttpError as exc:
            if exc.resp.status not in {404, 410}:
                raise
            event = {"id": target, "status": "cancelled"}
    present = event.get("status") != "cancelled"
    return _task_result("提醒存在" if present else "没有查到有效提醒", {
        "present": present, "event_id": target, "title": event.get("summary", ""),
        "start": event.get("start", {}),
        "resource": event if present else None,
    })


@mcp.tool()
def inspect_task(list_id: str, task_id: str) -> CallToolResult:
    """按已保存的精确 ID 查询任务是否仍存在，不修改清单。"""
    if not list_id.strip() or not task_id.strip():
        raise ToolError("清单和任务 ID 不能为空")
    with _TASKS_LOCK:
        try:
            task = get_tasks_service().tasks().get(tasklist=list_id, task=task_id).execute()
        except HttpError as exc:
            if exc.resp.status not in {404, 410}:
                raise
            task = {"deleted": True}
    return _task_result("任务存在" if not task.get("deleted") else "任务已不存在", {
        "present": not bool(task.get("deleted")), "list_id": list_id,
        "task_id": task_id, "title": task.get("title", ""),
        "resource": task if not task.get("deleted") else None,
    })


@mcp.tool()
def restore_task(list_id: str, resource: dict) -> CallToolResult:
    """按代码保存的删除前快照重新创建任务；Google 会分配新的任务 ID。"""
    if not list_id.strip() or not isinstance(resource.get("title"), str):
        raise ToolError("缺少清单 ID 或有效任务快照")
    body = {key: resource[key] for key in ("title", "notes", "status", "due", "completed") if key in resource}
    with _TASKS_LOCK:
        service = get_tasks_service()
        try:
            created = service.tasks().insert(tasklist=list_id, body=body,
                                             **({"parent": resource["parent"]} if resource.get("parent") else {})).execute()
        except RefreshError as exc:
            raise GoogleAuthRequired("已失效") from exc  # 请求没有发出，或已被 401 拒绝：都没有执行
        except Exception as exc:
            return _task_result("恢复结果未知，可能已生效，不要重试", {
                "status": "unknown", "error": f"{type(exc).__name__}: {exc}",
            }, is_error=True)
    if not created.get("id"):
        return _task_result("恢复结果未知，不要重试", {"status": "unknown"}, is_error=True)
    return _task_result(f"已恢复任务：{body['title']}", {
        "status": "succeeded", "list_id": list_id, "task_id": created["id"], "title": body["title"],
    })


@mcp.tool()
def restore_reminder(resource: dict) -> CallToolResult:
    """按代码保存的普通日程快照重新创建；不承诺恢复会议链接或参与者邀请。"""
    if not resource.get("start") or not resource.get("end"):
        raise ToolError("缺少有效的日程快照")
    if resource.get("recurrence") or resource.get("recurringEventId") or resource.get("eventType", "default") != "default":
        raise ToolError("暂不支持恢复重复或特殊日程")
    body = {key: resource[key] for key in ("summary", "description", "location", "start", "end",
            "reminders", "colorId", "visibility", "transparency") if key in resource}
    with build("calendar", "v3", credentials=get_credentials(), cache_discovery=False) as calendar:
        try:
            created = calendar.events().insert(calendarId="primary", body=body).execute()
        except RefreshError as exc:
            raise GoogleAuthRequired("已失效") from exc  # 请求没有发出，或已被 401 拒绝：都没有执行
        except Exception as exc:
            return _task_result("恢复结果未知，可能已生效，不要重试", {
                "status": "unknown", "error": f"{type(exc).__name__}: {exc}",
            }, is_error=True)
    if not created.get("id"):
        return _task_result("恢复结果未知，不要重试", {"status": "unknown"}, is_error=True)
    return _task_result(f"已恢复日程：{body.get('summary', '（无标题）')}", {
        "status": "succeeded", "event_id": created["id"],
    })


@mcp.tool()
def find_created_items(list_name: str, operation_id: str, items: list[str]) -> CallToolResult:
    """用操作编号标记查询这次创建的任务，不能按同名任务猜测结果。"""
    if not operation_id or not items:
        raise ToolError("操作编号和原始条目不能为空")
    with _TASKS_LOCK:
        service = get_tasks_service()
        list_id = _find_list_id(service, list_name)
        found: dict[int, list[dict[str, str]]] = {index: [] for index in range(len(items))}
        page_token = None
        while True:
            response = service.tasks().list(tasklist=list_id, pageToken=page_token,
                                            showCompleted=True, showHidden=True, showDeleted=False).execute()
            for task in response.get("items", []):
                for index, title in enumerate(items):
                    if (task.get("notes") == f"alpha-operation:{operation_id}:{index}"
                            and task.get("title") == title and not task.get("deleted")):
                        found[index].append({"task_id": task["id"], "title": title})
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    complete = all(len(found[index]) == 1 for index in found)
    return _task_result("已核实全部条目" if complete else "未能核实全部条目，不要重试添加", {
        "version": 1, "status": "succeeded" if complete else "unknown",
        "list_id": list_id, "list_name": list_name,
        "created": [found[index][0] for index in found if len(found[index]) == 1],
    })


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
        
    
    
register_google_tools(mcp)


if __name__ == "__main__":
    # 客户端启动此子进程后，经 stdin/stdout 收发 MCP 消息，不监听 HTTP 端口。
    mcp.run(transport="stdio")
