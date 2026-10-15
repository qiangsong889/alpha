"""日常语音只保留三个本地工具；不用计划编号、版本号或完整性工具。"""

from voice.plan_tools import _tool

CONTROL_TOOLS = {
    "cancel_pending": _tool("取消两秒等待窗口里所有尚未执行的操作。已经发出的请求无法保证取消。", {}, []),
    "undo_recent": _tool(
        "撤销 Alpha 在本次程序会话刚创建的清单条目或提醒，无需额外确认。"
        "title 可指定刚创建的内容；不知道指哪个时先问清楚。未执行的操作用 cancel_pending。",
        {"title": {"type": "string"}}, [],
    ),
    "delete_item": _tool(
        "删除一项明确的清单条目或日程。先查询取得精确 ID，不能猜测。"
        "首次调用返回需要确认；只有用户随后明确同意，才再次调用同一目标并传 confirmed=true。"
        "用户提问、要求修改、意思含糊时不能确认。本次会话由 Alpha 创建的内容可直接撤销。",
        {"kind": {"type": "string", "enum": ["task", "reminder"]},
         "item_id": {"type": "string", "minLength": 1},
         "list_id": {"type": "string", "description": "清单条目必须提供；日程不需要"},
         "confirmed": {"type": "boolean", "default": False}}, ["kind", "item_id"],
    ),
}

RULES = """你是 Alpha，用简短自然的普通话回答，不用 Markdown，不朗读内部 ID。
开关灯和调亮度直接调用工具，成功不用再说一遍。查询直接调用工具。
添加清单、完成条目、创建提醒会先登记，两秒后执行；尚未执行时不能说已经完成。
用户说算了、不加了，用 cancel_pending 取消所有等待操作。改成别的内容时先取消，再登记新的内容。
用户说刚才那个不要了，用 undo_recent 撤销本次 Alpha 创建的东西；目标不清楚先问。
删除原有内容先查询 ID，再调用 delete_item。它要求确认时询问用户；只有之后用户明确同意才传 confirmed=true。
用户提问、改口或含糊回应不代表同意。不能声称工具没有完成的事已经完成；未知结果不能盲目重试。
“等等”“还有……”等未完成表达由代码延长等待，无需调用额外的收集或计划工具。
"""
