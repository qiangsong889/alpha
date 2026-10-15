"""从真实的添加回执生成精确撤销请求；不登记、不批准、不执行工具。"""

from collections.abc import Iterable
from dataclasses import dataclass

from integrations.operations import Operation


@dataclass(frozen=True)
class CreatedTask:
    list_id: str
    task_id: str
    title: str


@dataclass(frozen=True)
class UndoRequest:
    source_operation_id: str
    list_id: str
    task_id: str
    title: str

    @property
    def name(self) -> str:
        return "delete_task"

    @property
    def arguments(self) -> dict[str, str]:
        return {"list_id": self.list_id, "task_id": self.task_id}


def created_tasks(op: Operation) -> tuple[CreatedTask, ...]:
    """只接受完整成功且与原请求对应的回执；旧的纯文字结果不能猜 ID。"""
    if op.name != "add_items" or op.status != "succeeded":
        raise ValueError("只能从已确认成功的添加操作生成撤销，其他结果需先核实")
    result = op.result
    data = result.get("structured_content") if isinstance(result, dict) else None
    if (not isinstance(data, dict) or type(data.get("version")) is not int
            or data["version"] != 1 or data.get("status") != "succeeded"):
        raise ValueError("缺少可用于撤销的创建回执，不能根据标题猜测 ID")
    list_id, entries, titles = data.get("list_id"), data.get("created"), op.arguments.get("items")
    if (not isinstance(list_id, str) or not list_id.strip() or not isinstance(entries, list)
            or not isinstance(titles, list) or not titles or len(entries) != len(titles)
            or data.get("list_name") != op.arguments.get("list_name")):
        raise ValueError("创建回执与原添加请求不匹配")
    tasks = []
    ids: set[str] = set()
    for entry, title in zip(entries, titles):
        if not isinstance(entry, dict):
            raise ValueError("创建回执缺少任务 ID")
        task_id = entry.get("task_id")
        if (not isinstance(task_id, str) or not task_id.strip() or task_id in ids
                or not isinstance(title, str) or not title.strip() or entry.get("title") != title):
            raise ValueError("创建回执中的任务 ID 或标题无效")
        ids.add(task_id)
        tasks.append(CreatedTask(list_id, task_id, title))
    return tuple(tasks)


def undo_add_items(op: Operation, *, task_ids: Iterable[str] | None = None) -> tuple[UndoRequest, ...]:
    """默认撤销本次全部新增；也可选择本次回执内的部分 ID，不影响原有同名条目。

    调用方仍需处理用户确认、重复撤销和结果未知。不能把此函数返回当成撤销成功。
    """
    tasks = created_tasks(op)
    if task_ids is None:
        selected = {task.task_id for task in tasks}
    else:
        if isinstance(task_ids, (str, bytes)):
            raise ValueError("task_ids 必须是 ID 列表")
        ids = tuple(task_ids)
        if any(not isinstance(key, str) or not key.strip() for key in ids):
            raise ValueError("task_ids 必须是非空字符串 ID")
        selected = set(ids)
        if not selected or len(selected) != len(ids):
            raise ValueError("选择的任务 ID 不能为空或重复")
        if not selected <= {task.task_id for task in tasks}:
            raise ValueError("只能撤销本次创建的任务 ID")
    return tuple(UndoRequest(op.id, task.list_id, task.task_id, task.title)
                 for task in tasks if task.task_id in selected)
