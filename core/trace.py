"""把一轮对话产生的费用、操作和存档关联起来的上下文变量。

asyncio.create_task 会复制创建时的上下文，所以由某一轮触发的后台任务（例如自动记忆提取）
天然带着那一轮的 turn_id；跨多轮的后台工作（例如上下文压缩）需要在任务里显式清除 turn_id。
"""

from contextvars import ContextVar

CURRENT_SESSION: ContextVar[str | None] = ContextVar('alpha_current_session', default=None)
CURRENT_TURN: ContextVar[str | None] = ContextVar('alpha_current_turn', default=None)
