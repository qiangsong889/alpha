"""按 HTTP 请求隔离的工具计时旁路；不携带参数或结果，不影响执行。"""
from collections.abc import Callable
from contextvars import ContextVar
import logging

TOOL_TIMING_SINK: ContextVar[Callable[[dict], None] | None] = ContextVar('tool_timing_sink', default=None)


def emit_tool_timing(name: str, op_id: str, phase: str, **fields) -> None:
    sink = TOOL_TIMING_SINK.get()
    if sink is not None:
        try:
            sink({'name': name, 'operation_id': op_id, 'phase': phase, **fields})
        except Exception:
            logging.getLogger('alpha.timing').debug('工具计时回调失败', exc_info=True)
