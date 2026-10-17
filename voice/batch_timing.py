"""只比较同一进程的单调时钟；跨 HTTP 只传 ID 和服务端相对耗时。"""
from dataclasses import dataclass, field
import logging
import time
from uuid import uuid4

logger = logging.getLogger('alpha.qwen_alpha')


@dataclass
class BatchTiming:
    stopped_at: float | None = None
    transcribed_at: float | None = None
    submitted_at: float = field(default_factory=time.perf_counter)
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    http_started_at: float | None = None
    brain_done_at: float | None = None
    tool_events: list[dict] = field(default_factory=list)

    def elapsed(self, end: float) -> str:
        return f'{end - self.stopped_at:.2f}s' if self.stopped_at is not None else '未知（未收到停说事件）'

    def log_submission(self):
        logger.info('[批次 %s] 停说事件→转写齐 %s ｜ 停说事件→提交 %s', self.id,
                    self.elapsed(self.transcribed_at) if self.transcribed_at is not None else '未知',
                    self.elapsed(self.submitted_at))

    def tool(self, event):
        self.tool_events.append(dict(event))
        logger.info('[批次 %s] 工具 %s %s ｜ 服务端请求内 +%.3fs%s', self.id,
                    event['name'], '开始' if event['phase'] == 'started' else '结束',
                    event['elapsed_s'], f" ｜ 执行 {event['duration_s']:.3f}s" if 'duration_s' in event else '')
