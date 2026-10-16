"""邮件的两阶段发送。语义同意由模型判断，代码核对新输入、快照、期限和一次性资格。"""

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Callable

from integrations.tool_executor import format_for_model
from integrations.tool_gateway import EMAIL_CONFIRMATION, ToolGateway
from integrations.tool_policy import EMAIL_TOOLS


@dataclass
class EmailDecision:
    text: str
    is_error: bool = False
    preview: bool = False


@dataclass
class _Draft:
    name: str
    arguments: dict
    op_id: str
    token: str
    prompt: str
    created_at: float
    created_turn: int
    used: bool = False


class EmailConfirmation:
    """一次只保留一份待确认邮件；更换草稿会取消上一份未发送的操作。"""

    def __init__(self, gateway: ToolGateway, *, clock: Callable[[], float] = time.monotonic):
        self.gateway = gateway
        self.clock = clock
        self.draft: _Draft | None = None
        self.turn = 0
        self.turn_id = ""
        self.last_used_turn = -1
        self.expired = False

    def begin_turn(self, turn_id: str) -> None:
        if turn_id != self.turn_id:
            self.turn_id = turn_id
            self.turn += 1
        self._expire()

    def _expire(self) -> None:
        if self.draft and not self.draft.used and self.clock() - self.draft.created_at >= 120:
            self.gateway.cancel(self.draft.op_id)
            self.draft = None
            self.expired = True

    def close(self) -> None:
        if self.draft and not self.draft.used:
            self.gateway.cancel(self.draft.op_id)
        self.draft = None

    async def handle(self, name: str, arguments: dict) -> EmailDecision:
        self._expire()
        if name not in EMAIL_TOOLS or not self.turn_id:
            return EmailDecision("邮件确认流程尚未初始化，本次没有发送。", True)
        args = deepcopy(arguments)
        confirmed = args.pop("confirmed", False)
        if type(confirmed) is not bool:
            return EmailDecision("confirmed 必须是布尔值，本次没有发送。", True)
        expected = {"to", "subject", "body"} if name == "send_email" else {"message_id", "body"}
        if set(args) != expected or any(not isinstance(v, str) or not v.strip() for v in args.values()):
            self.close()  # 无效的修改不能留下仍可确认的旧草稿。
            return EmailDecision("邮件参数不完整或包含未知字段，本次没有发送。", True)
        unknown = self.gateway.store.recent_unknown(name, args)
        if unknown is not None:
            return EmailDecision("相同邮件的发送结果未知，可能已发出。不要重试，先查询已发送邮件核实。", True)
        draft = self.draft
        if confirmed:
            if draft is None:
                return EmailDecision("草稿已过期，请重新展示并确认。" if self.expired else
                                     "还没有待确认的草稿，本次没有发送。请先准备草稿。", True)
            if draft.used or self.last_used_turn == self.turn:
                return EmailDecision("此确认已经使用，本次没有再次发送。", True)
            if draft.name != name or draft.arguments != args:
                return await self._prepare(name, args, changed=True)
            if self.turn <= draft.created_turn:
                return EmailDecision("还没有收到草稿之后的新一句用户输入，本次没有发送。", True)
            # 本方法不在检查与消耗资格之间 await；同一轮多次工具调用无法重放。
            draft.used = True
            self.last_used_turn = self.turn
            if not self.gateway.approve(draft.op_id):
                return EmailDecision("草稿对应的操作已取消或处理，本次没有发送。", True)
            token = EMAIL_CONFIRMATION.set(draft.token)
            try:
                op = await self.gateway.execute(draft.op_id)
            finally:
                EMAIL_CONFIRMATION.reset(token)
            formatted = format_for_model(op)
            return EmailDecision(formatted["text"], formatted["is_error"])
        if draft and not draft.used and draft.name == name and draft.arguments == args:
            return EmailDecision(draft.prompt, preview=True)
        return await self._prepare(name, args, changed=draft is not None and not draft.used)

    async def _prepare(self, name: str, args: dict, *, changed: bool) -> EmailDecision:
        # 解析地址/线程是只读操作，也经过网关；不批准 send_email/reply_email。
        prepared = await self.gateway.submit(self.turn_id, "prepare_email", {"tool_name": name, "arguments": args})
        if prepared.status != "succeeded":
            formatted = format_for_model(prepared)
            return EmailDecision(formatted["text"], True)
        payload = prepared.result
        data = payload.get("structured_content", {}) if isinstance(payload, dict) else {}
        preview = data.get("preview")
        token = data.get("confirmation_token")
        if (not isinstance(preview, dict) or not isinstance(token, str) or not token
                or any(not isinstance(preview.get(k), str) for k in ("to", "subject", "body"))
                or preview["body"] != args["body"]):
            return EmailDecision("没有收到完整邮件草稿，本次没有发送。", True)
        if self.draft and not self.draft.used:
            self.gateway.cancel(self.draft.op_id)
        op = self.gateway.register(self.turn_id, name, args)
        prefix = "草稿已修改，上一份未发送。" if changed else "尚未发送。"
        if self.gateway.store.recent_succeeded(name, args) is not None:
            prefix += "之前发过相同邮件，这是新的一封。"
        prompt = (f"{prefix}\n收件人：{preview['to']}\n主题：{preview['subject']}\n"
                  f"正文：\n{preview['body']}\n确认发送吗？")
        self.draft = _Draft(name, deepcopy(args), op.id, token, prompt, self.clock(), self.turn)
        self.expired = False
        return EmailDecision(prompt, preview=True)
