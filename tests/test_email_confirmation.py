"""确认状态用真网关/内存数据库验证，不发送任何真实邮件。"""

import asyncio
import unittest
from unittest.mock import AsyncMock

from brain.email_confirmation import EmailConfirmation
from integrations.operations import OperationStore
from integrations.tool_gateway import EMAIL_CONFIRMATION, ExecutorResult, ToolGateway
from integrations.tool_policy import get_policy


class EmailConfirmationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 10.
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.sent = []
        async def execute(name, args):
            if name == "prepare_email":
                return ExecutorResult(True, {"structured_content": {"confirmation_token": "opaque",
                    "preview": {"to": args["arguments"].get("to", "resolved@example.com"),
                                "subject": args["arguments"].get("subject", "Re: subject"),
                                "body": args["arguments"]["body"]}}})
            self.assertEqual(EMAIL_CONFIRMATION.get(), "opaque")
            self.sent.append((name, args))
            return ExecutorResult(True, "邮件已发送")
        self.executor = AsyncMock(side_effect=execute)
        self.gateway = ToolGateway(self.store, self.executor)
        self.coordinator = EmailConfirmation(self.gateway, clock=lambda: self.now)
        self.coordinator.begin_turn("first")
        self.args = {"to": "person@example.com", "subject": "会议", "body": "明天见"}

    async def prepare(self):
        result = await self.coordinator.handle("send_email", self.args)
        self.assertTrue(result.preview)
        return result

    async def test_first_call_previews_and_same_user_turn_cannot_auto_confirm(self):
        self.assertEqual(get_policy("send_email").commit, "confirm")
        self.assertEqual(get_policy("reply_email").commit, "confirm")
        result = await self.prepare()
        self.assertIn("person@example.com", result.text)
        self.assertIn("明天见", result.text)
        op_id = self.coordinator.draft.op_id
        self.assertEqual(self.store.get(op_id).status, "pending")
        self.coordinator.begin_turn("first")
        rejected = await self.coordinator.handle("send_email", {**self.args, "confirmed": True})
        self.assertTrue(rejected.is_error)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.store.get(op_id).status, "pending")

    async def test_next_user_turn_can_confirm_only_once(self):
        await self.prepare()
        op_id = self.coordinator.draft.op_id
        self.coordinator.begin_turn("user-says-yes")
        result = await self.coordinator.handle("send_email", {**self.args, "confirmed": True})
        self.assertFalse(result.is_error)
        self.assertEqual(self.store.get(op_id).status, "succeeded")
        self.assertEqual(self.sent, [("send_email", self.args)])
        self.assertIsNone(EMAIL_CONFIRMATION.get())
        for turn in ("user-says-yes", "another-turn"):
            self.coordinator.begin_turn(turn)
            self.assertTrue((await self.coordinator.handle("send_email", {**self.args, "confirmed": True})).is_error)
        self.assertEqual(len(self.sent), 1)

    async def test_changed_parameters_cancel_old_draft_and_require_new_confirmation(self):
        await self.prepare()
        old = self.coordinator.draft.op_id
        self.coordinator.begin_turn("change")
        changed_args = {**self.args, "body": "改成后天"}
        result = await self.coordinator.handle("send_email", {**changed_args, "confirmed": True})
        self.assertTrue(result.preview)
        self.assertIn("改成后天", result.text)
        self.assertEqual(self.store.get(old).status, "cancelled")
        self.assertTrue((await self.coordinator.handle("send_email", {**changed_args, "confirmed": True})).is_error)
        self.assertFalse(self.sent)
        self.coordinator.begin_turn("approve-changed")
        self.assertFalse((await self.coordinator.handle("send_email", {**changed_args, "confirmed": True})).is_error)
        self.assertEqual(self.sent, [("send_email", changed_args)])

    async def test_expiry_and_restart_do_not_restore_confirmation(self):
        await self.prepare()
        old = self.coordinator.draft.op_id
        self.now += 121
        self.coordinator.begin_turn("too-late")
        result = await self.coordinator.handle("send_email", {**self.args, "confirmed": True})
        self.assertTrue(result.is_error)
        self.assertIn("过期", result.text)
        self.assertEqual(self.store.get(old).status, "cancelled")
        new = EmailConfirmation(self.gateway)
        new.begin_turn("restart")
        self.assertTrue((await new.handle("send_email", {**self.args, "confirmed": True})).is_error)
        self.assertFalse(self.sent)

    async def test_close_cancels_unsubmitted_draft(self):
        await self.prepare()
        op_id = self.coordinator.draft.op_id
        self.coordinator.close()
        self.assertEqual(self.store.get(op_id).status, "cancelled")
        self.assertFalse(self.sent)

    async def test_reply_preview_uses_resolved_address_and_preserves_original_arguments(self):
        args = {"message_id": "original", "body": "收到"}
        result = await self.coordinator.handle("reply_email", args)
        self.assertIn("resolved@example.com", result.text)
        self.assertIn("Re: subject", result.text)
        self.coordinator.begin_turn("yes")
        self.assertFalse((await self.coordinator.handle("reply_email", {**args, "confirmed": True})).is_error)
        self.assertEqual(self.sent, [("reply_email", args)])

    async def test_timeout_is_unknown_and_does_not_restore_confirmation(self):
        await self.prepare()
        op_id = self.coordinator.draft.op_id
        self.coordinator.begin_turn("yes")
        self.executor.side_effect = TimeoutError("maybe sent")
        result = await self.coordinator.handle("send_email", {**self.args, "confirmed": True})
        self.assertTrue(result.is_error)
        self.assertIn("不要重试", result.text)
        self.assertEqual(self.store.get(op_id).status, "unknown")
        count = self.executor.await_count
        self.coordinator.begin_turn("another")
        self.assertTrue((await self.coordinator.handle("send_email", self.args)).is_error)
        self.assertEqual(self.executor.await_count, count)

    async def test_task_cancellation_marks_unknown_and_consumes_confirmation(self):
        await self.prepare()
        self.coordinator.begin_turn("yes")
        entered = asyncio.Event()
        async def hanging(*args):
            entered.set()
            await asyncio.Future()
        self.executor.side_effect = hanging
        task = asyncio.create_task(self.coordinator.handle("send_email", {**self.args, "confirmed": True}))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.store.get(self.coordinator.draft.op_id).status, "unknown")
        self.assertTrue(self.coordinator.draft.used)

    async def test_confirm_without_preview_and_non_boolean_flag_are_rejected(self):
        self.assertTrue((await self.coordinator.handle("send_email", {**self.args, "confirmed": True})).is_error)
        self.assertTrue((await self.coordinator.handle("send_email", {**self.args, "confirmed": "true"})).is_error)
        self.executor.assert_not_called()

    async def test_second_different_draft_supersedes_first(self):
        await self.prepare()
        old = self.coordinator.draft.op_id
        result = await self.coordinator.handle("send_email", {**self.args, "to": "other@example.com"})
        self.assertTrue(result.preview)
        self.assertIn("上一份未发送", result.text)
        self.assertEqual(self.store.get(old).status, "cancelled")
        self.assertFalse(self.sent)
