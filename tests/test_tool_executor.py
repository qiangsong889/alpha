"""真实 executor 的路由、发送边界和模型状态文本；全部使用假的 MCP 连接。"""

from contextlib import asynccontextmanager
from dataclasses import replace
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from mcp.types import CallToolResult, TextContent

import integrations.mcp_client as mcp_client
import integrations.tool_executor as module
from integrations.operations import OperationStore
from integrations.tool_gateway import ToolGateway, ToolNotSent


def result(text="完成", *, is_error=False):
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)

    async def test_personal_preflight_failures_are_definitely_not_sent(self):
        for worker in (None, Mock(closed=True, call=AsyncMock())):
            executor = module.make_executor(ha_url="", http_client=None, worker=worker)
            gateway = ToolGateway(self.store, executor)
            op = await gateway.submit("turn", "list_items", {})
            self.assertEqual(op.status, "failed")
            self.assertIn("ToolNotSent", op.result["error"])
            if worker is not None:
                worker.call.assert_not_awaited()

    async def test_missing_ha_configuration_is_definitely_not_sent(self):
        for url, client in (("", Mock(is_closed=False)), ("http://ha/api/mcp", None),
                            ("http://ha/api/mcp", Mock(is_closed=True))):
            executor = module.make_executor(ha_url=url, http_client=client, worker=None)
            with patch.object(module, "call_ha_tool", AsyncMock()) as call:
                with self.assertRaises(ToolNotSent):
                    await executor("homeassistant__GetLiveContext", {})
                call.assert_not_awaited()

    async def test_ha_initialization_failure_is_failed_but_call_failure_is_unknown(self):
        for phase, expected in (("transport", "failed"), ("initialize", "failed"),
                                ("call_tool", "unknown"), ("cleanup", "unknown")):
            with self.subTest(phase=phase):
                session = Mock(initialize=AsyncMock(), call_tool=AsyncMock(return_value=result("状态")))
                if phase == "initialize":
                    session.initialize.side_effect = OSError("握手失败")
                elif phase == "call_tool":
                    session.call_tool.side_effect = TimeoutError("工具结果超时")

                @asynccontextmanager
                async def transport(*args, **kwargs):
                    if phase == "transport":
                        raise OSError("无法连接")
                    yield Mock(), Mock()

                @asynccontextmanager
                async def session_context(*args):
                    yield session
                    if phase == "cleanup":
                        raise OSError("退出连接失败")

                executor = module.make_executor(ha_url="http://ha/api/mcp", http_client=Mock(is_closed=False), worker=None)
                with patch.object(mcp_client, "streamable_http_client", transport), \
                        patch.object(mcp_client, "ClientSession", session_context):
                    op = await ToolGateway(self.store, executor).submit("turn", "homeassistant__GetLiveContext", {})
                self.assertEqual(op.status, expected)
                if phase in {"transport", "initialize"}:
                    session.call_tool.assert_not_awaited()
                else:
                    session.call_tool.assert_awaited_once()

    async def test_worker_timeout_is_unknown(self):
        worker = Mock(closed=False, call=AsyncMock(side_effect=TimeoutError("可能已经发出")))
        op = await ToolGateway(self.store, module.make_executor(
            ha_url="", http_client=None, worker=worker,
        )).submit("turn", "list_items", {})
        self.assertEqual(op.status, "unknown")
        self.assertIn("不要重试", module.format_for_model(op)["text"])

    async def test_ha_actions_require_action_done_without_failures(self):
        executor = module.make_executor(ha_url="http://ha/api/mcp", http_client=Mock(is_closed=False), worker=None)
        for payload, ok in (({"response_type": "action_done", "data": {}}, True),
                            ({"response_type": "action_done", "data": {"failed": ["light.room"]}}, False),
                            ({"response_type": "error"}, False)):
            response = CallToolResult(content=[], structured_content={"response": payload})
            with patch.object(module, "call_ha_tool", AsyncMock(return_value=response)):
                outcome = await executor("intent__HassTurnOff", {"name": "客厅灯"})
            self.assertEqual(outcome.ok, ok)
            self.assertEqual(outcome.result["structured_content"]["response"], payload)

    async def test_read_only_and_personal_tools_use_is_error_and_return_text(self):
        worker = Mock(closed=False, call=AsyncMock())
        executor = module.make_executor(ha_url="http://ha/api/mcp", http_client=Mock(is_closed=False), worker=worker)
        for is_error in (False, True):
            worker.call.return_value = result("个人结果", is_error=is_error)
            personal = await executor("add_items", {"items": ["牛奶"]})
            self.assertEqual(personal.ok, not is_error)
            self.assertEqual(personal.result, "个人结果")
            with patch.object(module, "call_ha_tool", AsyncMock(return_value=result("客厅灯关着", is_error=is_error))):
                read = await executor("homeassistant__GetLiveContext", {})
            self.assertEqual(read.ok, not is_error)
            self.assertEqual(read.result, "客厅灯关着")


class ModelFormatTests(unittest.TestCase):
    def setUp(self):
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)
        self.op = self.store.create("turn", "add_items", {"items": ["牛奶"]})

    def test_all_statuses_report_facts(self):
        cases = (
            ("pending", None, False, "尚未执行"),
            ("approved", None, False, "尚未执行"),
            ("executing", None, False, "正在执行"),
            ("succeeded", "工具返回文字", False, "工具返回文字"),
            ("failed", {"error": "ToolNotSent: 未发送"}, True, "ToolNotSent: 未发送"),
            ("unknown", {"error": "连接断了"}, True, "结果未知，可能已经生效。不要重试，先查询确认"),
            ("cancelled", None, False, "已取消"),
        )
        for status, result_value, is_error, expected in cases:
            with self.subTest(status=status):
                formatted = module.format_for_model(replace(self.op, status=status, result=result_value))
                self.assertEqual(formatted["is_error"], is_error)
                self.assertIn(expected, formatted["text"])

    def test_duplicate_does_not_claim_current_operation_succeeded(self):
        previous = replace(self.op, id="previous", status="succeeded", result="已添加")
        current = replace(self.op, id="current", status="cancelled")
        formatted = module.format_for_model(current, duplicate_of=previous)
        self.assertFalse(formatted["is_error"])
        self.assertIn("本次没有执行", formatted["text"])
        self.assertIn("那是之前的那次", formatted["text"])
        self.assertIn("先向用户确认", formatted["text"])
        with self.assertRaises(ValueError):
            module.format_for_model(previous, duplicate_of=previous)

    def test_success_includes_document_id_alongside_text(self):
        payload = {"text": "已创建文档：采购计划", "structured_content": {
            "document_id": "doc-eval-01", "title": "采购计划"}}
        op = replace(self.op, name="create_document", status="succeeded", result=payload)
        formatted = module.format_for_model(op)
        text, data = formatted["text"].split("\n", 1)
        self.assertEqual(text, payload["text"])
        self.assertEqual(json.loads(data)["document_id"], "doc-eval-01")
        self.assertFalse(formatted["is_error"])

    def test_private_email_token_is_not_exposed_with_structured_results(self):
        payload = {"text": "草稿已准备", "structured_content": {
            "confirmation_token": "secret", "preview": {"to": "person@example.com"}}}
        text = module.format_for_model(replace(self.op, status="succeeded", result=payload))["text"]
        self.assertNotIn("secret", text)
        self.assertNotIn("confirmation_token", text)
        self.assertIn("person@example.com", text)
        payload["text"] = json.dumps(payload["structured_content"])
        text = module.format_for_model(replace(self.op, status="succeeded", result=payload))["text"]
        self.assertNotIn("secret", text)
        self.assertEqual(json.loads(text), {"preview": {"to": "person@example.com"}})


if __name__ == "__main__":
    unittest.main()
