"""提醒撤销和写入核实：全部使用假的 Google 请求。"""

import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

with patch("dotenv.load_dotenv"):
    from servers import personal_server as server

from integrations.mcp_client import load_tools, merge_tools
from integrations.operations import OperationStore
from integrations.tool_executor import make_executor
from integrations.tool_gateway import ToolGateway


class VerificationToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tasks = Mock()
        self.calendar = MagicMock()
        self.patches = [patch.object(server, "get_tasks_service", return_value=self.tasks),
                        patch.object(server, "_find_list_id", return_value="list"),
                        patch.object(server, "get_credentials", return_value=object()),
                        patch.object(server, "build", return_value=self.calendar),
                        patch.dict(server.os.environ, {"REMINDER_TIMEZONE": "America/Los_Angeles"})]
        for context in self.patches:
            context.start()
            self.addCleanup(context.stop)
        self.events = self.calendar.__enter__.return_value.events.return_value

    async def call(self, name, args):
        return await server.mcp.call_tool(name, args)

    async def test_reminder_uses_stable_id_returns_receipt_and_deletes_exact_id(self):
        event_id = server._event_id("operation-1")
        self.events.insert.return_value.execute.return_value = {"id": event_id}
        result = await self.call("create_reminder", {
            "title": "开会", "local_time": "2026-11-03T08:00:00", "operation_id": "operation-1",
        })
        body = self.events.insert.call_args.kwargs["body"]
        self.assertEqual(body["id"], event_id)
        self.assertEqual(body["start"]["dateTime"], "2026-11-03T08:00:00-08:00")
        self.assertEqual(result.structured_content["event_id"], event_id)
        deleted = await self.call("delete_reminder", {"event_id": event_id})
        self.assertFalse(deleted.is_error)
        self.events.delete.assert_called_once_with(calendarId="primary", eventId=event_id)

    async def test_reminder_timeout_returns_unknown_with_known_request_id(self):
        self.events.insert.return_value.execute.side_effect = TimeoutError("返回丢失")
        result = await self.call("create_reminder", {
            "title": "开会", "local_time": "2026-11-03T08:00:00", "operation_id": "operation-1",
        })
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["status"], "unknown")
        self.assertEqual(result.structured_content["event_id"], server._event_id("operation-1"))

    async def test_inspect_reminder_by_operation_id_and_deleted_event(self):
        self.events.get.return_value.execute.return_value = {"id": "id", "summary": "开会", "status": "confirmed"}
        result = await self.call("inspect_reminder", {"operation_id": "operation-1"})
        self.assertTrue(result.structured_content["present"])
        self.events.get.assert_called_once_with(calendarId="primary", eventId=server._event_id("operation-1"))
        self.events.get.return_value.execute.side_effect = HttpError(Mock(status=404, reason="Not Found"), b"{}")
        result = await self.call("inspect_reminder", {"event_id": "gone"})
        self.assertFalse(result.structured_content["present"])

    async def test_find_tasks_by_marker_across_pages_not_by_equal_title(self):
        self.tasks.tasks.return_value.list.return_value.execute.side_effect = [
            {"items": [{"id": "old", "title": "牛奶"}], "nextPageToken": "page2"},
            {"items": [{"id": "new", "title": "牛奶", "notes": "alpha-operation:operation-1:0"}]},
        ]
        result = await self.call("find_created_items", {"list_name": "购物清单", "operation_id": "operation-1", "items": ["牛奶"]})
        self.assertEqual(result.structured_content["created"], [{"task_id": "new", "title": "牛奶"}])
        self.assertEqual(result.structured_content["status"], "succeeded")
        self.assertEqual(self.tasks.tasks.return_value.list.call_args.kwargs["pageToken"], "page2")

    async def test_duplicate_markers_cannot_prove_single_success(self):
        self.tasks.tasks.return_value.list.return_value.execute.return_value = {"items": [
            {"id": key, "title": "牛奶", "notes": "alpha-operation:op:0"} for key in ("one", "two")
        ]}
        result = await self.call("find_created_items", {"list_name": "购物清单", "operation_id": "op", "items": ["牛奶"]})
        self.assertEqual(result.structured_content["status"], "unknown")

    async def test_inspect_task_is_read_only_and_404_means_absent(self):
        self.tasks.tasks.return_value.get.return_value.execute.side_effect = HttpError(Mock(status=404, reason="Not Found"), b"{}")
        result = await self.call("inspect_task", {"list_id": "list", "task_id": "gone"})
        self.assertFalse(result.structured_content["present"])
        self.tasks.tasks.return_value.delete.assert_not_called()

    async def test_gateway_injects_operation_marker_and_model_schema_hides_it(self):
        store = OperationStore(":memory:")
        self.addCleanup(store.close)
        self.tasks.tasks.return_value.insert.return_value.execute.return_value = {"id": "new"}
        worker = Mock(closed=False, call=AsyncMock(side_effect=self.call))
        gateway = ToolGateway(store, make_executor(ha_url="", http_client=None, worker=worker))
        op = gateway.register("turn", "add_items", {"list_name": "购物清单", "items": ["牛奶"]})
        gateway.approve(op.id)
        await gateway.execute(op.id)
        self.assertEqual(self.tasks.tasks.return_value.insert.call_args.kwargs["body"]["notes"], f"alpha-operation:{op.id}:0")
        session = Mock(list_tools=AsyncMock(return_value=Mock(tools=await server.mcp.list_tools(), next_cursor=None)))
        tools = await load_tools(session)
        catalog, _ = merge_tools({"personal": tools})
        for tool in catalog:
            self.assertNotIn("operation_id", tool["input_schema"].get("properties", {}))
        self.assertNotIn("delete_reminder", {tool["name"] for tool in catalog})
