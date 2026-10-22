"""假的 Google service；不读取真实 .env、凭据，不调用 Google 或麦克风。"""

import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

# personal_server 导入时会 load_dotenv，测试不允许读取用户的实际配置。
with patch("dotenv.load_dotenv"):
    from servers import personal_server as server

from integrations.mcp_client import merge_tools
from integrations.operations import OperationStore
from integrations.tool_executor import make_executor, format_for_model
from integrations.tool_gateway import ToolGateway


class PersonalTaskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.api = Mock()
        self.resource = self.api.tasks.return_value
        self.patches = [patch.object(server, "get_tasks_service", return_value=self.api),
                        patch.object(server, "_find_list_id", return_value="list-1")]
        for mock_patch in self.patches:
            mock_patch.start()
            self.addCleanup(mock_patch.stop)
        self.store = OperationStore(":memory:")
        self.addCleanup(self.store.close)

    async def call(self, name, args):
        # 使用真实 MCP 工具转换，验证 content 和 structured_content 均穿过 SDK。
        return await server.mcp.call_tool(name, args)

    async def execute(self, name, args):
        worker = Mock(closed=False, call=AsyncMock(side_effect=self.call))
        gateway = ToolGateway(self.store, make_executor(ha_url="", http_client=None, worker=worker))
        op = await gateway.submit("turn", name, args)
        gateway.approve(op.id)
        return gateway, await gateway.execute(op.id)

    async def test_receipts_survive_mcp_gateway_and_sqlite_then_undo_only_new_task(self):
        inventory = {"old-milk": "牛奶"}

        def inserted():
            inventory["new-milk"] = "牛奶"
            return {"id": "new-milk", "title": "牛奶"}

        self.resource.insert.return_value.execute.side_effect = inserted

        def delete(**kwargs):
            return Mock(execute=Mock(side_effect=lambda: inventory.pop(kwargs["task"])))

        self.resource.delete.side_effect = delete
        gateway, added = await self.execute("add_items", {"list_name": "购物清单", "items": ["牛奶"]})
        self.assertEqual(added.status, "succeeded")
        saved = self.store.get(added.id)
        self.assertEqual(saved.result["structured_content"]["created"][0]["task_id"], "new-milk")
        text, data = format_for_model(saved)["text"].split("\n", 1)
        self.assertEqual(text, "已添加到 购物清单：牛奶")
        self.assertEqual(json.loads(data), saved.result["structured_content"])
        receipt = saved.result["structured_content"]
        undo = await gateway.submit("undo-turn", "delete_task", {
            "list_id": receipt["list_id"], "task_id": receipt["created"][0]["task_id"],
        })
        self.assertEqual(undo.status, "pending")  # 删除不会直接执行。
        self.resource.delete.assert_not_called()
        gateway.approve(undo.id)
        removed = await gateway.execute(undo.id)
        self.assertEqual(removed.status, "succeeded")
        self.assertEqual(inventory, {"old-milk": "牛奶"})
        self.resource.delete.assert_called_once_with(tasklist="list-1", task="new-milk")
        self.assertEqual(self.store.get(added.id).status, "succeeded")

    async def test_duplicate_titles_receive_distinct_ids(self):
        self.resource.insert.return_value.execute.side_effect = [{"id": "one"}, {"id": "two"}]
        result = await self.call("add_items", {"list_name": "购物清单", "items": ["牛奶", "牛奶"]})
        self.assertEqual([item["task_id"] for item in result.structured_content["created"]], ["one", "two"])
        self.assertFalse(result.is_error)

    async def test_partial_creation_preserves_ids_and_unknown_state_without_retry(self):
        self.resource.insert.return_value.execute.side_effect = [{"id": "milk"}, TimeoutError("断线")]
        _, op = await self.execute("add_items", {
            "list_name": "购物清单", "items": ["牛奶", "鸡蛋", "酱油"],
        })
        self.assertEqual(op.status, "unknown")
        data = self.store.get(op.id).result["structured_content"]
        self.assertEqual(data["created"], [{"task_id": "milk", "title": "牛奶"}])
        self.assertEqual(data["unconfirmed"], {"title": "鸡蛋"})
        self.assertEqual(data["unattempted"], ["酱油"])
        self.assertEqual(self.resource.insert.call_count, 2)
        self.assertIn("不要重试", format_for_model(op)["text"])

    async def test_missing_insert_id_is_unknown_not_success(self):
        self.resource.insert.return_value.execute.return_value = {}
        _, op = await self.execute("add_items", {"list_name": "购物清单", "items": ["牛奶"]})
        self.assertEqual(op.status, "unknown")
        self.assertEqual(op.result["structured_content"]["created"], [])

    async def test_delete_timeout_is_unknown_and_never_retried(self):
        self.resource.delete.return_value.execute.side_effect = TimeoutError("断线")
        gateway, op = await self.execute("delete_task", {"list_id": "list-1", "task_id": "new-milk"})
        self.assertEqual(op.status, "unknown")
        await gateway.execute(op.id)
        self.resource.delete.assert_called_once()
        server._find_list_id.assert_not_called()  # ID 定位，不按名字重新找清单。

    async def test_invalid_inputs_do_not_call_google(self):
        for args in ({"list_name": "", "items": ["牛奶"]},
                     {"list_name": "购物清单", "items": []}):
            with self.assertRaises(Exception):
                await self.call("add_items", args)
        for args in ({"list_id": "", "task_id": "id"}, {"list_id": "list", "task_id": " "}):
            with self.assertRaises(Exception):
                await self.call("delete_task", args)
        server.get_tasks_service.assert_not_called()

    async def test_delete_not_exposed_to_model_catalog(self):
        tools, owners = merge_tools({"personal": [
            {"name": "delete_task", "input_schema": {}},
            {"name": "add_items", "input_schema": {}},
        ]})
        self.assertEqual([tool["name"] for tool in tools], ["add_items"])
        self.assertNotIn("delete_task", owners)
