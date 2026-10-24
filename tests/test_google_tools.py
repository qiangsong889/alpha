"""通过真实 MCP 与网关连接假的 Google API；不访问凭据、网络或真实邮箱。"""

import base64
from email import policy
from email.parser import BytesParser
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httplib2
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from googleapiclient.http import HttpRequest
from google.auth.credentials import AnonymousCredentials
from google.auth.exceptions import RefreshError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

with patch("dotenv.load_dotenv"):
    from servers import personal_server
from servers import google_tools as tools
from servers.google_auth import SCOPES, YOUTUBE_SCOPES
from integrations.mcp_client import load_tools, merge_tools
from integrations.operations import OperationStore
from integrations.tool_executor import PERSONAL_TOOLS, make_executor, format_for_model
from integrations.tool_gateway import EMAIL_CONFIRMATION, ToolGateway
from integrations.tool_policy import GOOGLE_TOOLS, GOOGLE_READ_TOOLS, GOOGLE_WRITE_TOOLS, get_policy


def execute_result(resource, method, value):
    request = getattr(resource, method).return_value
    request.execute.return_value = value
    return request


def http_error(status):
    return HttpError(httplib2.Response({"status": str(status)}), b'{"error":{"message":"sensitive"}}')


def encoded(text):
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


class GoogleToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tools._MAIL_DRAFTS.clear()
        self.addCleanup(tools._MAIL_DRAFTS.clear)
        self.api = Mock()
        self.patch = patch.object(tools, "_service", return_value=self.api)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.env = patch.dict("os.environ", {"REMINDER_TIMEZONE": "America/Los_Angeles"})
        self.env.start()
        self.addCleanup(self.env.stop)

    async def call(self, name, args):
        return await personal_server.mcp.call_tool(name, args)

    async def wire_call(self, name, args):
        """和 stdio 上的 MCP 服务端一样：工具抛出的异常变成 is_error 结果，而不是让客户端收到异常。"""
        try:
            return await self.call(name, args)
        except Exception as exc:
            return CallToolResult(content=[TextContent(type="text", text=str(exc))], is_error=True)

    def test_docs_nested_tabs_and_tables_are_read(self):
        paragraph = {"paragraph": {"elements": [{"textRun": {"content": "正文\n"}}]}}
        execute_result(self.api.documents.return_value, "get", {"title": "笔记", "tabs": [{
            "tabProperties": {"tabId": "first", "title": "第一节"},
            "documentTab": {"body": {"content": [paragraph]}},
            "childTabs": [{"tabProperties": {"tabId": "child", "title": "第二节"},
                "documentTab": {"body": {"content": [{"table": {"tableRows": [{"tableCells": [
                    {"content": [paragraph]}]}]}}]}}}]}]})
        result = tools.read_document("doc")
        self.assertFalse(result.is_error)
        self.assertIn("第二节", result.content[0].text)
        self.assertEqual(result.content[0].text.count("正文"), 2)
        self.api.documents.return_value.get.assert_called_once_with(documentId="doc", includeTabsContent=True)

    def test_create_and_append_document_use_id_tab_and_revision(self):
        docs = self.api.documents.return_value
        execute_result(docs, "create", {"documentId": "new-doc"})
        self.assertEqual(tools.create_document("新笔记").structured_content["document_id"], "new-doc")
        execute_result(docs, "get", {"revisionId": "r1", "tabs": [{"tabProperties": {"tabId": "tab"}}]})
        execute_result(docs, "batchUpdate", {})
        self.assertFalse(tools.append_document("new-doc", "原样\n").is_error)
        self.assertEqual(docs.batchUpdate.call_args.kwargs["body"], {
            "requests": [{"insertText": {"text": "原样\n", "endOfSegmentLocation": {"tabId": "tab"}}}],
            "writeControl": {"requiredRevisionId": "r1"}})
        with self.assertRaisesRegex(Exception, "tab_id"):
            tools.append_document("doc", "x", "nonexistent")
        self.assertEqual(docs.batchUpdate.call_count, 1)

    def test_drive_search_escapes_query_and_preserves_pagination(self):
        files = self.api.files.return_value
        execute_result(files, "list", {"files": [{"id": "a", "name": "笔记"}], "nextPageToken": "next"})
        result = tools.search_drive("a' or trashed = true", 3, "previous")
        self.assertEqual(result.structured_content["nextPageToken"], "next")
        self.assertIn("a\\' or trashed = true", files.list.call_args.kwargs["q"])
        self.assertTrue(files.list.call_args.kwargs["q"].startswith("trashed = false"))
        self.assertEqual(files.list.call_args.kwargs["pageToken"], "previous")

    def test_drive_export_and_text_download_reject_binary_and_large_files(self):
        files = self.api.files.return_value
        for mime, method in [("application/vnd.google-apps.document", "export"), ("text/plain", "get_media")]:
            execute_result(files, "get", {"mimeType": mime, "name": "笔记"})
            execute_result(files, method, "中文正文".encode())
            self.assertIn("中文正文", tools.read_drive_file("file").content[0].text)
        execute_result(files, "get", {"mimeType": "application/pdf"})
        with self.assertRaisesRegex(Exception, "只支持"):
            tools.read_drive_file("file")
        execute_result(files, "get", {"mimeType": "text/plain", "size": 3_000_000})
        with self.assertRaisesRegex(Exception, "2 MB"):
            tools.read_drive_file("file")

    def test_create_folder_returns_receipt_and_parent(self):
        files = self.api.files.return_value
        execute_result(files, "create", {"id": "folder", "name": "资料"})
        result = tools.create_drive_folder("资料", "parent")
        self.assertEqual(result.structured_content["id"], "folder")
        self.assertEqual(files.create.call_args.kwargs["body"]["parents"], ["parent"])

    def test_calendar_list_and_event_query_support_all_day_and_pagination(self):
        execute_result(self.api.calendarList.return_value, "list", {"items": [{"id": "work", "summary": "工作"}]})
        self.assertEqual(tools.list_calendars().structured_content["calendars"][0]["id"], "work")
        events = self.api.events.return_value
        execute_result(events, "list", {"items": [{"id": "e", "summary": "假日", "start": {"date": "2026-10-16"}}],
                                        "nextPageToken": "next"})
        result = tools.list_events("2026-10-16T00:00:00", "2026-10-17T00:00:00", "work", page_token="previous")
        self.assertEqual(result.structured_content["events"][0]["start"], {"date": "2026-10-16"})
        args = events.list.call_args.kwargs
        self.assertTrue(args["timeMin"].endswith("-07:00"))
        self.assertEqual(args["pageToken"], "previous")
        self.assertEqual(result.structured_content["next_page_token"], "next")

    def test_calendar_validates_dates_before_requests_and_calculates_dst(self):
        events = self.api.events.return_value
        for start, end in [("2026-03-08T02:30:00", "2026-03-08T04:00:00"),
                           ("2026-11-01T01:30:00", "2026-11-01T03:00:00"),
                           ("2026-10-16T09:00:00", "2026-10-16T08:00:00"),
                           ("2026-10-16T08:00:00-07:00", "2026-10-16T09:00:00")]:
            with self.assertRaises(Exception):
                tools.create_event("会", start, end)
        self.api.events.assert_not_called()
        execute_result(events, "insert", {"id": "event"})
        result = tools.create_event("会", "2026-11-03T08:00:00", "2026-11-03T09:00:00", operation_id="op")
        body = events.insert.call_args.kwargs["body"]
        self.assertTrue(body["start"]["dateTime"].endswith("-08:00"))
        self.assertEqual(body["id"], tools.hashlib.sha256(b"op").hexdigest())
        self.assertEqual(result.structured_content["event_id"], "event")
        self.assertNotIn("attendees", body)

    def test_gmail_search_metadata_and_read_multipart_never_mutate(self):
        messages = self.api.users.return_value.messages.return_value
        execute_result(messages, "list", {"messages": [{"id": "m"}], "nextPageToken": "next"})
        execute_result(messages, "get", {"id": "m", "threadId": "t", "payload": {
            "headers": [{"name": "Subject", "value": "标题"}], "mimeType": "multipart/mixed", "parts": [
                {"mimeType": "text/plain", "body": {"data": encoded("正文")}},
                {"mimeType": "text/plain", "filename": "secret.txt", "body": {"data": encoded("附件不可见")}}]}})
        found = tools.search_email("is:unread", page_token="previous")
        self.assertEqual(found.structured_content["next_page_token"], "next")
        self.assertFalse(messages.list.call_args.kwargs["includeSpamTrash"])
        result = tools.read_email("m")
        self.assertIn("正文", result.content[0].text)
        self.assertNotIn("附件不可见", result.content[0].text)
        messages.modify.assert_not_called()
        messages.trash.assert_not_called()
        messages.delete.assert_not_called()

    def test_gmail_html_fallback_and_charset(self):
        messages = self.api.users.return_value.messages.return_value
        execute_result(messages, "get", {"payload": {"mimeType": "text/html", "body": {
            "data": encoded("<p>你好 &amp; 世界</p><script>bad()</script>")}}})
        result = tools.read_email("m")
        self.assertIn("你好 & 世界", result.content[0].text)
        self.assertNotIn("bad()", result.content[0].text)
        execute_result(messages, "get", {"payload": {"mimeType": "text/plain", "headers": [
            {"name": "Content-Type", "value": "text/plain; charset=gb18030"}], "body": {
            "data": base64.urlsafe_b64encode("中文".encode("gb18030")).decode()}}})
        self.assertIn("中文", tools.read_email("m").content[0].text)

    def setup_send(self):
        users = self.api.users.return_value
        execute_result(users, "getProfile", {"emailAddress": "owner@example.com"})
        execute_result(users.messages.return_value, "send", {"id": "sent", "threadId": "thread"})
        return users.messages.return_value

    def test_send_email_encodes_chinese_mime_without_draft_or_delete(self):
        messages = self.setup_send()
        args = {"to": "person@example.com", "subject": "会议", "body": "明天见"}
        prepared = tools.prepare_email("send_email", args)
        messages.send.assert_not_called()
        result = tools.send_email(**args, confirmed=True, confirmation_token=prepared.structured_content["confirmation_token"])
        request = messages.send.call_args.kwargs
        parsed = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(request["body"]["raw"]))
        self.assertEqual(str(parsed["Subject"]), "会议")
        self.assertEqual(str(parsed["From"]), "owner@example.com")
        self.assertIn("明天见", parsed.get_content())
        self.assertEqual(result.structured_content["message_id"], "sent")
        self.api.users.return_value.drafts.assert_not_called()
        messages.delete.assert_not_called()

    def test_reply_uses_reply_to_and_original_thread_headers(self):
        messages = self.setup_send()
        execute_result(messages, "get", {"threadId": "original-thread", "payload": {"headers": [
            {"name": "From", "value": "sender@example.com"}, {"name": "Reply-To", "value": "reply@example.com"},
            {"name": "Subject", "value": "会议"}, {"name": "Message-ID", "value": "<old@example.com>"},
            {"name": "References", "value": "<first@example.com>"}]}})
        prepared = tools.prepare_email("reply_email", {"message_id": "original", "body": "收到"})
        messages.send.assert_not_called()
        self.assertFalse(tools.reply_email("original", "收到", confirmed=True,
            confirmation_token=prepared.structured_content["confirmation_token"]).is_error)
        body = messages.send.call_args.kwargs["body"]
        parsed = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(body["raw"]))
        self.assertEqual(body["threadId"], "original-thread")
        self.assertEqual(str(parsed["To"]), "reply@example.com")
        self.assertEqual(str(parsed["In-Reply-To"]), "<old@example.com>")
        self.assertIn("<first@example.com>", str(parsed["References"]))
        self.assertEqual(str(parsed["Subject"]), "Re: 会议")

    def test_email_cannot_send_without_snapshot_and_token_is_single_use(self):
        messages = self.setup_send()
        args = {"to": "person@example.com", "subject": "s", "body": "body"}
        for confirmed in (False, True):
            with self.assertRaises(ToolError):
                tools.send_email(**args, confirmed=confirmed)
        prepared = tools.prepare_email("send_email", args)
        token = prepared.structured_content["confirmation_token"]
        with self.assertRaises(ToolError):
            tools.send_email(**{**args, "to": "other@example.com"}, confirmed=True, confirmation_token=token)
        messages.send.assert_not_called()
        self.assertFalse(tools.send_email(**args, confirmed=True, confirmation_token=token).is_error)
        with self.assertRaises(ToolError):
            tools.send_email(**args, confirmed=True, confirmation_token=token)
        self.assertEqual(messages.send.call_count, 1)

    def test_reply_snapshot_does_not_reresolve_changed_recipient(self):
        messages = self.setup_send()
        request = execute_result(messages, "get", {"threadId": "original", "payload": {"headers": [
            {"name": "Reply-To", "value": "original@example.com"},
            {"name": "Subject", "value": "old subject"},
            {"name": "Message-ID", "value": "<original@example.com>"}]}})
        args = {"message_id": "original", "body": "收到"}
        prepared = tools.prepare_email("reply_email", args)
        request.execute.return_value = {"threadId": "changed", "payload": {"headers": [
            {"name": "Reply-To", "value": "changed@example.com"}]}}
        self.assertFalse(tools.reply_email(**args, confirmed=True,
            confirmation_token=prepared.structured_content["confirmation_token"]).is_error)
        payload = messages.send.call_args.kwargs["body"]
        parsed = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(payload["raw"]))
        self.assertEqual(str(parsed["To"]), "original@example.com")
        self.assertEqual(payload["threadId"], "original")
        self.assertEqual(messages.get.call_count, 1)

    def test_expired_or_unknown_delivery_cannot_reuse_token(self):
        messages = self.setup_send()
        args = {"to": "person@example.com", "subject": "s", "body": "body"}
        prepared = tools.prepare_email("send_email", args)
        token = prepared.structured_content["confirmation_token"]
        tools._MAIL_DRAFTS[token]["created_at"] -= 121
        with self.assertRaises(ToolError):
            tools.send_email(**args, confirmed=True, confirmation_token=token)
        messages.send.assert_not_called()
        prepared = tools.prepare_email("send_email", args)
        token = prepared.structured_content["confirmation_token"]
        messages.send.return_value.execute.side_effect = TimeoutError()
        result = tools.send_email(**args, confirmed=True, confirmation_token=token)
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["status"], "unknown")
        with self.assertRaises(ToolError):
            tools.send_email(**args, confirmed=True, confirmation_token=token)
        self.assertEqual(messages.send.call_count, 1)

    def sent_to(self, messages):
        raw = messages.send.call_args.kwargs["body"]["raw"]
        return str(BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(raw))["To"])

    def test_me_resolves_to_owner_email_or_gmail_account_and_preview_shows_it(self):
        # 语音念不准邮箱：“发给我自己”由代码填地址，确认时听到的是实际地址。
        messages = self.setup_send()
        for owner, expected in (("", "owner@example.com"), ("me@yahoo.example", "me@yahoo.example")):
            with self.subTest(owner=owner), patch.dict("os.environ", {"OWNER_EMAIL": owner}):
                args = {"to": "me", "subject": "测试", "body": "正文"}
                prepared = tools.prepare_email("send_email", args)
                self.assertEqual(prepared.structured_content["preview"]["to"], expected)
                tools.send_email(**args, confirmed=True,
                                 confirmation_token=prepared.structured_content["confirmation_token"])
                self.assertEqual(self.sent_to(messages), expected)

    def test_me_can_be_mixed_with_other_recipients(self):
        self.setup_send()
        with patch.dict("os.environ", {"OWNER_EMAIL": ""}):
            prepared = tools.prepare_email("send_email", {"to": "张三 <z@example.com>, ME", "subject": "s", "body": "b"})
        self.assertEqual(prepared.structured_content["preview"]["to"], "张三 <z@example.com>, owner@example.com")

    def test_reply_address_me_is_not_treated_as_the_owner(self):
        # 回复地址来自原邮件，外人写 Reply-To: me 不能变成发给用户自己（也不是有效地址）。
        messages = self.setup_send()
        execute_result(messages, "get", {"threadId": "t", "payload": {"headers": [
            {"name": "Reply-To", "value": "me"}, {"name": "Subject", "value": "s"},
            {"name": "Message-ID", "value": "<x@example.com>"}]}})
        with self.assertRaises(ToolError):
            tools.prepare_email("reply_email", {"message_id": "m", "body": "收到"})

    def test_invalid_owner_email_is_rejected(self):
        self.setup_send()
        with patch.dict("os.environ", {"OWNER_EMAIL": "强颂八八九"}), self.assertRaises(ToolError):
            tools.prepare_email("send_email", {"to": "me", "subject": "s", "body": "b"})

    def test_send_rejects_missing_address_and_header_injection_before_api(self):
        for to, subject in [("小王", "Hi"), ("a@example.com\nBcc: victim@example.com", "Hi"),
                            ("a@example.com", "Hi\r\nBcc: victim@example.com")]:
            with self.assertRaises(Exception):
                tools.prepare_email("send_email", {"to": to, "subject": subject, "body": "正文"})
        self.api.users.assert_not_called()

    def test_youtube_metrics_revenue_and_no_data_are_distinct(self):
        reports = self.api.reports.return_value
        execute_result(reports, "query", {"columnHeaders": [{"name": "views"}, {"name": "estimatedRevenue"}],
                                          "rows": [[123, 1.23]]})
        result = tools.get_youtube_analytics("2026-10-01", "2026-10-14", True)
        args = reports.query.call_args.kwargs
        self.assertEqual(args["ids"], "channel==MINE")
        self.assertIn("estimatedRevenue", args["metrics"])
        self.assertEqual(result.structured_content["rows"][0]["estimatedRevenue"], 1.23)
        execute_result(reports, "query", {"columnHeaders": [], "rows": []})
        self.assertIn("不代表", tools.get_youtube_analytics("2026-10-01", "2026-10-14").content[0].text)
        with self.assertRaises(Exception):
            tools.get_youtube_analytics("2026-10-14", "2026-10-01")
        self.assertEqual(reports.query.call_count, 2)

    async def test_catalog_policy_routing_and_hidden_operation_id(self):
        catalog = await personal_server.mcp.list_tools()
        self.assertEqual(tools.GOOGLE_TOOL_NAMES, GOOGLE_TOOLS)
        self.assertTrue(GOOGLE_TOOLS <= {t.name for t in catalog})
        self.assertTrue(GOOGLE_TOOLS <= PERSONAL_TOOLS)
        session = Mock(list_tools=AsyncMock(return_value=Mock(tools=catalog, next_cursor=None)))
        loaded = await load_tools(session)
        selected, owners = merge_tools({"personal": loaded})
        self.assertTrue(GOOGLE_TOOLS <= {t["name"] for t in selected})
        self.assertNotIn("prepare_email", {t["name"] for t in selected})
        for tool in selected:
            if tool["name"] in {"send_email", "reply_email"}:
                self.assertNotIn("confirmation_token", tool["input_schema"]["properties"])
                self.assertIn("confirmed", tool["input_schema"]["properties"])
        event_tool = next(t for t in selected if t["name"] == "create_event")
        self.assertNotIn("operation_id", event_tool["input_schema"]["properties"])
        for name in GOOGLE_TOOLS:
            self.assertEqual(owners[name], "personal")
            self.assertEqual(get_policy(name).mutates, name in GOOGLE_WRITE_TOOLS)
        self.assertFalse(any("delete" in name or "trash" in name or "draft" in name for name in GOOGLE_TOOLS))
        worker = Mock(closed=False, call=AsyncMock(return_value=tools._result("完成", {"id": "x"})))
        executor = make_executor(ha_url="", http_client=None, worker=worker)
        for name in GOOGLE_TOOLS:
            worker.call.reset_mock()
            if name in {"send_email", "reply_email"}:
                with self.assertRaises(Exception):
                    await executor(name, {})
                worker.call.assert_not_called()
            else:
                await executor(name, {})
                worker.call.assert_awaited_once_with(name, {})

    async def test_write_timeout_and_http_errors_have_truthful_gateway_status(self):
        messages = self.setup_send()
        store = OperationStore(":memory:")
        self.addCleanup(store.close)
        worker = Mock(closed=False, call=AsyncMock(side_effect=self.wire_call))
        gateway = ToolGateway(store, make_executor(ha_url="", http_client=None, worker=worker))
        for error, status in [(TimeoutError(), "unknown"), (http_error(503), "unknown"),
                              (http_error(403), "failed"), (RefreshError("invalid_grant"), "failed")]:
            args = {"to": "person@example.com", "subject": "会", "body": "正文"}
            prepared = tools.prepare_email("send_email", args)
            messages.send.return_value.execute.side_effect = error
            op = await gateway.submit("turn", "send_email", args)
            gateway.approve(op.id)
            context_token = EMAIL_CONFIRMATION.set(prepared.structured_content["confirmation_token"])
            try:
                op = await gateway.execute(op.id)
            finally:
                EMAIL_CONFIRMATION.reset(context_token)
            self.assertEqual(op.status, status)
            if status == "unknown":
                self.assertIn("不要重试", format_for_model(op)["text"])
            if isinstance(error, RefreshError):
                # 授权失败一定没有执行；模型拿到的文字要告诉用户怎么重新授权。
                self.assertIn("python -m servers.google_auth", format_for_model(op)["text"])
            await gateway.execute(op.id)
        self.assertEqual(messages.send.return_value.execute.call_count, 4)
        self.assertNotIn("sensitive", str(op.result))

    async def test_bad_inputs_are_mcp_errors_and_no_write_is_sent(self):
        for name, arguments in [("create_document", {"title": ""}),
                                ("send_email", {"to": "x", "subject": "s", "body": "b"}),
                                ("list_events", {"local_start": "bad", "local_end": "bad"})]:
            # 直接调用 server helper 尚未经过 JSON-RPC handler；这里应抛 ToolError。
            with self.assertRaises(ToolError):
                await self.call(name, arguments)
        self.api.documents.assert_not_called()
        self.api.users.assert_not_called()
        self.api.events.assert_not_called()

    async def test_stdio_client_discovers_new_tools_and_receives_validation_error(self):
        # 真正经过 SDK 的 stdio/握手/JSON-RPC，仍只调用校验失败的工具，不访问 Google。
        server = StdioServerParameters(command=sys.executable, args=["-m", "servers.personal_server"],
                                       cwd=str(Path(__file__).resolve().parents[1]))
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                catalog = await session.list_tools()
                self.assertTrue(GOOGLE_TOOLS <= {tool.name for tool in catalog.tools})
                result = await session.call_tool("create_document", arguments={"title": ""})
                self.assertTrue(result.is_error)
                self.assertIn("title", result.content[0].text)

    def test_missing_created_id_is_unknown_and_long_text_is_bounded(self):
        execute_result(self.api.documents.return_value, "create", {})
        self.assertEqual(tools.create_document("文档").structured_content["status"], "unknown")
        result = tools._text_result("x" * (tools.MAX_TEXT + 100), {})
        self.assertTrue(result.structured_content["truncated"])
        self.assertEqual(len(result.structured_content["text"]), tools.MAX_TEXT)

    def test_actual_google_sdk_accepts_every_tool_request_without_network(self):
        # Mock service 能容忍拼错的方法/参数；再用安装版本的离线 discovery 检查一次。
        apis = {api: build(api, version, credentials=AnonymousCredentials(),
                          cache_discovery=False, static_discovery=True)
                for api, version in [("docs", "v1"), ("drive", "v3"), ("calendar", "v3"),
                                     ("gmail", "v1"), ("youtubeAnalytics", "v2")]}
        self.addCleanup(lambda: [api.close() for api in apis.values()])
        self.patch.stop()
        requests = []
        def respond(request, *, num_retries):
            self.assertEqual(num_retries, 0)
            requests.append(request)
            if "alt=media" in request.uri:
                return b"text"
            return {"id": "id", "documentId": "doc", "emailAddress": "owner@example.com",
                    "mimeType": "text/plain", "threadId": "thread", "payload": {"headers": [
                        {"name": "From", "value": "person@example.com"},
                        {"name": "Subject", "value": "subject"},
                        {"name": "Message-ID", "value": "<old@example.com>"}]}}
        start, end = "2026-10-16T08:00:00", "2026-10-16T09:00:00"
        calls = [
            (tools.read_document, ("doc",)), (tools.create_document, ("title",)),
            (tools.append_document, ("doc", "text")), (tools.search_drive, ("file",)),
            (tools.read_drive_file, ("file",)), (tools.create_drive_folder, ("folder",)),
            (tools.list_calendars, ()), (tools.list_events, (start, end)),
            (tools.create_event, ("event", start, end)), (tools.search_email, ("is:unread",)),
            (tools.read_email, ("mail",)), (tools.send_email, ("person@example.com", "s", "body")),
            (tools.reply_email, ("mail", "body")),
            (tools.get_youtube_analytics, ("2026-10-01", "2026-10-14", True)),
        ]
        with patch.object(tools, "_service", side_effect=lambda api, version: apis[api]), \
                patch.object(HttpRequest, "execute", respond):
            for function, arguments in calls:
                with self.subTest(tool=function.__name__):
                    if function.__name__ in {"send_email", "reply_email"}:
                        names = ("to", "subject", "body") if function.__name__ == "send_email" else ("message_id", "body")
                        draft = tools.prepare_email(function.__name__, dict(zip(names, arguments)))
                        result = function(*arguments, confirmed=True,
                                          confirmation_token=draft.structured_content["confirmation_token"])
                    else:
                        result = function(*arguments)
                    self.assertFalse(result.is_error)
        self.assertGreaterEqual(len(requests), len(calls))

    def test_service_reloads_credentials_each_call_and_rebuilds_only_when_token_changes(self):
        self.patch.stop()
        tools._SERVICES.clear()
        self.addCleanup(tools._SERVICES.clear)
        workspace, youtube = Mock(token="w1"), Mock(token="y1")
        with patch.object(tools, "get_credentials", return_value=workspace) as get_workspace, \
                patch.object(tools, "get_youtube_credentials", return_value=youtube) as get_youtube, \
                patch.object(tools, "AuthorizedHttp") as transport, patch.object(tools, "build"):
            tools._service("docs", "v1")
            tools._service("youtubeAnalytics", "v2")
            tools._service("docs", "v1")
            self.assertEqual([call.args[0] for call in transport.call_args_list], [workspace, youtube])
            # 刷新或重新授权后令牌变了：用新凭据重建，不需要重启 Alpha。
            renewed = Mock(token="w2")
            get_workspace.return_value = renewed
            tools._service("docs", "v1")
        self.assertEqual(get_workspace.call_count, 3)  # 每次都从磁盘读取，不缓存凭据
        self.assertEqual(get_youtube.call_count, 1)
        self.assertEqual([call.args[0] for call in transport.call_args_list], [workspace, youtube, renewed])

    def test_oauth_allows_gmail_read_send_only_and_required_youtube_read(self):
        gmail = {scope for scope in SCOPES if "gmail" in scope or "mail.google.com" in scope}
        self.assertEqual(gmail, {"https://www.googleapis.com/auth/gmail.readonly",
                                 "https://www.googleapis.com/auth/gmail.send"})
        self.assertIn("https://www.googleapis.com/auth/youtube.readonly", YOUTUBE_SCOPES)
        self.assertFalse(set(SCOPES) & set(YOUTUBE_SCOPES))
