"""流式 HTTP 协议以及断开后仍完成原请求的行为；不调用真实模型或工具。"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from brain import server


class BrainStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_announcement_marker_is_forwarded(self):
        seen = []
        async def ask(text, *, on_text, on_signal, search_announced=False):
            seen.append(search_announced)
            await on_text("查到了。")
            return "查到了。"

        with patch.object(server, "brain", SimpleNamespace(ask=ask, last_timings=[])):
            for marker in [False, True]:
                response = await server.ask_stream(server.AskRequest(text="搜索新闻", search_announced=marker))
                events = [json.loads(line) async for line in response.body_iterator]
                self.assertEqual(events[0], {"type": "delta", "text": "查到了。"})
        self.assertEqual(seen, [False, True])
    async def test_ha_signal_is_delivered_without_tts_text(self):
        async def ask(text, *, on_text, on_signal, search_announced=False):
            await on_signal("ha_done")
            return "开灯成功"

        with patch.object(server, "brain", SimpleNamespace(ask=ask, last_timings=[])):
            response = await server.ask_stream(server.AskRequest(text="开灯"))
            events = [json.loads(line) async for line in response.body_iterator]
        self.assertEqual(events, [{"type": "signal", "name": "ha_done"}, {"type": "done", "timings": []}])

    async def test_delta_arrives_before_done_and_disconnect_does_not_cancel_brain(self):
        gate, finished = asyncio.Event(), asyncio.Event()

        async def ask(text, *, on_text, on_signal, search_announced=False):
            await on_text("第一句。")
            await gate.wait()
            finished.set()
            return "第一句。"

        with patch.object(server, "brain", SimpleNamespace(ask=ask, last_timings=[])):
            response = await server.ask_stream(server.AskRequest(text="你好"))
            try:
                first = json.loads(await anext(response.body_iterator))
                self.assertEqual(first, {"type": "delta", "text": "第一句。"})
                self.assertFalse(finished.is_set())
                await response.body_iterator.aclose()
            finally:
                gate.set()
                await asyncio.gather(*server.stream_tasks)
            self.assertTrue(finished.is_set())

    async def test_done_has_timings(self):
        async def ask(text, *, on_text, on_signal, search_announced=False):
            await on_text("好了。")
            return "好了。"

        with patch.object(server, "brain", SimpleNamespace(ask=ask, last_timings=[("工具", 0.1)])):
            response = await server.ask_stream(server.AskRequest(text="你好"))
            events = [json.loads(line) async for line in response.body_iterator]
        self.assertEqual(events[-1], {"type": "done", "timings": [["工具", 0.1]]})

    async def test_processing_error_is_terminal_and_does_not_retry(self):
        calls = []

        async def ask(text, *, on_text, on_signal, search_announced=False):
            calls.append(text)
            raise TimeoutError()

        with patch.object(server, "brain", SimpleNamespace(ask=ask)), self.assertLogs("alpha.brain_server", "ERROR"):
            response = await server.ask_stream(server.AskRequest(text="加牛奶"))
            events = [json.loads(line) async for line in response.body_iterator]
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(calls, ["加牛奶"])
