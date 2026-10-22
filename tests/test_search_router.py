"""TypeSafe 官方协议和失败降级；不使用真实密钥或网络。"""

import asyncio
import json
import unittest

import httpx2 as httpx

from voice.search_router import JevSearchRouter, JEV_URL


def answer(choice="web", confidence=0.95):
    return {"model": "jev-latest", "answers": {"web_route": {
        "type": "choice", "choice": choice, "confidence": confidence,
        "probabilities": {"web": 0.95, "direct": 0.04, "uncertain": 0.01},
    }}}


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_protocol_context_and_high_confidence_decisions(self):
        requests = []
        def handle(request):
            requests.append(request)
            return httpx.Response(200, json=answer())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            router = JevSearchRouter(client, api_key="fake-key")
            history = [{"role": "user", "content": "餐馆" + "x" * 1000} for _ in range(10)]
            self.assertIs(await router.classify("台湾的呢？", history), True)
        request = requests[0]
        self.assertEqual(str(request.url), JEV_URL)
        self.assertEqual(request.headers["Authorization"], "Bearer fake-key")
        payload = json.loads(request.content)
        self.assertEqual(payload["state"]["request"], "台湾的呢？")
        self.assertEqual(len(payload["state"]["recent_dialogue"]), 6)
        self.assertTrue(all(len(item["content"]) <= 800 for item in payload["state"]["recent_dialogue"]))
        question = payload["questions"]["web_route"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(set(question["criteria"]), {"web", "direct", "uncertain"})
        self.assertIn("家附近今明天气已有专用工具", question["instructions"])

    async def test_turn_complete_protocol_and_fallback_decisions(self):
        for choice, confidence, expected in [("complete", .99, True), ("incomplete", .99, False),
                                             ("uncertain", .99, None), ("complete", .2, None)]:
            requests = []
            def handle(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json={"answers": {"turn_complete": {
                    "type": "choice", "choice": choice, "confidence": confidence}}})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                result = await JevSearchRouter(client, api_key="fake").turn_complete("把牛奶", [])
            self.assertIs(result, expected)
            self.assertEqual(requests[0]["state"]["request"], "把牛奶")
            self.assertEqual(set(requests[0]["questions"]), {"turn_complete"})

    async def test_interruption_choice_context_and_conservative_threshold(self):
        for choice, confidence, expected in [("addressed", .98, True), ("other", .99, False),
                                             ("uncertain", .99, None), ("addressed", .85, None)]:
            requests = []
            def handle(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json={"answers": {"interruption": {
                    "type": "choice", "choice": choice, "confidence": confidence}}})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                result = await JevSearchRouter(client, api_key="fake").interruption(
                    "对", assistant="要牛奶吗？", question_open=True)
            self.assertIs(result, expected)
            self.assertEqual(requests[0]["state"], {
                "request": "对", "assistant_reply": "要牛奶吗？", "answer_window_open": True})
            self.assertEqual(set(requests[0]["questions"]), {"interruption"})

    async def test_interruption_timeout_and_cancellation(self):
        async def handle(request):
            await asyncio.Future()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            router = JevSearchRouter(client, api_key="fake", timeout_s=.01)
            self.assertIsNone(await router.interruption("小王", assistant="汇报", question_open=False))
            task = asyncio.create_task(router.interruption("小王", assistant="汇报", question_open=False))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_vercel_protocol_and_model(self):
        requests = []
        def handle(request):
            requests.append(request)
            return httpx.Response(200, json=answer())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            router = JevSearchRouter(client, api_key="fake-gateway-key", provider="vercel")
            self.assertIs(await router.classify("查餐馆", []), True)
        self.assertEqual(str(requests[0].url), "https://ai-gateway.vercel.sh/typesafe/v1/systemone")
        self.assertEqual(json.loads(requests[0].content)["model"], "typesafe-ai/jev")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer fake-gateway-key")

    async def test_direct_uncertain_low_confidence_and_invalid_results(self):
        for payload, expected in [(answer("direct"), False), (answer("uncertain"), None),
                                  (answer("web", 0.2), None), (answer("invented"), None),
                                  (answer("web", True), None), (answer("web", 2), None),
                                  ({"answers": []}, None), ({}, None)]:
            with self.subTest(payload=payload):
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, json=payload))) as client:
                    self.assertIs(await JevSearchRouter(client, api_key="fake").classify("关灯", []), expected)

    async def test_failure_is_not_retried_and_secrets_are_not_logged(self):
        for status in [401, 429, 500]:
            requests = []
            def handle(request):
                requests.append(request)
                return httpx.Response(status, text="sensitive response")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                with self.assertLogs("alpha.jev", "INFO") as logs:
                    self.assertIsNone(await JevSearchRouter(client, api_key="fake-secret").classify("hi", []))
                self.assertEqual(len(requests), 1)
                self.assertNotIn("fake-secret", " ".join(logs.output))
                self.assertNotIn("sensitive response", " ".join(logs.output))

    async def test_total_timeout_includes_response_body_and_cancels_read(self):
        cancelled = asyncio.Event()
        class SlowBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                try:
                    await asyncio.Future()
                    yield b"{}"
                finally:
                    cancelled.set()

        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=SlowBody()))) as client:
            router = JevSearchRouter(client, api_key="fake", timeout_s=0.01)
            self.assertIsNone(await asyncio.wait_for(router.classify("hi", []), 0.2))
            self.assertTrue(cancelled.is_set())

    async def test_cancellation_propagates_without_fallback(self):
        async def handle(request):
            raise asyncio.CancelledError()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(asyncio.CancelledError):
                await JevSearchRouter(client, api_key="fake").classify("hi", [])

    async def test_missing_key_does_not_send_request(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: self.fail("无密钥不应发请求"))) as client:
            self.assertIsNone(await JevSearchRouter(client, api_key="").classify("hi", []))

    def test_invalid_latency_and_confidence_configuration(self):
        for value in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                JevSearchRouter(None, api_key="", timeout_s=value)
        for value in [-1, 2, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                JevSearchRouter(None, api_key="", min_confidence=value)
