"""一个 HTTP 请求包含两个分类问题；不访问真实 Jev。"""
import asyncio
import json
from unittest import IsolatedAsyncioTestCase
import httpx2
from voice_pipeline.search_router import JevSearchRouter


def answer(choice, confidence=.99):
    return {'type': 'choice', 'choice': choice, 'confidence': confidence}


class CombinedGateTests(IsolatedAsyncioTestCase):
    async def test_addressed_choice_is_not_erased_by_low_confidence(self):
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request:
            httpx2.Response(200, json={'answers': {'interruption': answer('addressed', .35),
                'turn_complete': answer('complete')}}))) as client:
            router = JevSearchRouter(client, api_key='fake')
            self.assertEqual(await router.interruption_and_turn('把提醒改到九点', combined_text='把提醒改到九点',
                assistant='提醒已设', question_open=False, history=[]), (True, True))

    async def test_two_questions_one_request_and_independent_thresholds(self):
        requests = []
        async def handle(request):
            requests.append(json.loads(request.content))
            return httpx2.Response(200, json={'answers': {
                'interruption': answer('other', .89), 'turn_complete': answer('complete')}})
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            router = JevSearchRouter(client, api_key='fake')
            decision = await router.interruption_and_turn('小王吃什么', combined_text='加牛奶；小王吃什么',
                assistant='清单里有牛奶', question_open=False, history=[])
        self.assertEqual(decision, (False, True))
        self.assertEqual(len(requests), 1)
        self.assertEqual(set(requests[0]['questions']), {'interruption', 'turn_complete'})
        self.assertEqual(requests[0]['state']['request'], '小王吃什么')
        self.assertNotIn('fake', json.dumps(requests[0]))

    async def test_uncertain_low_confidence_and_malformed_do_not_become_other(self):
        for interruption in (answer('uncertain'), answer('other', .84), answer('other', True), {},
                             answer('other', float('nan'))):
            with self.subTest(interruption=interruption):
                async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request:
                    httpx2.Response(200, text=json.dumps({'answers': {'interruption': interruption,
                                                         'turn_complete': answer('incomplete')}})))) as client:
                    router = JevSearchRouter(client, api_key='fake')
                    result = await router.interruption_and_turn('改一下', combined_text='改一下',
                        assistant='完成了', question_open=False, history=[])
                self.assertEqual(result, (None, False))

    async def test_http_error_and_missing_key_fall_back(self):
        for key in ('fake', ''):
            calls = []
            def handle(request):
                calls.append(request)
                return httpx2.Response(503)
            async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
                router = JevSearchRouter(client, api_key=key)
                self.assertEqual(await router.interruption_and_turn('改一下', combined_text='改一下',
                    assistant='完成了', question_open=False, history=[]), (None, None))
                self.assertEqual(len(calls), int(bool(key)))

    async def test_timeout_and_cancellation(self):
        async def handle(request):
            await asyncio.Future()
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
            router = JevSearchRouter(client, api_key='fake', timeout_s=.01)
            self.assertEqual(await router.interruption_and_turn('等等', combined_text='等等',
                assistant='汇报', question_open=False, history=[]), (None, None))
            router.timeout_s = 10
            task = asyncio.create_task(router.interruption_and_turn('等等', combined_text='等等',
                assistant='汇报', question_open=False, history=[]))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
