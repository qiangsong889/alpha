import asyncio
import unittest

from collection import RequestCollection, unfinished


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    def test_only_unfinished_tail_extends_window(self):
        for text in ["关灯，然后……", "再帮我", "还有。", "把牛奶", "等等"]:
            self.assertTrue(unfinished(text), text)
        for text in ["再帮我关厨房灯", "关灯，然后把牛奶加到购物清单", "把灯关掉"]:
            self.assertFalse(unfinished(text), text)

    async def test_incomplete_prompts_once_and_never_submits_until_continued(self):
        collection = RequestCollection(0, continuation_s=0.03)
        collection.add("关灯，然后……")
        waiting = asyncio.create_task(collection.take())
        await asyncio.sleep(0.01)
        self.assertFalse(waiting.done())
        self.assertEqual(await waiting, "")
        self.assertIn("接下来", collection.prompt)
        collection.prompt = None  # 语音层取走追问。
        self.assertEqual(await collection.take(), "")
        self.assertIsNone(collection.prompt)
        self.assertEqual(len(collection.parts), 1)
        collection.add("把牛奶加到购物清单")
        self.assertEqual(await collection.take(), "关灯，然后……；把牛奶加到购物清单")
        self.assertEqual(collection.parts, [])

    async def test_execute_cannot_submit_known_incomplete_request(self):
        collection = RequestCollection(0, continuation_s=0)
        collection.add("把牛奶")
        collection.add("执行")
        self.assertEqual(await collection.take(), "")
        self.assertEqual(collection.parts, ["把牛奶"])

    async def test_done_command_finishes_complete_request_immediately(self):
        collection = RequestCollection(30)
        collection.add("关灯")
        collection.add("就这些")
        self.assertEqual(await asyncio.wait_for(collection.take(), 0.1), "关灯")

    def test_invalid_continuation_window(self):
        for value in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                RequestCollection(continuation_s=value)

    async def test_continuations_are_submitted_together(self):
        collection = RequestCollection(0.03)
        collection.add("关厨房灯")
        task = asyncio.create_task(collection.take())
        await asyncio.sleep(0.01)
        collection.speech_started()
        await asyncio.sleep(0.04)
        self.assertFalse(task.done())
        collection.speech_stopped()
        await asyncio.sleep(0.04)
        self.assertFalse(task.done())  # 尚未拿到第二段转写，不能先发第一段。
        collection.transcript_resolved()
        collection.add("还有，把牛奶加入购物清单")
        self.assertEqual(await task, "关厨房灯；还有，把牛奶加入购物清单")

    async def test_cancelling_wait_preserves_unsubmitted_text(self):
        collection = RequestCollection(0.03)
        collection.add("关厨房灯")
        task = asyncio.create_task(collection.take())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        collection.add("加牛奶")
        self.assertEqual(await collection.take(), "关厨房灯；加牛奶")

    async def test_execute_submits_without_waiting(self):
        collection = RequestCollection(30)
        collection.add("关厨房灯")
        collection.add("执行")
        self.assertEqual(await asyncio.wait_for(collection.take(), 0.1), "关厨房灯")

    async def test_trailing_execute_and_normal_word_are_distinguished(self):
        collection = RequestCollection(30)
        collection.add("关厨房灯，执行。")
        self.assertEqual(await asyncio.wait_for(collection.take(), 0.1), "关厨房灯")
        collection.add("提醒我执行")
        self.assertFalse(collection._immediate)

    async def test_ignored_noise_does_not_add_text(self):
        collection = RequestCollection(0.01)
        collection.add("关厨房灯")
        collection.speech_started()
        collection.speech_stopped()
        collection.transcript_resolved()
        self.assertEqual(await collection.take(), "关厨房灯")

    async def test_close_discards_unsubmitted_request(self):
        collection = RequestCollection(30)
        collection.add("关厨房灯")
        task = asyncio.create_task(collection.take())
        await asyncio.sleep(0)
        collection.close()
        self.assertEqual(await task, "")

    def test_invalid_window_is_rejected(self):
        for value in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                RequestCollection(value)
