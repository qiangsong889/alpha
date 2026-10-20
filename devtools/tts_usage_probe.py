"""同一连接连续合成两次“你好”，观察原始服务端 usage；不播放、不写费用账本。"""
import asyncio
import json
import os

from dotenv import load_dotenv
from core.paths import ENV_FILE
from voice.qwen_tts import QwenTTS


async def main():
    load_dotenv(ENV_FILE)
    service = QwenTTS(os.environ['DASHSCOPE_API_KEY'], os.environ['QWEN_WORKSPACE_ID'],
                      os.getenv('QWEN_REGION', 'singapore'), os.getenv('QWEN_TTS_VOICE', 'Ethan'))
    receive = service.receive
    usages = []

    async def inspect():
        event = await receive()
        if event['type'] == 'response.done':
            usage = event.get('response', {}).get('usage', {})
            usages.append(usage)
            print(f'第{len(usages)}次 原始usage=' + json.dumps(usage, ensure_ascii=False), flush=True)
        return event

    service.receive = inspect
    try:
        for _ in range(2):
            async for _ in service.stream('你好'):
                pass
        print('characters=' + json.dumps([u.get('characters') for u in usages]), flush=True)
    finally:
        await service.close()


if __name__ == '__main__':
    asyncio.run(main())
