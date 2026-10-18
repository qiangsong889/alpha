"""专用流式 TTS；与 Omni 收音连接隔离，取消后丢弃连接避免晚到音频串台。"""
import asyncio
import base64
import json
import logging
from urllib.parse import urlencode
from uuid import uuid4
from core.costs import record_cost, log_summary

from websockets.asyncio.client import connect

logger = logging.getLogger('alpha.qwen_alpha')


class QwenTTS:
    def __init__(self, key, workspace, region='singapore', voice='Ethan'):
        regions = {'singapore': 'ap-southeast-1', 'beijing': 'cn-beijing'}
        if region not in regions or not workspace or not all(c.isalnum() or c == '-' for c in workspace):
            raise ValueError('TTS 需要有效的 QWEN_REGION 和 QWEN_WORKSPACE_ID')
        self.url = (f'wss://{workspace}.{regions[region]}.maas.aliyuncs.com/api-ws/v1/realtime?'
                    + urlencode({'model': 'qwen3-tts-flash-realtime'}))
        self.key, self.voice, self.ws = key, voice, None
        self.region = region

    async def send(self, kind, **fields):
        await self.ws.send(json.dumps({'event_id': uuid4().hex, 'type': kind, **fields}))

    async def receive(self):
        event = json.loads(await self.ws.recv())
        if event.get('type') == 'error':
            error = event.get('error', {})
            raise RuntimeError(f"TTS {error.get('code', '')}: {error.get('message', '服务错误')}")
        return event

    async def ready(self):
        if self.ws is not None:
            return
        self.ws = await connect(self.url, additional_headers={'Authorization': f'Bearer {self.key}'},
                                open_timeout=15, max_size=16 * 1024 * 1024)
        async with asyncio.timeout(15):
            while (await self.receive())['type'] != 'session.created':
                pass
            await self.send('session.update', session={'voice': self.voice, 'mode': 'commit',
                'response_format': 'pcm', 'sample_rate': 24000, 'language_type': 'Chinese'})
            while True:
                event = await self.receive()
                if event['type'] == 'session.updated':
                    if event['session'].get('voice') != self.voice:
                        raise RuntimeError('TTS 音色配置未生效')
                    break

    async def stream(self, text):
        """按原文合成 PCM16 24kHz。失败不自动重播；取消后关闭连接。"""
        completed = False
        cost_id = None
        try:
            await self.ready()
            cost_id = record_cost('TTS', 'qwen3-tts-flash-realtime',
                region=self.region, reason='synthesis_started_no_final_usage')
            await self.send('input_text_buffer.append', text=text)
            await self.send('input_text_buffer.commit')
            async with asyncio.timeout(90):
                while True:
                    event = await self.receive()
                    if event['type'] == 'response.audio.delta':
                        yield base64.b64decode(event['delta'])
                    elif event['type'] == 'response.done':
                        response = event.get('response', {})
                        if response.get('status') not in {None, 'completed'}:
                            raise RuntimeError('TTS 合成没有完成')
                        logger.info('[TTS用量] %s', json.dumps(response.get('usage', {}), ensure_ascii=False))
                        record_cost('TTS', 'qwen3-tts-flash-realtime', response.get('usage'),
                                    event_id=cost_id, region=self.region)
                        log_summary()
                        completed = True
                        return
        finally:
            if not completed:
                await self.close()

    async def close(self):
        ws, self.ws = self.ws, None
        if ws is not None:
            await ws.close()
