"""专用流式 TTS；与 Omni 收音连接隔离，取消后丢弃连接避免晚到音频串台。"""
import asyncio
import base64
import json
import logging
from urllib.parse import urlencode
from uuid import uuid4
from core.costs import CumulativeUsage, record_cost, log_summary

from websockets.asyncio.client import connect

from voice.speech_markup import plain_speech

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
        self.usage_meter = CumulativeUsage()

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
        self.usage_meter.reset()
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

    async def stream(self, text, *, turn_id=None, style=None):
        """按原文合成 PCM16 24kHz。失败不自动重播；取消后关闭连接。

        turn_id 是大脑那一轮的 ID：语音进程没有大脑的上下文变量，所以显式传入，费用才能归到那一轮。
        朗读标记（声音事件、语气，见 voice/speech_markup.py）云端不支持：去掉事件标记，忽略语气。
        """
        text = plain_speech(text)
        completed = False
        cost_id = None
        try:
            await self.ready()
            cost_id = record_cost('TTS', 'qwen3-tts-flash-realtime',
                region=self.region, reason='synthesis_started_no_final_usage', turn_id=turn_id)
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
                        usage = self.usage_meter.delta(response.get('usage') or {})
                        logger.info('[TTS用量] %s', json.dumps(usage, ensure_ascii=False))
                        record_cost('TTS', 'qwen3-tts-flash-realtime', usage,
                                    event_id=cost_id, region=self.region, turn_id=turn_id)
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


class QwenAudioTTS:
    """Qwen-Audio-TTS（qwen-audio-3.1-tts-flash/plus）：接口和 QwenTTS 一样，协议不同。

    走 /api-ws/v1/inference 的 run-task → continue-task → finish-task，音频是二进制帧。
    一句一个任务，任务结束连接保留给下一句；usage 每个任务单独给出，不是累计值。
    """

    def __init__(self, key, workspace, region='singapore', voice='longanlingxin_v3.1',
                 model='qwen-audio-3.1-tts-flash'):
        regions = {'singapore': 'ap-southeast-1', 'beijing': 'cn-beijing'}
        if region not in regions or not workspace or not all(c.isalnum() or c == '-' for c in workspace):
            raise ValueError('TTS 需要有效的 QWEN_REGION 和 QWEN_WORKSPACE_ID')
        self.url = f'wss://{workspace}.{regions[region]}.maas.aliyuncs.com/api-ws/v1/inference'
        self.key, self.voice, self.model, self.region, self.ws = key, voice, model, region, None

    async def ready(self):
        if self.ws is None:
            # 服务端不及时回关闭帧，默认关闭要等 10 秒；连接只用来合成，不用等它确认。
            self.ws = await connect(self.url, additional_headers={'Authorization': f'Bearer {self.key}'},
                                    open_timeout=15, close_timeout=1, max_size=16 * 1024 * 1024)

    async def send(self, action, task_id, payload):
        await self.ws.send(json.dumps({'header': {'action': action, 'task_id': task_id, 'streaming': 'duplex'},
                                       'payload': payload}))

    async def stream(self, text, *, turn_id=None, style=None):
        """按原文合成 PCM16 24kHz。失败不自动重播；取消或出错时关闭连接，免得旧任务的音频串到下一句。
        朗读标记同 QwenTTS：去掉事件标记，忽略语气。"""
        text = plain_speech(text)
        completed = False
        cost_id = None
        try:
            await self.ready()
            cost_id = record_cost('TTS', self.model, region=self.region,
                                  reason='synthesis_started_no_final_usage', turn_id=turn_id)
            task_id = uuid4().hex
            await self.send('run-task', task_id, {
                'task_group': 'audio', 'task': 'tts', 'function': 'SpeechSynthesizer', 'model': self.model,
                'parameters': {'text_type': 'PlainText', 'voice': self.voice, 'format': 'pcm', 'sample_rate': 24000},
                'input': {}})
            async with asyncio.timeout(90):
                async for message in self.ws:
                    if isinstance(message, bytes):
                        yield message
                        continue
                    event = json.loads(message)
                    header = event.get('header', {})
                    if header.get('task_id') not in {None, task_id}:
                        continue
                    kind = header.get('event')
                    if kind == 'task-started':
                        await self.send('continue-task', task_id, {'input': {'text': text}})
                        await self.send('finish-task', task_id, {'input': {}})
                    elif kind == 'task-finished':
                        usage = event.get('payload', {}).get('usage') or {}
                        logger.info('[TTS用量] %s', json.dumps(usage, ensure_ascii=False))
                        record_cost('TTS', self.model, usage, event_id=cost_id, region=self.region, turn_id=turn_id)
                        log_summary()
                        completed = True
                        return
                    elif kind == 'task-failed':
                        raise RuntimeError(f"TTS {header.get('error_code', '')}: {header.get('error_message', '服务错误')}")
                raise RuntimeError('TTS 连接意外关闭')
        finally:
            if not completed:
                await self.close()

    async def close(self):
        ws, self.ws = self.ws, None
        if ws is not None:
            await ws.close()


def make_tts(key, workspace, region, voice, model='qwen3-tts-flash-realtime'):
    """按 QWEN_TTS_MODEL 选客户端；两种模型的音色不通用，配错时服务端会报 InvalidParameter。"""
    if model == 'qwen3-tts-flash-realtime':
        return QwenTTS(key, workspace, region, voice)
    return QwenAudioTTS(key, workspace, region, voice, model)
