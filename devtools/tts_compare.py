"""TTS 试听对比：同样几句 Alpha 的回答，用现在的 qwen3-tts 和 Qwen-Audio-TTS 各合成一遍，存成 WAV 由人来听。

在 alpha 目录运行：
    .venv/bin/python -m devtools.tts_compare [输出目录]

读取 .env 的 DASHSCOPE_API_KEY、QWEN_WORKSPACE_ID、QWEN_REGION。不播放、不写费用账本（基线那一路除外，
它直接用 voice.qwen_tts.QwenTTS，照常记账）。每个候选每句一个任务，和 Alpha 按句合成一样；
记录从发出文字到收到第一块音频的时间（首包），不含建连。
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4
import wave

from dotenv import load_dotenv
from websockets.asyncio.client import connect

from core.paths import ENV_FILE
from voice.qwen_tts import QwenTTS

SAMPLE_RATE = 24000
SENTENCES = [
    "好的，客厅灯已经关了。",
    "明天多云，十二到十九度，下午有三成概率下雨，出门可以带把伞。",
    "你购物清单上还有三样：鸡蛋、tofu 和 green tea。",
    "这个问题有两种看法。一种认为先把基础打牢更重要；另一种认为边做项目边学更快。我建议先做一个小项目，遇到不会的再补。",
]
# 新加坡可用的 Qwen-Audio 3.1 中文音色（2026-10 官方音色列表）。
VOICES = ["xunanchuan_v3.1", "longanhuan_v3.1", "longanlingxin_v3.1", "longanfengyue_v3.1"]
INSTRUCTION = "用自然、放松的语气说话，像朋友之间聊天，语速适中，不要播音腔。"


def save(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)


async def qwen_audio(url: str, key: str, model: str, voice: str, text: str,
                     instruction: str | None) -> tuple[bytes, float | None, dict]:
    """run-task → continue-task → finish-task；音频是二进制帧。返回 PCM、首包秒数和最后的 usage。"""
    task_id = uuid4().hex
    parameters = {"text_type": "PlainText", "voice": voice, "format": "pcm", "sample_rate": SAMPLE_RATE}
    if instruction:
        parameters["instruction"] = instruction
    header = {"task_id": task_id, "streaming": "duplex"}
    pcm, first, usage = bytearray(), None, {}
    async with connect(url, additional_headers={"Authorization": f"Bearer {key}"},
                       open_timeout=15, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"header": {"action": "run-task", **header}, "payload": {
            "task_group": "audio", "task": "tts", "function": "SpeechSynthesizer", "model": model,
            "parameters": parameters, "input": {}}}))
        sent = None
        async with asyncio.timeout(60):
            async for message in ws:
                if isinstance(message, bytes):
                    if first is None and sent is not None:
                        first = time.perf_counter() - sent
                    pcm += message
                    continue
                event = json.loads(message)
                kind = event["header"].get("event")
                if kind == "task-started":
                    sent = time.perf_counter()
                    await ws.send(json.dumps({"header": {"action": "continue-task", **header},
                                              "payload": {"input": {"text": text}}}))
                    await ws.send(json.dumps({"header": {"action": "finish-task", **header},
                                              "payload": {"input": {}}}))
                elif kind == "result-generated":
                    usage = event.get("payload", {}).get("usage") or usage
                elif kind == "task-finished":
                    usage = event.get("payload", {}).get("usage") or usage
                    break
                elif kind == "task-failed":
                    raise RuntimeError(f"{event['header'].get('error_code')}: {event['header'].get('error_message')}")
    return bytes(pcm), first, usage


async def baseline(key: str, workspace: str, region: str, voice: str, text: str) -> tuple[bytes, float | None, dict]:
    tts = QwenTTS(key, workspace, region, voice)
    try:
        await tts.ready()  # 建连不计入首包
        start, first, pcm = time.perf_counter(), None, bytearray()
        async for chunk in tts.stream(text):
            first = first if first is not None else time.perf_counter() - start
            pcm += chunk
        return bytes(pcm), first, {}
    finally:
        await tts.close()


async def main() -> None:
    load_dotenv(ENV_FILE)
    key, workspace = os.environ["DASHSCOPE_API_KEY"], os.environ["QWEN_WORKSPACE_ID"]
    region = os.getenv("QWEN_REGION", "singapore")
    host = {"singapore": "ap-southeast-1", "beijing": "cn-beijing"}[region]
    url = f"wss://{workspace}.{host}.maas.aliyuncs.com/api-ws/v1/inference"
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/tts_compare")
    out.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[str, object]] = [
        (f"0_现在_qwen3-tts_{os.getenv('QWEN_TTS_VOICE', 'Ethan')}",
         lambda text: baseline(key, workspace, region, os.getenv("QWEN_TTS_VOICE", "Ethan"), text))]
    for model in ("qwen-audio-3.1-tts-flash", "qwen-audio-3.1-tts-plus"):
        tag = model.removeprefix("qwen-audio-3.1-tts-")
        for voice in VOICES:
            candidates.append((f"{tag}_{voice}", lambda text, m=model, v=voice: qwen_audio(url, key, m, v, text, None)))
        # 只给男声和一个女声加指令版，听指令控制有没有用。
        for voice in VOICES[:2]:
            candidates.append((f"{tag}_{voice}_指令",
                               lambda text, m=model, v=voice: qwen_audio(url, key, m, v, text, INSTRUCTION)))

    for name, synth in candidates:
        joined, firsts, error = bytearray(), [], None
        for index, text in enumerate(SENTENCES, 1):
            try:
                pcm, first, usage = await synth(text)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
            save(out / f"{name}_{index}.wav", pcm)
            joined += pcm + bytes(SAMPLE_RATE)  # 句间留 0.5 秒（PCM16 每秒 48000 字节）
            firsts.append(first)
            if index == 1 and usage:
                print(f"    usage 示例：{json.dumps(usage, ensure_ascii=False)}")
        if error:
            print(f"✗ {name}：{error}")
            continue
        save(out / f"{name}_全部.wav", bytes(joined))
        shown = "、".join(f"{f:.2f}s" if f is not None else "-" for f in firsts)
        print(f"✓ {name}：首包 {shown}")
    print(f"输出目录：{out.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
