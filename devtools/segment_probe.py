"""A2：麦克风→Silero→本地段落队列→串行ASR，不调用Jev/大脑/业务工具。

.venv/bin/python -m devtools.segment_probe --checklist
.venv/bin/python -m devtools.segment_probe --wav paused.wav --offline
--offline 仅分段，不连接ASR；--wav 不使用麦克风。模型配置见 devtools/SEGMENT_PROBE.md。
"""

import argparse
import asyncio
from collections import Counter
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from devtools.asr_probe import read_wav, websocket_url
from voice.asr_client import AsrClient
from voice.segmenter import DEFAULT_MODEL, FRAME, RATE, Segmenter, SegmentReady, SileroVAD

CHECKLIST = [('五句', 45, '请依次说5句，每句之间停约1秒：今天天气怎么样；把客厅灯关了；购物清单里有什么；明天八点提醒我开会；你好。只转写，不执行。'),
             ('单字', 10, '请单独说一次“对”。'),
             ('停顿', 20, '请说“把牛奶”，停顿1秒，再说“和鸡蛋加到购物清单”。'),
             ('咳嗽', 10, '请咳嗽一次，观察是否触发分段。'),
             ('安静', 60, '请保持安静坐1分钟，不说话、不播放声音。')]


async def probe(args):
    detector = SileroVAD(args.model)
    segmenter = Segmenter(detector)
    key = os.getenv('DASHSCOPE_API_KEY', '')
    if not args.offline and not key:
        raise ValueError('缺少DASHSCOPE_API_KEY')
    counts = Counter()
    for name, _, _ in CHECKLIST if args.checklist else [('自由测试', 0, '')]:
        counts[name, 'started'] = 0
    segments = {}
    origin = None
    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    overflow = False
    loop = asyncio.get_running_loop()

    def log(text):
        print(text, flush=True)

    def phase_at(seconds):
        if not args.checklist:
            return 'wav' if args.wav else '自由测试'
        end = 0
        for name, duration, _ in CHECKLIST:
            end += duration
            if seconds < end:
                return name
        return '结束'

    async def result(event):
        phase, segment = segments.get(event.seg_id, ('未知', None))
        counts[(phase, event.kind)] += 1
        if event.kind == 'transcript':
            log(f'[文字][{phase}] {event.seg_id}：{event.text}')
        else:
            log(f'[失败][{phase}] {event.seg_id}：{event.text}')
        timing = (f'commit→文字 {event.commit_to_result_s:.3f}s' if event.commit_to_result_s is not None else '')
        if origin is not None and segment is not None and not args.wav:
            speech_end_to_result_s = time.perf_counter() - (origin + segment.speech_end_s)
            timing += f' ｜ 最后人声→结果（估计）{speech_end_to_result_s:.3f}s'
        log(f'[延迟] {event.seg_id} {timing} ｜ 排队 {event.queue_s:.3f}s ｜ 本段处理 {event.elapsed_s:.3f}s')

    client = (None if args.offline else AsrClient(websocket_url(os.getenv('QWEN_WORKSPACE_ID', ''),
              os.getenv('QWEN_REGION', 'singapore')), key, result))

    async def dispatch(events):
        for event in events:
            if isinstance(event, SegmentReady):
                phase, _ = segments[event.seg_id]
                segments[event.seg_id] = phase, event
                counts[(phase, 'ready')] += 1
                log(f'[结束][{phase}] {event.seg_id} [{event.start_s:.3f}, {event.end_s:.3f}] '
                    f'最后人声={event.speech_end_s:.3f}s reason={event.reason} PCM={len(event.pcm)}字节')
                if client is not None:
                    await client.submit(event)
            else:
                phase = phase_at(event.speech_start_s)
                segments[event.seg_id] = phase, None
                counts[(phase, 'started')] += 1
                log(f'[开始][{phase}] {event.seg_id} 语音={event.speech_start_s:.3f}s '
                    f'含缓冲起点={event.start_s:.3f}s 确认={event.confirmed_s:.3f}s continuation={event.continuation}')

    def offer(pcm, captured_at):
        nonlocal overflow
        try:
            audio_queue.put_nowait((pcm, captured_at))
        except asyncio.QueueFull:
            overflow = True

    def callback(indata, frames, timing, status):
        # PortAudio回调只复制和投递；推理、分段、上传都在下面的处理任务中。
        payload = RuntimeError(f'麦克风流异常：{status}') if status else bytes(indata)
        loop.call_soon_threadsafe(offer, payload, time.perf_counter())

    async def process_audio():
        nonlocal origin
        diagnostic_at = time.perf_counter()
        rms_sum = rms_max = probability_max = 0.0
        blocks = 0
        def diagnostic():
            log(f'[音频诊断] 处理块={blocks} RMS均值={rms_sum / blocks:.5f} '
                f'最大={rms_max:.5f} Silero概率最大={probability_max:.3f}')
        while True:
            record = await audio_queue.get()
            if overflow:
                raise RuntimeError('麦克风音频积压超过约8秒，停止探测，避免静默丢音')
            if record is None:
                if args.diagnostics and blocks:
                    diagnostic()
                await dispatch(await asyncio.to_thread(segmenter.flush))
                return
            pcm, captured_at = record
            if isinstance(pcm, Exception):
                raise pcm
            if origin is None:
                origin = captured_at - len(pcm) / (RATE * 2)
            await dispatch(await asyncio.to_thread(segmenter.feed, pcm))
            counts[phase_at(segmenter.position / RATE), 'blocks'] += 1
            if args.diagnostics:
                samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
                rms = float(np.sqrt(np.mean(samples * samples)))
                rms_sum += rms
                rms_max = max(rms_max, rms)
                probability_max = max(probability_max, segmenter.last_probability)
                blocks += 1
                if time.perf_counter() - diagnostic_at >= 10:
                    diagnostic()
                    diagnostic_at = time.perf_counter()
                    rms_sum = rms_max = probability_max = 0.0
                    blocks = 0

    async def duration():
        if args.checklist:
            for name, seconds, instruction in CHECKLIST:
                log(f'[验收阶段][{name}] {seconds}秒：{instruction}')
                await asyncio.sleep(seconds)
        else:
            log(f'[录音] {args.duration}秒，Ctrl+C退出。')
            await asyncio.sleep(args.duration)

    try:
        async with asyncio.TaskGroup() as tasks:
            worker = tasks.create_task(client.run()) if client is not None else None
            processor = tasks.create_task(process_audio())
            if args.wav:
                pcm = read_wav(args.wav)
                for offset in range(0, len(pcm), FRAME * 2):
                    await audio_queue.put((pcm[offset:offset + FRAME * 2], time.perf_counter()))
            else:
                device = args.device or os.getenv('REALTIME_INPUT_DEVICE') or None
                if isinstance(device, str) and device.isdecimal():
                    device = int(device)
                info = sd.query_devices(device, 'input')
                log(f'[麦克风] {info["name"]} ｜ 16kHz PCM16 ｜ Silero ONNX ｜ ASR={client is not None}')
                with sd.RawInputStream(samplerate=RATE, channels=1, dtype='int16', device=device,
                                       blocksize=FRAME, callback=callback):
                    await duration()
                await asyncio.sleep(0)  # 收完停止流之前已投递的回调。
            await audio_queue.put(None)
            await processor
            if client is not None:
                await client.queue.join()
            if worker is not None:
                worker.cancel()
    finally:
        for phase in sorted({p for p, _ in counts}):
            log(f'[汇总][{phase}] 开始={counts[phase, "started"]} 分段={counts[phase, "ready"]} '
                f'文字={counts[phase, "transcript"]} 失败={counts[phase, "failed"]} 音频块={counts[phase, "blocks"]}')


async def main(argv=None):
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(os.getenv('SILERO_MODEL_PATH') or DEFAULT_MODEL))
    parser.add_argument('--wav', type=Path, help='从WAV读取，不打开麦克风')
    parser.add_argument('--offline', action='store_true', help='只运行本地分段，不调用ASR')
    parser.add_argument('--duration', type=float, default=160)
    parser.add_argument('--device', help='麦克风设备名称或编号；默认REALTIME_INPUT_DEVICE')
    parser.add_argument('--checklist', action='store_true', help='按验收清单依次提示，共145秒')
    parser.add_argument('--diagnostics', action='store_true', help='每10秒打印音量及Silero概率，只用于诊断，不作为门槛')
    args = parser.parse_args(argv)
    if args.duration <= 0 or not math.isfinite(args.duration) or (args.wav and args.checklist):
        parser.error('duration必须是正有限数，wav与checklist不能同时设置')
    await probe(args)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\n已退出', flush=True)
    except Exception as exc:
        def describe(error):
            if isinstance(error, BaseExceptionGroup):
                return '；'.join(describe(e) for e in error.exceptions)
            return f'{type(error).__name__}: {error}'
        message = describe(exc)
        key = os.getenv('DASHSCOPE_API_KEY', '')
        print('[探测失败] ' + (message.replace(key, '[REDACTED]') if key else message), flush=True)
        raise SystemExit(1)
