"""回车开始、VAD 自动停止录音，使用本地 Whisper 模型连续测试中文转写。

运行：alpha/.venv/bin/python alpha/listen.py
首次加载 small 模型时会自动下载模型文件。
"""

import time
import os
from dotenv import load_dotenv

from core.paths import ENV_FILE
import queue
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import webrtcvad  # type: ignore[import-untyped]
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd  # type: ignore[import-untyped]
from faster_whisper import WhisperModel  # type: ignore[import-untyped]
from opencc import OpenCC

_t2s = OpenCC("t2s")  
SAMPLE_RATE = 16000
_TRANSCRIBE_LOCK = Lock()
load_dotenv(ENV_FILE)


def looks_complete(text: str) -> bool:
    """保守识别短句和未结束的连接词；不是语义理解。"""
    text = re.sub(r"[\s，。！？、；：,.!?;:…]+$", "", text.strip())
    if len(re.sub(r"\s+", "", text)) <= 2:
        return False
    if text.endswith(("和", "还有", "把", "给", "然后", "或者", "跟")):
        return False
    # “把牛奶”缺少动作，按用户的例子继续等待。
    if text.startswith("把") and not any(
        verb in text[1:] for verb in ("加", "放", "开", "关", "调", "设", "删", "标", "移", "改", "完成")
    ):
        return False
    return True


def record() -> NDArray[np.float32]:
    """按回车开始，VAD 检测说完后停止；输入退出命令时抛出 EOFError。"""
    try:
        silence_ms = int(os.environ.get("SILENCE_MS", "1500"))
    except ValueError as exc:
        raise ValueError("SILENCE_MS 必须是正整数毫秒数") from exc
    if silence_ms < 400:
        raise ValueError("SILENCE_MS 必须至少为 400 毫秒")
    while True:
        command = input("按回车开始录音（q 退出）：").strip()
        if command.lower() in {"q", "exit", "quit", "退出"}:
            raise EOFError
        if not command:
            break
        print("请按回车开始，或输入 q 退出。")
    frame_samples = SAMPLE_RATE * 30 // 1000  # 480 个采样点
    frames_queue: queue.Queue[tuple[NDArray[np.float32], float]] = queue.Queue()
    warnings: set[str] = set()
    chunks: list[NDArray[np.float32]] = []
    vad = webrtcvad.Vad(2)

    def callback(indata, frames, time_info, status):
        if status:
            warnings.add(str(status))
        # 保持音频回调简短；主线程逐帧执行 VAD。
        frames_queue.put((indata[:, 0].copy(), time.perf_counter()))

    speech_started = False
    silent_samples = 0
    speaking_samples = 0
    pending = np.empty(0, dtype=np.float32)
    last_speech_at: float | None = None
    stop_reason = "无人声超时"
    print(f"录音中，静音 400 ms 后判断句尾，最长等待 {silence_ms} ms。")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vad-preview")
    preview = None
    preview_epoch = -1
    checked_epoch = -1
    speech_epoch = 0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            callback=callback,
        ):
            waiting_started = time.perf_counter()
            finished = False
            while not finished:
                if not speech_started and time.perf_counter() - waiting_started >= 5:
                    break
                try:
                    block, captured_at = frames_queue.get(timeout=0.1)
                except queue.Empty:
                    if speech_started:
                        raise RuntimeError("麦克风音频流中断")
                    continue
                # 即使设备返回不同大小的块，也只将完整 30 ms 帧送入 VAD。
                pending = np.concatenate((pending, block))
                while pending.size >= frame_samples:
                    frame = pending[:frame_samples]
                    pending = pending[frame_samples:]
                    frame_bytes = (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()
                    is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
                    chunks.append(frame.copy())
                    if is_speech:
                        speech_started = True
                        speech_epoch += 1
                        silent_samples = 0
                        last_speech_at = captured_at - pending.size / SAMPLE_RATE
                    elif speech_started:
                        silent_samples += frame_samples
                    if speech_started:
                        speaking_samples += frame_samples
                        if preview is not None and preview.done():
                            try:
                                preview_text = preview.result()
                                if preview_epoch == speech_epoch:
                                    checked_epoch = speech_epoch
                                    complete = looks_complete(preview_text)
                                    print(f"[VAD 句尾] {preview_text or '（空）'} → {'像说完了' if complete else '继续等待'}")
                                    if complete and silent_samples * 1000 >= SAMPLE_RATE * 400:
                                        stop_reason = "句尾判断完成"
                                        finished = True
                                        break
                            except Exception as exc:
                                checked_epoch = preview_epoch
                                print(f"[VAD 句尾] 临时转写失败，继续等待：{exc}")
                            preview = None
                        if (silent_samples * 1000 >= SAMPLE_RATE * 400
                                and preview is None and checked_epoch != speech_epoch):
                            preview_epoch = speech_epoch
                            preview = executor.submit(transcribe, np.concatenate(chunks))
                        if silent_samples * 1000 >= SAMPLE_RATE * silence_ms:
                            stop_reason = "连续静音"
                            finished = True
                            break
                        if speaking_samples >= SAMPLE_RATE * 15:
                            stop_reason = "15 秒上限"
                            finished = True
                            break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    stopped_at = time.perf_counter()
    if last_speech_at is not None:
        print(
            f"[VAD] 最后一段人声到停止录音：{(stopped_at - last_speech_at) * 1000:.0f} ms"
            f" ｜ SILENCE_MS={silence_ms} ｜ 静音音频={silent_samples / SAMPLE_RATE * 1000:.0f} ms"
            f" ｜ 原因：{stop_reason}"
        )
    for warning in sorted(warnings):
        print(f"录音警告：{warning}")
    if not speech_started:
        print("5 秒内未检测到人声，已放弃录音。")
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    # 导入模块时不加载模型；第一次转写加载，以后复用。
    print("加载 Whisper small 模型（CPU / int8）；首次运行会下载模型，请稍候。")
    return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(audio: NDArray[np.float32]) -> str:
    """将 16 kHz、float32、一维单声道音频转为中文文本。"""
    if audio.size == 0:
        return ""
    with _TRANSCRIBE_LOCK:
        segments, _ = _get_model().transcribe(
            audio,
            language="zh",
            initial_prompt=(
                "以下是普通话的句子，可能提到 "
                "Dining Room、Living Room、客厅灯、餐厅灯。"
            ),
        )
        text = "".join(segment.text for segment in segments).strip()
        return _t2s.convert(text)


def main() -> None:
    _get_model()
    while True:
        try:
            audio = record()
            print(f"录音时长：{audio.size / SAMPLE_RATE:.2f} 秒")
            started = time.perf_counter()
            text = transcribe(audio)
            elapsed = time.perf_counter() - started
            print(f"识别文字：{text or '（未识别到文字）'}")
            print(f"转写耗时：{elapsed:.2f} 秒\n")
        except (EOFError, KeyboardInterrupt):
            raise
        except Exception as exc:
            print(f"本次测试失败：{exc}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n已退出")
