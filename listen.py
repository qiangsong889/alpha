"""回车控制录音，使用本地 Whisper 模型连续测试中文转写。

运行：alpha/.venv/bin/python alpha/listen.py
首次加载 small 模型时会自动下载模型文件。
"""

import time
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000


def record() -> NDArray[np.float32]:
    """按回车开始、再次回车停止；输入退出命令时抛出 EOFError。"""
    while True:
        command = input("按回车开始录音（q 退出）：").strip()
        if command.lower() in {"q", "exit", "quit", "退出"}:
            raise EOFError
        if not command:
            break
        print("请按回车开始，或输入 q 退出。")
    chunks: list[NDArray[np.float32]] = []
    warnings: set[str] = set()

    def callback(indata, frames, time_info, status):
        if status:
            warnings.add(str(status))
        # 回调缓冲区会复用，必须复制再保存。
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        input("录音中，再按回车停止：")

    for warning in sorted(warnings):
        print(f"录音警告：{warning}")
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0).reshape(-1)


@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    # 导入模块时不加载模型；第一次转写加载，以后复用。
    print("加载 Whisper small 模型（CPU / int8）；首次运行会下载模型，请稍候。")
    return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(audio: NDArray[np.float32]) -> str:
    """将 16 kHz、float32、一维单声道音频转为中文文本。"""
    if audio.size == 0:
        return ""
    segments, _ = _get_model().transcribe(
        audio,
        language="zh",
        initial_prompt=(
            "以下是普通话的句子，可能提到 "
            "Dining Room、Living Room、客厅灯、餐厅灯。"
        ),
    )
    return "".join(segment.text for segment in segments).strip()


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
