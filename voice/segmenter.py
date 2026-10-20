"""16kHz PCM16 本地分段；feed 在音频处理任务调用，不在麦克风回调运行。"""

from dataclasses import dataclass
import math
from pathlib import Path
from uuid import uuid4

import numpy as np

RATE = 16000
FRAME = 512  # Silero ONNX 每次32ms；150/600ms向上取整为160/608ms。
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'data/models/silero_vad.onnx'


class SileroVAD:
    """官方 input/state/sr ONNX 接口的 numpy 实现，无 torch/LiveKit 依赖。

    与官方 OnnxWrapper 相同，保持64采样的上下文及循环状态。
    来源：https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
    """
    def __init__(self, model_path=DEFAULT_MODEL):
        import onnxruntime as ort  # type: ignore[import-untyped]
        if not Path(model_path).is_file():
            raise FileNotFoundError(f'缺少 Silero ONNX：{model_path}；请设置 SILERO_MODEL_PATH 或 --model')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
        if {p.name for p in self.session.get_inputs()} != {'input', 'state', 'sr'}:
            raise ValueError('需要官方 Silero input/state/sr 版本的 ONNX 模型')
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, frame):
        if frame.shape != (FRAME,):
            raise ValueError('Silero 需要512个16kHz采样')
        audio = np.concatenate((self.context, frame.reshape(1, FRAME)), axis=1)
        probability, self.state = self.session.run(None, {
            'input': audio, 'state': self.state, 'sr': np.array(RATE, dtype=np.int64)})
        self.context = audio[:, -64:].copy()
        return float(probability.reshape(-1)[0])


@dataclass(frozen=True)
class SpeechStarted:
    seg_id: str
    start_s: float              # 音频起点，包含开口前缓冲。
    speech_start_s: float
    confirmed_s: float
    continuation: bool = False  # 15秒切段后继续，不能解释为整句已说完。


@dataclass(frozen=True)
class SegmentReady:
    seg_id: str
    pcm: bytes
    start_s: float
    end_s: float
    speech_end_s: float
    reason: str                # silence / max_duration / eof


class Segmenter:
    def __init__(self, detector=None, *, onset_ms=150, silence_ms=600, pre_ms=400,
                 min_ms=100, max_s=15, threshold=.5):
        values = (onset_ms, silence_ms, min_ms, max_s)
        if any(not math.isfinite(v) or v <= 0 for v in values) or not math.isfinite(pre_ms) or pre_ms < 0:
            raise ValueError('时间参数必须有限，且除前置缓冲外均大于零')
        if not 0 < threshold < 1 or max_s * 1000 < onset_ms:
            raise ValueError('阈值必须在0到1之间，最长时间不能小于开口确认时间')
        self.detector = detector if detector is not None else SileroVAD()
        self.onset = math.ceil(onset_ms * RATE / 1000)
        self.silence = math.ceil(silence_ms * RATE / 1000)
        self.pre = math.ceil(pre_ms * RATE / 1000)
        self.minimum = math.ceil(min_ms * RATE / 1000)
        self.maximum = math.ceil(max_s * RATE)
        self.threshold = threshold
        self.remainder = bytearray()
        self.history = bytearray()
        self.position = self.voiced_run = self.silent = 0
        self.active = None
        self.continuing = False
        self.last_probability = 0.0

    def feed(self, pcm: bytes) -> list[SpeechStarted | SegmentReady]:
        if len(pcm) % 2:
            raise ValueError('PCM16 字节长度必须为偶数')
        self.remainder.extend(pcm)
        events: list[SpeechStarted | SegmentReady] = []
        offset = 0
        while len(self.remainder) - offset >= FRAME * 2:
            events.extend(self._frame(bytes(self.remainder[offset:offset + FRAME * 2])))
            offset += FRAME * 2
        del self.remainder[:offset]
        return events

    def _frame(self, pcm):
        samples = len(pcm) // 2
        floats = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
        probability = self.detector(np.pad(floats, (0, FRAME - samples)))
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('VAD 概率无效')
        self.last_probability = probability
        voiced = probability >= self.threshold
        self.position += samples
        events = []
        if self.active is None:
            self.history.extend(pcm)
            # 确认用掉的160ms也要保留，不能挤掉开口前400ms。
            keep = (self.pre + self.onset + FRAME) * 2
            if len(self.history) > keep:
                del self.history[:-keep]
            self.voiced_run = self.voiced_run + samples if voiced else 0
            self.silent = 0 if voiced else self.silent + samples
            if self.silent >= self.silence:
                self.continuing = False
            required = self.minimum if self.continuing else self.onset
            if not voiced or self.voiced_run < required:
                return []
            # 常规开口保留400ms+实际确认段；强制切段后只取尚未发送的新音频。
            take = min(len(self.history), (self.pre + self.voiced_run) * 2)
            audio = bytes(self.history[-take:])
            start = self.position - len(audio) // 2
            self.active = {'id': uuid4().hex, 'audio': bytearray(audio), 'start': start,
                           'speech_start': self.position - self.voiced_run,
                           'last_voice': self.position, 'voiced': self.voiced_run}
            events.append(SpeechStarted(self.active['id'], start / RATE,
                          self.active['speech_start'] / RATE, self.position / RATE, self.continuing))
            self.history.clear()
            self.continuing = False
            self.silent = 0
        else:
            self.active['audio'].extend(pcm)
            self.silent = 0 if voiced else self.silent + samples
            if voiced:
                self.active['last_voice'] = self.position
                self.active['voiced'] += samples
        if self.silent >= self.silence:
            events.extend(self._finish('silence'))
        elif self.position - self.active['speech_start'] >= self.maximum:
            events.extend(self._finish('max_duration'))
            self.continuing = True
        return events

    def _finish(self, reason):
        active, self.active = self.active, None
        self.voiced_run = self.silent = 0
        self.history.clear()
        if active is not None and reason == 'silence' and self.pre:
            self.history.extend(active['audio'][-self.pre * 2:])
        if active is None or active['voiced'] < self.minimum or len(active['audio']) < self.minimum * 2:
            return []
        return [SegmentReady(active['id'], bytes(active['audio']), active['start'] / RATE,
                             self.position / RATE, active['last_voice'] / RATE, reason)]

    def flush(self) -> list[SpeechStarted | SegmentReady]:
        """仅在文件/录音真正结束时调用；不为补齐模型输入而发送伪造静音。"""
        events: list[SpeechStarted | SegmentReady] = []
        if self.remainder:
            events.extend(self._frame(bytes(self.remainder)))
            self.remainder.clear()
        events.extend(self._finish('eof'))
        self.continuing = False
        if hasattr(self.detector, 'reset'):
            self.detector.reset()
        return events
