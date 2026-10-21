"""本地控制 PCM16/16kHz 上传；不负责断句、提交任务或判断对话对象。"""
from collections import deque
import logging
import math
import time
import numpy as np
import os

from core.settings import setting

import webrtcvad  # type: ignore[import-untyped]

logger = logging.getLogger('alpha.qwen_alpha')


class UploadGate:
    FRAME_BYTES = 640  # 20ms * 16000Hz * 2 bytes，单声道。

    def __init__(self, *, pre_ms=400, tail_ms=1000, mode=3, onset_ms=80, min_rms=.003, debug=False, detector=None):
        if type(pre_ms) is not int or not 100 <= pre_ms <= 1000 or pre_ms % 20:
            raise ValueError('本地 VAD 前置缓冲必须为 100～1000ms，且为 20 的倍数')
        if type(tail_ms) is not int or not 200 <= tail_ms <= 6200 or tail_ms % 20:
            raise ValueError('本地 VAD 尾部必须为 200～6200ms，且为 20 的倍数')
        if type(mode) is not int or mode not in range(4):
            raise ValueError('本地 VAD mode 必须是 0～3')
        if type(onset_ms) is not int or not 20 <= onset_ms <= 200 or onset_ms % 20:
            raise ValueError('本地 VAD 开口确认必须为 20～200ms，且为 20 的倍数')
        if isinstance(min_rms, bool) or not math.isfinite(min_rms) or not 0 <= min_rms <= .1:
            raise ValueError('本地 VAD 最低 RMS 必须为 0～0.1 的有限数')
        self.onset_ms, self.min_rms, self.debug = onset_ms, min_rms, debug
        self.pre_ms, self.tail_ms, self.mode = pre_ms, tail_ms, mode
        self.detector = detector if detector is not None else webrtcvad.Vad(mode)
        self.reset()

    def reset(self):
        # 设备重开/连接更换后不能补发旧音频。
        # 额外保留确认期间的音频：确认用掉的 80ms 不能挤掉开口前 400ms。
        self.before = deque(maxlen=(self.pre_ms + self.onset_ms) // 20)
        self.voiced_run_ms = 0
        self.diag_at = time.monotonic()
        self.diag_frames = self.diag_raw = self.diag_qualified = 0
        self.diag_rms_sum = self.diag_rms_max = 0.0
        self.open = False
        self.silent_ms = 0
        self.captured_bytes = 0
        self.forwarded_bytes = 0

    def feed(self, chunk: bytes) -> bytes | None:
        """每次输入一帧；返回要上传的音频，None 表示只在本地监听。"""
        if len(chunk) != self.FRAME_BYTES:
            raise ValueError('上传 VAD 需要 20ms PCM16 单声道 16kHz 音频')
        self.captured_bytes += len(chunk)
        pcm = np.frombuffer(chunk, dtype='<i2').astype(np.float32)
        rms = float(np.sqrt(np.mean(pcm * pcm))) / 32768
        raw_voiced = self.detector.is_speech(chunk, 16000)
        voiced = raw_voiced and rms >= self.min_rms
        self.diagnose(rms, raw_voiced, voiced)
        if not self.open:
            self.before.append(chunk)
            self.voiced_run_ms = self.voiced_run_ms + 20 if voiced else 0
            if self.voiced_run_ms < self.onset_ms:
                return None
            self.voiced_run_ms = 0
            self.open = True
            self.silent_ms = 0
            result = b''.join(self.before)
            self.before.clear()
            logger.info('[本地VAD] 疑似人声连续 %dms，RMS %.4f ≥ %.4f，开始上传（含前置和确认缓冲）',
                        self.onset_ms, rms, self.min_rms)
        else:
            result = chunk
            self.silent_ms = 0 if voiced else self.silent_ms + 20
            if self.silent_ms >= self.tail_ms:
                self.open = False
                self.before.clear()
                self.voiced_run_ms = 0
                logger.info('[本地VAD] 尾部静音 %dms，停止上传，继续本地监听', self.tail_ms)
        self.forwarded_bytes += len(result)
        return result

    def diagnose(self, rms, raw, qualified):
        if not self.debug:
            return
        self.diag_frames += 1
        self.diag_raw += bool(raw)
        self.diag_qualified += bool(qualified)
        self.diag_rms_sum += rms
        self.diag_rms_max = max(self.diag_rms_max, rms)
        now = time.monotonic()
        if now - self.diag_at >= 10:
            logger.info('[VAD诊断] RMS均值 %.4f / 最大 %.4f ｜ 最低 %.4f ｜ 原始人声 %.1f%% / 达标 %.1f%% ｜ 上传中=%s',
                        self.diag_rms_sum / self.diag_frames, self.diag_rms_max, self.min_rms,
                        self.diag_raw * 100 / self.diag_frames, self.diag_qualified * 100 / self.diag_frames,
                        self.open)
            self.diag_at = now
            self.diag_frames = self.diag_raw = self.diag_qualified = 0
            self.diag_rms_sum = self.diag_rms_max = 0.0

    def report(self):
        if not self.captured_bytes:
            return
        logger.info('[上传统计] 本地录音 %.2fs ｜ 送往发送器 %.2fs ｜ 留在本地 %.1f%%',
                    self.captured_bytes / 32000, self.forwarded_bytes / 32000,
                    (1 - self.forwarded_bytes / self.captured_bytes) * 100)


def make_upload_gate():
    """仅 Qwen Alpha 入口调用。环境变量=0 可恢复原来的连续上传。"""
    if not setting('QWEN_ALPHA_LOCAL_VAD'):
        logger.info('[上传模式] 连续上传（本地 VAD 门控关闭）')
        return None
    if setting('QWEN_VAD') != 'server_vad':
        raise ValueError('本地上传门控需要 QWEN_VAD=server_vad，以尾部静音结束转写')
    silence_ms = setting('QWEN_SILENCE_MS')
    minimum_tail = ((silence_ms + 200 + 19) // 20) * 20
    configured_tail = setting('QWEN_ALPHA_VAD_TAIL_MS')
    # 没配置尾部时长时按静音时长自动计算；这个默认值依赖另一个参数，所以不写在清单里。
    tail_ms = max(1000, minimum_tail) if configured_tail is None else configured_tail
    if tail_ms < minimum_tail:
        raise ValueError(f'QWEN_ALPHA_VAD_TAIL_MS 需要至少 {minimum_tail}ms，保证服务端正常结束转写')
    gate = UploadGate(pre_ms=setting('QWEN_ALPHA_VAD_PRE_MS'),
                      tail_ms=tail_ms, mode=setting('QWEN_ALPHA_VAD_MODE'),
                      onset_ms=setting('QWEN_ALPHA_VAD_ONSET_MS'),
                      min_rms=setting('QWEN_ALPHA_VAD_MIN_RMS'),
                      debug=setting('QWEN_ALPHA_VAD_DEBUG'))
    logger.info('[上传模式] 本地VAD ｜ 前置 %dms ｜ 开口确认 %dms ｜ 尾部 %dms ｜ mode=%d ｜ 最低RMS=%.4f',
                gate.pre_ms, gate.onset_ms, gate.tail_ms, gate.mode, gate.min_rms)
    return gate
