"""回声消除实测：通过 XVF3800 接的音箱播放 Alpha 的 TTS，同时录下 XVF3800 的两个声道和 Mac 自带麦克风。

运行（在 alpha/ 目录，人在电脑前按提示操作；全程不要调音量、不要挪设备）：
    .venv/bin/python -m devtools.xvf_echo_probe

回声消除要听一段时间的播放才会收敛，音量或位置一变就要重新适应，所以一次跑完、条件固定：
1. 预热：播放约 30 秒，让回声消除收敛（不计入结果）；
2. 安静 + 播放：每个声道里还残留多少 Alpha 自己的声音；
3. 不播放，你说一句话：你的声音在每个声道里的"标准音量"；
4. 边播放边说同一句话：和第 3 步比，你的声音被保留了多少。
不调用任何模型，不改 Alpha。录音保存在 data/xvf-probe/（data/ 不进 git）。
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
import wave

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from core.paths import PROJECT_ROOT

XVF_NAME = 'reSpeaker Flex'
MAC_MIC_NAME = 'MacBook Pro Microphone'
RATE = 16000
LEAD_S, TAIL_S, VOICE_S = 2.0, 1.0, 6.0
SENTENCE = '今天天气怎么样，帮我把客厅灯打开'
SPEECH = [PROJECT_ROOT / 'data' / 'tts-preview' / name for name in ('ethan-1.wav', 'ethan-2.wav')]
OUTPUT = PROJECT_ROOT / 'data' / 'xvf-probe'
CHANNELS = {'xvf_left': 'XVF 左声道', 'xvf_right': 'XVF 右声道', 'mac_mic': 'Mac 自带麦克风'}
WINDOW_S = 0.25


def find_device(name: str, kind: str) -> int:
    for index, device in enumerate(sd.query_devices()):
        if name in device['name'] and device[f'max_{kind}_channels'] > 0:
            return index
    raise SystemExit(f'找不到{kind}设备：{name}')


def load_speech() -> np.ndarray:
    """两段 24kHz TTS 拼起来（中间留 0.4 秒），转成 16kHz float32。"""
    parts = []
    for path in SPEECH:
        with wave.open(str(path)) as handle:
            rate = handle.getframerate()
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype='<i2').astype(np.float32) / 32768
        count = round(samples.size * RATE / rate)
        parts += [np.interp(np.arange(count) * rate / RATE, np.arange(samples.size), samples).astype(np.float32),
                  np.zeros(int(0.4 * RATE), np.float32)]
    return np.concatenate(parts)


class Recorder:
    def __init__(self, device: int, channels: int, rate: int):
        self.rate, self.chunks, self.lock = rate, [], threading.Lock()
        self.stream = sd.InputStream(device=device, channels=channels, samplerate=rate,
                                     dtype='float32', callback=self._callback)

    def _callback(self, data, frames, timing, status):
        with self.lock:
            self.chunks.append(data.copy())

    def audio(self) -> np.ndarray:
        with self.lock:
            return np.concatenate(self.chunks) if self.chunks else np.zeros((0, self.stream.channels), np.float32)


def dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    return 20 * np.log10(max(rms, 1e-7))


def save(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())


def to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE:
        return samples
    count = round(samples.size * RATE / rate)
    return np.interp(np.arange(count) * rate / RATE, np.arange(samples.size), samples).astype(np.float32)


def record(name: str, xvf: int, mac: int, *, play: np.ndarray | None, seconds: float) -> dict[str, np.ndarray]:
    """录下三路声音（统一成 16kHz）；play 不为空时，先录 LEAD_S 秒底噪再播放。"""
    xvf_rec = Recorder(xvf, 2, RATE)
    mac_rate = int(sd.query_devices(mac)['default_samplerate'])
    mac_rec = Recorder(mac, 1, mac_rate)
    with xvf_rec.stream, mac_rec.stream:
        if play is None:
            time.sleep(seconds)
        else:
            time.sleep(LEAD_S)
            sd.play(np.column_stack([play, play]), RATE, device=xvf)
            sd.wait()
            time.sleep(TAIL_S)
    xvf_audio = xvf_rec.audio()
    tracks = {'xvf_left': xvf_audio[:, 0], 'xvf_right': xvf_audio[:, 1],
              'mac_mic': to_16k(mac_rec.audio()[:, 0], mac_rate)}
    for key, samples in tracks.items():
        save(OUTPUT / f'{name}_{key}.wav', samples, RATE)
    return tracks


def windows(samples: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """每 WINDOW_S 秒一个电平（dBFS）。"""
    size = int(WINDOW_S * RATE)
    part = samples[int(start_s * RATE):int(end_s * RATE)]
    return np.array([dbfs(part[i:i + size]) for i in range(0, len(part) - size + 1, size)])


def echo_correlation(samples: np.ndarray, speech: np.ndarray) -> float:
    """录音包络和播放内容包络的相关性（允许 0～400ms 延迟），接近 0 说明回声基本消掉了。"""
    hop = 160
    def envelope(x):
        n = len(x) // hop
        return np.log(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-6)
    reference = envelope(speech)
    recorded = envelope(samples)
    start = int(LEAD_S * RATE / hop)
    scores = []
    for lag in range(40):
        part = recorded[start + lag:start + lag + len(reference)]
        if len(part) == len(reference) and part.std() > 0:
            scores.append(np.corrcoef(part, reference)[0, 1])
    return max(scores, default=float('nan'))


def report(speech: np.ndarray, quiet: dict, voice: dict, both: dict) -> None:
    play_end = LEAD_S + speech.size / RATE
    # 你说话的时间段：第 4 步比第 2 步（同样的播放）明显更响的窗口，用 Mac 麦克风判断。
    talking = windows(both['mac_mic'], LEAD_S, play_end) - windows(quiet['mac_mic'], LEAD_S, play_end) > 6
    print('\n结果（dBFS，越低越安静）')
    print(f'  {"":<14}{"底噪":>7}{"回声残留":>9}{"高出底噪":>9}{"回声相关":>9}{"单独说话":>9}{"边播边说":>9}{"保留":>8}')
    for key, label in CHANNELS.items():
        floor = dbfs(quiet[key][int(0.3 * RATE):int((LEAD_S - 0.2) * RATE)])
        echo = dbfs(quiet[key][int((LEAD_S + 0.3) * RATE):int(play_end * RATE)])
        alone_windows = windows(voice[key], 0, VOICE_S)
        alone = float(np.mean(np.sort(alone_windows)[-6:]))  # 最响的 1.5 秒
        over = windows(both[key], LEAD_S, play_end)
        over_voice = float(np.mean(over[talking])) if talking.any() else float('nan')
        print(f'  {label:<14}{floor:>7.1f}{echo:>9.1f}{echo - floor:>+9.1f}{echo_correlation(quiet[key], speech):>+9.2f}'
              f'{alone:>9.1f}{over_voice:>9.1f}{over_voice - alone:>+8.1f}')
    print(f'\n  判断为你在说话的时间：{talking.sum() * WINDOW_S:.1f} 秒'
          '（"保留"越接近 0，说明边播边说时你的声音保留得越完整；回声残留时也会偏高）')


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    xvf, mac = find_device(XVF_NAME, 'output'), find_device(MAC_MIC_NAME, 'input')
    print(f'播放和录音：{sd.query_devices(xvf)["name"]} ｜ 对照：{sd.query_devices(mac)["name"]}')
    print('全程不要调音量、不要挪音箱和板子。')
    speech = load_speech()
    input('\n第 1 步 预热：播放约 30 秒，让回声消除收敛，请保持安静。准备好按回车……')
    sd.play(np.column_stack([np.concatenate([speech] * 3)] * 2), RATE, device=xvf)
    sd.wait()
    input('第 2 步：保持安静，测回声残留。准备好按回车……')
    quiet = record('quiet', xvf, mac, play=speech, seconds=0)
    input(f'第 3 步：按回车后，请正常说一句话：“{SENTENCE}”（不播放，录 {VOICE_S:.0f} 秒）……')
    print('  现在说……', flush=True)
    voice = record('voice', xvf, mac, play=None, seconds=VOICE_S)
    input(f'第 4 步：按回车后，等音箱开始说话，再说同一句话：“{SENTENCE}”……')
    both = record('talk', xvf, mac, play=speech, seconds=0)
    report(speech, quiet, voice, both)
    print(f'\n录音已保存到 {OUTPUT}')


if __name__ == '__main__':
    main()
