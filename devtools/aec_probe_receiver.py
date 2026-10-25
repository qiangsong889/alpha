"""J9 回声消除诊断的接收端：收 esp-alpha/xvf_aec_probe 推来的音频，对比播放中和停顿中的麦克风音量。

在 alpha 目录运行：
    .venv/bin/python -m devtools.aec_probe_receiver [秒数，默认 30]

包格式：uint32 序号 + uint32 是否在播放 + PCM（16 kHz、2 声道、int32，有效位在高 16 位）。
回声消除正常时，播放中的音量应接近停顿中的底噪；失效时播放中会大很多。
每次切换后的 0.5 秒不计（回声尾巴和切换瞬间）。两个声道分别存成 WAV，可以直接听。
"""
import socket
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np

PORT = 12345
RATE = 16000


def db(value):
    return 20 * np.log10(max(value, 1e-9) / 32768)


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    out = Path("data/aec_probe")
    out.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(5)
    print(f"等待 UDP 音频（端口 {PORT}）……")
    left, right, flags = [], [], []
    started = None
    first = last = None
    received = 0
    while True:
        try:
            data, sender = sock.recvfrom(4096)
        except socket.timeout:
            print("5 秒没有收到数据，结束。")
            break
        if started is None:
            started = time.perf_counter()
            print(f"开始接收，来自 {sender[0]}，录 {seconds:g} 秒")
        seq, playing = struct.unpack_from("<II", data)
        first = seq if first is None else first
        last = seq
        received += 1
        frames = np.frombuffer(data[8:], "<i4").reshape(-1, 2) >> 16
        left.append(frames[:, 0].astype(np.int16))
        right.append(frames[:, 1].astype(np.int16))
        flags.append(np.full(len(frames), playing, np.int8))
        if time.perf_counter() - started >= seconds:
            break
    sock.close()
    if not left:
        return
    left, right, flags = np.concatenate(left), np.concatenate(right), np.concatenate(flags)
    print(f"收到 {received} 包，丢包 {last - first + 1 - received}")
    for name, channel in (("ch1", left), ("ch2", right)):
        with wave.open(str(out / f"{name}.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(channel.tobytes())
    # 去掉每次切换后 0.5 秒
    settled = np.ones(len(flags), bool)
    for edge in np.flatnonzero(np.diff(flags)) + 1:
        settled[edge:edge + RATE // 2] = False
    for name, channel in (("声道 1", left), ("声道 2", right)):
        x = channel.astype(np.float64)
        on = x[(flags == 1) & settled]
        off = x[(flags == 0) & settled]
        rms_on = np.sqrt(np.mean(on ** 2)) if len(on) else 0
        rms_off = np.sqrt(np.mean(off ** 2)) if len(off) else 0
        print(f"{name}：播放中 {db(rms_on):6.1f} dBFS ｜ 停顿中 {db(rms_off):6.1f} dBFS ｜ 差 {db(rms_on) - db(rms_off):5.1f} dB")
    print(f"WAV：{out.resolve()}/ch1.wav、ch2.wav")


if __name__ == "__main__":
    main()
