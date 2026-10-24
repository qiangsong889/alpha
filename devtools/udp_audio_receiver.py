"""J9 第一步的接收端：收 esp-alpha/xvf_udp_probe 推来的 UDP 音频，存成 WAV 并统计丢包。

在 alpha 目录运行（Mac 防火墙首次会询问是否允许 Python 接收连接）：
    .venv/bin/python -m devtools.udp_audio_receiver [秒数，默认 10]

包格式：uint32 序号（小端）+ PCM（16 kHz、2 声道、int32）。XVF3800 的有效位在高 16 位，
所以每个声道取高 16 位存成 16 位单声道 WAV（Alpha 的 ASR 也用 16 位），分别听哪个声道是 ASR 波束。
"""
import socket
import struct
import sys
import time
import wave
from pathlib import Path

PORT = 12345
SAMPLE_RATE = 16000


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    out = Path("data/udp_probe")
    out.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(5)
    print(f"等待 UDP 音频（端口 {PORT}）……")
    channels = [bytearray(), bytearray()]
    first_seq = last_seq = None
    received = reordered = 0
    started = None
    sender = None
    while True:
        try:
            data, sender = sock.recvfrom(4096)
        except socket.timeout:
            print("5 秒没有收到数据，结束。")
            break
        if started is None:
            started = time.perf_counter()
            print(f"开始接收，来自 {sender[0]}，录 {seconds:g} 秒")
        seq = struct.unpack_from("<I", data)[0]
        if last_seq is not None and seq <= last_seq:
            reordered += 1
        first_seq = seq if first_seq is None else first_seq
        last_seq = seq if last_seq is None else max(last_seq, seq)
        received += 1
        pcm = data[4:]
        for i in range(0, len(pcm) - 7, 8):
            # 每帧左右各一个 int32，取高 16 位（小端时是每个 4 字节的后 2 字节）。
            channels[0] += pcm[i + 2:i + 4]
            channels[1] += pcm[i + 6:i + 8]
        if time.perf_counter() - started >= seconds:
            break
    sock.close()
    if started is None:
        return
    elapsed = time.perf_counter() - started
    expected = last_seq - first_seq + 1
    lost = expected - received + reordered
    audio_s = len(channels[0]) / 2 / SAMPLE_RATE
    print(f"收到 {received} 包，序号跨度 {expected}，丢包 {lost}（{lost / expected:.1%}），乱序 {reordered}")
    print(f"音频 {audio_s:.2f} 秒 / 实际 {elapsed:.2f} 秒")
    for index, pcm in enumerate(channels, 1):
        path = out / f"ch{index}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(bytes(pcm))
        samples = memoryview(bytes(pcm)).cast("h")
        peak = max((abs(s) for s in samples), default=0)
        print(f"声道 {index}：峰值 {peak * 100 / 32768:.0f}% → {path.resolve()}")


if __name__ == "__main__":
    main()
