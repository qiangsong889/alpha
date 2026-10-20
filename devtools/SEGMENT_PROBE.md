# A2：本地分段与串行 ASR

此入口不导入或修改 qwen_alpha，不连接 Jev、AlphaBrain 或 MCP。

## 准备和运行

现有 alpha venv 已有 numpy、sounddevice、websockets、python-dotenv、onnxruntime。
新环境额外安装 `onnxruntime` 即可；不安装或导入 torch、silero-vad、LiveKit。
本次直接用 numpy 包装官方 ONNX 的 input/state/sr 接口。

模型下载到被 git 忽略的 data 目录：

```bash
mkdir -p data/models
curl -fL https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx -o data/models/silero_vad.onnx
.venv/bin/python -m devtools.segment_probe --checklist --diagnostics
```

模型 SHA256：`2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f`。
也可用 `--model /path/to/silero_vad.onnx` 或 `SILERO_MODEL_PATH` 指定同接口模型。
模型上下文和循环状态参考[官方 OnnxWrapper](https://github.com/snakers4/silero-vad/blob/v5.1.2/src/silero_vad/utils_vad.py)。

读取项目 `.env` 的 DASHSCOPE_API_KEY、QWEN_WORKSPACE_ID、QWEN_REGION。
麦克风由 `--device` 或 REALTIME_INPUT_DEVICE 指定，默认系统输入设备。
`--checklist` 会提示五句45秒、单字10秒、停顿20秒、咳嗽10秒、安静60秒，共145秒。
普通模式可用 `--duration 60`；Ctrl+C 关闭音频流及连接。默认不保存录音。

```bash
# WAV验证，不打开麦克风；offline不连接任何云端服务。
.venv/bin/python -m devtools.segment_probe --wav paused.wav --offline
# 同一个WAV，经分段后串行交给真实ASR。
.venv/bin/python -m devtools.segment_probe --wav paused.wav
```

`--diagnostics` 每10秒及退出时打印音量和Silero概率；RMS只用于观察，不作为检测门槛。

## 分段模块

`voice.segmenter.Segmenter.feed(pcm)` 接收任意块长的16kHz、单声道、小端PCM16，
返回 SpeechStarted / SegmentReady 列表。内部按512采样（32ms）运行Silero：

- 连续150ms达到概率0.5：按帧取整到160ms确认开口。
- 连续600ms未达到阈值：取整到608ms结束。保留开口前400ms，确认音频不挤掉前置缓冲。
- 不到100ms的人声不输出可提交段；不会只凭前置静音长度提交。
- 从本段人声开始约15秒仍未结束：reason=max_duration，后续SpeechStarted带continuation=True。
  这是物理切段，不是用户整句话已说完；连续切段之间不重复或丢失人声采样。
- `flush()` 仅用于文件或录音真正结束，reason=eof；补齐模型输入的零不发送给ASR。

SegmentReady包含本地seg_id、pcm、start_s、end_s、speech_end_s、reason。
起止时间基于音频采样计数；起点含前置缓冲，终点含尾部静音。
SpeechStarted确认前的短噪音不会产生事件；最长切段后的不足100ms尾段也不提交。

## 串行 ASR 模块

`voice.asr_client.AsrClient` 的 `submit(segment)` 放入有界队列，`run()` 是单个worker。
事件回调收到 AsrEvent(kind=transcript|failed, seg_id, text, elapsed_s,
commit_to_result_s, queue_s)。测试可注入假的connector与回调。

上一段completed或failed前，下一段连append也不发送。上传按100ms音频块尽快发送，
收到完成结果才关联本地ID，不要求服务端item_id。独立接收任务持续读取中间事件，
避免发送期间接收缓冲堆积。中间结果不向Jev或大脑提交。

默认握手、上传各有5秒超时；最终结果的5秒从commit发出开始计算，不含排队时间。
出错、超时、空转写都会failed，断开并丢弃该连接的接收队列；当前段不自动重试，
下一段新建连接。不能依赖错误code，只有message同样处理。
程序取消时，正在处理与尚在队列中的段落均报告failed。成功时复用同一连接。

麦克风回调只复制PCM并通过call_soon_threadsafe入队。处理任务用to_thread串行执行
Silero，再把完整段送ASR队列，ASR不会阻塞麦克风回调或事件循环。
原始音频队列256块（约8秒），ASR队列8段；队列压力导致原始音频溢出时明确停止，
避免静默丢音后继续错配。最长段包含前置缓冲，实际PCM可能略长于15秒。

## 2026-10-20 自动验证

10项新增单元测试通过；全量726项通过；三个新模块的mypy检查通过。
覆盖两段串行复用、第二段append晚于第一段completed、超时和错误后新连接、
旧连接结果隔离、短音频不发送、分段时刻、96ms噪音、15秒切段、部分帧不填充上传。

真实Silero ONNX验证（没有导入torch）：

- 80ms合成噪音：0个事件。
- 21.943秒中文长WAV：切成 `[0, 15.072] max_duration` 和 `[15.072, 22.560] silence`。
- 两句话之间插入1秒零值静音：准确切成两段，见下表。

| 段落 | 含前置缓冲的起点 | 最后人声 | 段落结束 | ASR文字 |
| --- | --- | --- | --- | --- |
| 1 | 0.656s | 4.864s | 5.472s | 这是第一段语音测试，今天的天气怎么样？ |
| 2 | 5.552s | 11.296s | 11.904s | 这是第二段语音测试，请确认同一个连接可以继续识别。 |

真实ASR两段均成功，无合并或缺少结果：commit→文字为1.023s、0.642s。
第二段本地排队2.093s。这是文件快速送入的协议测试，不是实时麦克风延迟。
原始日志 `/tmp/alpha-segment-offline.log`、`/tmp/alpha-segment-asr-smoke.log`。

## 现场麦克风：复测通过

首轮 MacBook Pro Microphone 按清单运行145秒，没有产生人声事件。用户随后明确说明
“刚才还没说，现在开始”，因此首轮不计入说话验收，也不将它误报为模型漏检。

补充的纯本地12秒诊断接收374个音频块，未上传ASR：

| 窗口 | 块数 | RMS均值 | RMS最大 | Silero概率最大 |
| --- | --- | --- | --- | --- |
| 约10秒 | 313 | 0.00059 | 0.00135 | 0.029 |
| 剩余约2秒 | 61 | 0.00056 | 0.00154 | 0.002 |

此诊断确认输入有底噪而非全零。随后按原阈值重跑现场验收，没有降低概率门槛。
首轮日志保留：`/tmp/alpha-segment-mic.log`、`/tmp/alpha-segment-mic-levels.log`。

第二轮使用同一麦克风，清单运行145秒，正常退出。结果：

| 项目 | 观察 |
| --- | --- |
| 连续5句 | 天气、关灯、查询购物清单、明天八点提醒、你好，全部正确识别 |
| 单字“对” | 正确识别，最后人声到文字约0.954s |
| 句中停顿 | 分别识别“把牛奶。”和“和鸡蛋加到购物清单。”，没有合并或缺少结果 |
| 咳嗽 | 本次咳嗽未触发段落；不等于保证任意咳嗽都会被过滤 |
| 安静1分钟 | 1875个32ms音频块，恰好60s，0次开始、0个段落、0次ASR上传 |
| 失败/卡住 | 0次failed，所有生成的段落均收到转写，程序正常退出 |

另有一句额外人声被正常识别，因此总数是9段（指定5句、单字1句、停顿2段、额外1句），
不是漏段或重复结果。单字“对”实际在59.104秒开口，落在程序的“停顿”时间窗；
按转写内容验收通过，不因固定阶段标签将其误判为单字漏检。

指定5句的“最后人声→文字”估计为2.574s、1.252s、0.995s、1.010s、0.952s。
首句包含冷连接；后续各项约0.95～1.25s。此处仅测转写，不含Jev、大脑或TTS。
咳嗽附近的10秒诊断窗口里，RMS峰值为0.10562、Silero最高概率0.023，未触发段落。
后续安静窗口的概率也一直低于0.5门槛。

现场复测日志：`/tmp/alpha-segment-mic-retest.log`。默认没有保存原始麦克风录音。
代码仍未接入qwen_alpha，未修改现有输入模式或.env。重新现场测试可运行：

```bash
.venv/bin/python -m devtools.segment_probe --checklist --diagnostics
```
