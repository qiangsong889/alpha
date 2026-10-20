# A1：独立 ASR 手动模式探测

在 alpha 目录运行：

```bash
.venv/bin/python -m devtools.asr_probe first.wav second.wav
```

支持两个不同 WAV，也可以重复传入同一个。文件必须为 16kHz、单声道、16 位整数
PCM WAV，至少 100ms；不会把 WAV 文件头发送给服务端，也不自动转换格式。
默认逐块按实际音频时长发送，每块 100ms，最后一块可以不足 100ms。

读取项目 `.env` 中的 `DASHSCOPE_API_KEY`、`QWEN_WORKSPACE_ID`、`QWEN_REGION`
（默认 singapore）。只连接 `qwen3-asr-flash-realtime`，不导入 `qwen_alpha`、
不启动麦克风、不调用 Jev/大脑/TTS/MCP。`--timeout 30` 设置每阶段等待事件的超时。

接收任务在上传期间持续运行，每个事件都打印原始 JSON、类型、服务端 item_id
及时间。commit 前的事件先以连接时间打印；commit 后按对应段落的 commit 起点计时。
每段最后汇总三个相对 commit 的延迟，包括 commit 之前已经返回的首个中间结果。
缺失的 item_id 打印为 `-`，不会伪造服务端 ID。

## 2026-10-20 在线结果

使用 macOS Tingting 本地合成的两段测试 WAV，不含私人录音：

1. 3.88 秒：“这是第一段语音测试，今天的天气怎么样。”
2. 5.41 秒：“这是第二段语音测试，请确认同一个连接可以继续识别。”

两段在**同一连接**依次 append、commit 并收到正确的最终文字，中间没有
`session.finish` 或重新连接。以下为单次协议探测结果，不代表真实麦克风端到端延迟：

| 段落 | commit → committed | commit → 首个中间结果 | commit → completed |
| --- | --- | --- | --- |
| 第1段 | 0.202s | -3.295s | 0.422s |
| 第2段 | 0.202s | -4.824s | 0.399s |

负数表示中间结果在上传过程中、commit 之前已到达。不能假设所有识别都等 commit
之后才开始。实时预览按当前事件的 `text + stash` 读取，不能将每次事件不断累加；
最终文字读取 `.completed` 的 `transcript`。

最后故意对空缓冲区 commit，0.332s 后收到：

```json
{"type":"invalid_request_error","message":"Error committing input audio buffer, maybe no invalid audio stream."}
```

这是实际 error 对象；本次没有 `error.code` 字段。随后发送一次 `session.finish`，
正常收到 `session.finished`，探测脚本退出码为 0（预期的空提交错误已观察到）。

## 与文档示例不同的地方

- 发送 `turn_detection: null` 后，实际 `session.updated` 省略该字段。脚本允许省略，
  拒绝非空配置，并检查没有在手动 commit 前收到自动提交或 VAD 起止事件。
- 实际的 committed、中间结果、最终结果均未携带 `item_id`，created 里也没有 item.id。
  A1 串行等待上一段完成才发下一段，可以用本地段号关联；保留服务端字段缺失这一事实。
  A3 不能只依赖服务端 ID，也不能据此假设无 ID 的多个并发提交可以可靠关联。
- `.text` 中间结果在 commit 前就会到达。后续若提前问 Jev，仍需避免用不稳定的预览
  直接触发大脑执行工具。

文档参考：[客户端事件](https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-client-events)、
[服务端事件](https://www.alibabacloud.com/help/en/model-studio/qwen-asr-realtime-server-events)。

离线测试：`.venv/bin/python -m unittest tests.test_asr_probe`，8 项通过。
测试使用假连接；覆盖 WAV 检查、两段复用、空提交、无中间结果、null 省略、无 item_id、
commit 前的中间结果、超时清理和日志脱敏。

## A1.5：连续 commit，2026-10-20 三轮实测

```bash
.venv/bin/python -m devtools.asr_probe --back-to-back --timeout 60 long.wav short.wav
```

新模式仍每块发送 100ms 音频，但**不按实际时长 sleep**，尽快上传两段并连续 commit，
不等第一段的 committed 或 completed。目的是制造两个在途提交；如果实时慢速上传，
短句上传完之前第一段可能已经完成，便测不到这个情形。

所有服务端事件以连接起点记录，不把无 item_id 的结果强行绑定到“当前段”或队首。
completed 的原文、接收顺序和当时已发送的 commit 数分别记录；即使超时，也打印
已经到达的结果及数量。该模式仅是并发协议压力探测，不代表正常实时麦克风的延迟。

三轮使用相同的本地 Tingting 合成 WAV：

- 长句 21.943s：太阳系介绍，包含“太阳是一颗恒星”“土星拥有美丽的光环”等。
- 短句 1.347s：“苹果很好吃。”
- 每轮新建连接，连接内连续提交两段；三个试验均确实在首个 completed 到达前发出两次 commit。
- 第二个结果等待上限为 60 秒；三轮均超时退出，不按成功计入。

| 轮次 | committed 数 | completed 数 | 唯一 completed 的内容 | 第二个结果 |
| --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 太阳系长句，没有苹果短句 | 等待60秒未收到 |
| 2 | 2 | 1 | 太阳系长句 + 完整“苹果很好吃。”，合在一条里 | 等待60秒未收到 |
| 3 | 2 | 1 | 太阳系长句，末尾多出“苹。” | 等待60秒未收到 |

没有得到两条可用于验证顺序的独立 completed，因此**不能声称3/3顺序正确，更不能据此启用FIFO**。
实测还暴露了边界合并或缺少最终结果的问题。这里只描述观测，不推断服务端内部原因。
A3 先用单段在途：上一段 completed 前，后续音频保留在本地，连 append 也不发出。
本地 Silero 可以继续采集和分段，但上传任务串行处理队列。

三轮完整转写、用量、音频指纹和数量记录在 [asr_order_results.jsonl](asr_order_results.jsonl)。
原始事件日志位于 `/tmp/alpha-asr-order-run-1.log`、`-2.log`、`-3.log`，探测失败记录均保留。
实测完成后补充了“超时也打印部分 completed”的汇总逻辑，没有修改发送协议，也没有
将失败试验重跑成成功。本节未再次发送付费请求。

离线测试更新为10项通过，包括反向返回时保留真实顺序、第二次commit不等待第一段结果、
缺少第二个completed时仍汇总部分结果；类型检查通过。未修改 `qwen_alpha.py`。
