# Qwen voice + AlphaBrain

**Current dev default: Jev-driven submission.** Final transcripts are classified immediately; a complete result submits without the fixed two-second window. See the mode switch below to restore the previous behavior.

The retired standalone Qwen tool agent and LiveKit entry point have been removed. Shared Omni protocol helpers live in `voice/omni_transport.py`, and Jev routing lives in `voice/search_router.py`. It uses the existing AlphaBrain HTTP server, so HA tools, Google services, weather, web search, persistent memory, email confirmation, operation records and unknown-result handling stay in the brain. No business tools are published to Qwen.

## Start

One command from `alpha/`:

```bash
./start-alpha
```

This uses Alpha's `.venv` automatically, starts the brain, waits for `/health`, then starts Qwen. Press Ctrl+C once to stop both processes it started. A healthy existing brain is reused and is not closed by this launcher. Stop an old standalone Qwen process before using this command. Startup logs from both services share the terminal. Optional: `./start-alpha --startup-timeout 90`.

From `alpha/`, in two terminals, using **Alpha's** environment in both:

```bash
# Terminal 1; reuse an already-running brain.server rather than starting a second instance.
.venv/bin/python -m brain.server

# Terminal 2
.venv/bin/python -m voice.qwen_alpha
```

Existing `.env` settings apply: `DASHSCOPE_API_KEY`, `QWEN_WORKSPACE_ID`, `QWEN_REGION`, `QWEN_VOICE`, `REALTIME_INPUT_DEVICE` and `REALTIME_OUTPUT_DEVICE`. Brain authentication, reminder timezone and Google credentials remain unchanged. Auto memory still requires `ALPHA_AUTO_MEMORY=1`. The default brain URL is `http://127.0.0.1:8765`; override with `ALPHA_BRAIN_URL`.

## Input switch: Omni or independent ASR

The code defaults are `QWEN_ALPHA_INPUT=asr` and `QWEN_ALPHA_SPEAKER=tts`, even
without `.env` overrides. Omni input and its upload gate remain explicitly selectable:

```bash
QWEN_ALPHA_INPUT=asr ./start-alpha
# Select the original input (still using dedicated TTS by default):
QWEN_ALPHA_INPUT=omni ./start-alpha
# Select both original Omni input and Omni speaker:
QWEN_ALPHA_INPUT=omni QWEN_ALPHA_SPEAKER=omni ./start-alpha
```

ASR mode uses Silero ONNX (160ms confirmed onset, 400ms pre-roll, 608ms silence)
and `qwen3-asr-flash-realtime` in manual mode. It requires the ONNX runtime and
model installed during A2; see [setup and microphone checks](../devtools/SEGMENT_PROBE.md).
`SILERO_MODEL_PATH` can override `data/models/silero_vad.onnx`.
ASR preconnects before opening the microphone. Only completed local segments are uploaded,
one at a time; no next-segment append is sent before the previous completed/error/timeout.
The local microphone continues collecting while ASR is busy. A failed segment ends with
an empty transcript so it cannot block later inputs. Failed audio is not automatically retried.

Local speech onset ducks playback first. Silero-confirmed speech lasting 400ms pauses it;
the existing Jev addressed-to-Alpha check then either restores playback or confirms interruption.
Fifteen-second cuts keep collection open until real silence, including a very short final tail.
`TurnCollector` and AlphaBrain are unchanged. ASR mode always uses dedicated Qwen TTS
(`QWEN_TTS_VOICE`, default Ethan), even if `QWEN_ALPHA_SPEAKER=omni`; no Omni socket
is opened. The original speaker setting still applies when switching back to Omni input.
`QWEN_VAD`, `QWEN_SILENCE_MS`, and the Omni upload gate do not control ASR mode.

Independent ASR cost records use the server's `completed.usage.duration` in seconds,
including empty final transcripts. These are provider-reported usage multiplied by the
configured regional price, not actual billing receipts. Missing final usage remains pending
verification. Legacy Omni ASR reference estimates remain separate and excluded from the
known subtotal; independent ASR usage is included. No wall-clock listening estimate is
added in ASR mode.

Optional trial settings:

```dotenv
QWEN_VAD=server_vad
QWEN_SILENCE_MS=800
QWEN_ALPHA_COLLECTION_SECONDS=2
QWEN_ALPHA_CONTINUATION_SECONDS=5
QWEN_ALPHA_SUBMIT_MODE=jev
```

Jev uses the existing `JEV_PROVIDER`, `TYPESAFE_API_KEY`/`AI_GATEWAY_API_KEY`, `JEV_MODEL`, and `JEV_TIMEOUT_MS`. Without credentials, completeness falls back to local unfinished-sentence rules. It is not equivalent to a semantic detector in that configuration.

## Submission and execution

`QWEN_ALPHA_SUBMIT_MODE=jev` is the default trial mode. Jev decides as soon as the known final transcripts are ready. A complete classification submits immediately. An incomplete classification, or a clearly unfinished trailing phrase, holds the batch; after the five-second continuation period it asks once and keeps waiting for new text. Uncertainty, missing credentials and request failures fall back to the original two-second window (five for an unfinished phrase). New speech cancels either path. `执行吧` still bypasses Jev and the extra window when the request is complete.

To restore the previous mode for one run, without changing `.env`:

```bash
QWEN_ALPHA_SUBMIT_MODE=window ./start-alpha
```

Set `QWEN_ALPHA_SUBMIT_MODE=window` in `.env` for a persistent rollback. Use `QWEN_ALPHA_SUBMIT_MODE=jev ./start-alpha` to explicitly select the trial again. A complete sentence does not prove the user has finished a whole task chain: “加牛奶” may be submitted before a later “还有鸡蛋”. The `window` mode remains available for that tradeoff.

Qwen receives microphone audio continuously but automatic replies and automatic interruptions are disabled and verified against the server's returned configuration. A pause produces transcription, not permission to execute. The client waits for final transcripts for all known segments, then follows the selected submission mode. In `window` mode, the collection timer and Jev check run together and both must finish. An unfinished sentence or an `incomplete` decision retains the request; it does not execute automatically after five seconds. Uncertainty falls back to local sentence rules and the window.

Say `执行吧` or `就这些` as a separate command or a punctuation-separated ending to submit without the additional quiet window. A clearly unfinished phrase still needs completion. Example:

> 把牛奶加到购物清单，然后……明天早上八点提醒我开会，牛奶不要了，换成酱油，执行吧。

All segments, including the correction, reach AlphaBrain as one request. Every accepted batch uses one HTTP submission, and batches execute serially. New speech during execution collects into the next batch. Interrupting playback does not cancel or undo already-started business operations.

No detector can predict a correction made after a complete request and a long silence. In `window` mode, increase the collection window if you pause between complete tasks. In `jev` mode, a complete classification is enough to submit, so a later correction may become a new batch. **This trial does not promise lower end-to-end latency:** it still waits for transcription, classification and Claude, then asks Qwen to read the returned answer. It compares Qwen's audio path with OpenAI STT/TTS, rather than bypassing Claude for casual conversation.

## Playback and reconnection

Search progress is forwarded when the brain emits `我查一下。`, before the final result. Device-only `ha_done` plays a local confirmation tone. Other replies are read by Qwen from the brain's text. Actual Qwen output transcription is logged; instruction-based reading is not a guarantee of literal TTS fidelity.

VAD first ducks audio. Sustained local speech pauses it while final transcription is pending; recognized noise resumes buffered audio. With `QWEN_ALPHA_JEV_INTERRUPTIONS=1` (the default), speech that started during a report is classified before it can enter the collector or cancel playback. An explicit call to Alpha bypasses addressee classification. For other overlapping speech, one Jev request answers both addressee and completeness questions. Clear side conversation resumes the original report without clearing it. Addressed speech clears and cancels that report. The default `QWEN_ALPHA_UNCERTAIN_INTERRUPTION=hold` also resumes on uncertainty, missing credentials or service failure; it logs the unaccepted text but does not pass it to the collector or brain. Ambiguous commands may need to be repeated with “Alpha”. Idle speech skips addressee classification. Use headphones to avoid speaker echo.

The filter remembers overlap at speech start, even if the original report finishes before STT/Jev returns. Classification runs in background tasks so WebSocket audio/events continue to be received. Its completeness decision is reused only while the exact combined text still matches; stale hints are not used for modified task chains. An `addressed` choice is retained even with a low valid confidence score, so genuine corrections are not collapsed into uncertainty. `other` requires at least `max(0.85, JEV_MIN_CONFIDENCE)`; lower scores become uncertainty and follow the configured hold/accept policy. Raw choices and scores are logged before policy mapping. Scores are not a guarantee of the true addressee. Side conversation can still cause temporary ducking or pausing while deciding, then playback resumes.

The earlier policy accepted all uncertain overlapping speech, which caused unrelated statements such as “我吃过了” to cancel a report. It remains available for comparison, but is no longer the default:

```bash
QWEN_ALPHA_UNCERTAIN_INTERRUPTION=accept ./start-alpha
```

After a question has actually finished playback, a ten-second window accepts short answers such as “对”“不是”“不用了” without another addressee/completeness request. An interrupted question does not open this window. Late classification cannot cancel a different, newer report. To disable this trial filter for one run:

```bash
QWEN_ALPHA_JEV_INTERRUPTIONS=0 ./start-alpha
```

Each meaningful input advances a voice epoch. Older queued reports are discarded, and late results from older HTTP requests are retained in the brain but not added back to the speech queue. This prevents an old long report and the new short answer from both saying the same result. It does not cancel or replay business operations. Playback claims its own report before waiting, so finishing an interrupted report cannot accidentally remove the next report.

On Qwen reconnect, the running brain and serial HTTP worker remain alive. Already-submitted batches are not replayed. Unsubmitted text remains paused until new speech or `执行吧`; a reconnect prompt explains this. A report interrupted by a connection loss may be read again on reconnect; the underlying business operation is not resubmitted. API failures after submission are treated as uncertain and are not automatically retried. Qwen's own audio conversation restarts, while AlphaBrain retains its history.

## Checks

```bash
.venv/bin/python -m unittest tests.test_qwen_alpha
.venv/bin/python -m unittest discover -s tests -t .
```

Tests cover collection and corrections, pending transcription, duplicate events, explicit submission, incomplete follow-up, cancellation during Jev classification, reconnect retention, short answers, web progress, device tone, broken HTTP streams, unsolicited Qwen replies and interruption before response acknowledgement. They do not open audio devices or call real business tools.

A no-microphone protocol smoke test confirmed zero tools and manual response configuration, then received the exact test sentence `这是语音播报测试。` and six audio chunks. Microphone latency and actual home operations require a local trial. Logs report brain duration and Qwen's first-audio-packet latency; arrival at the client is earlier than physical speaker playback. Qwen usage cost is separate from Claude and Jev cost.

Protocol reference: [Alibaba Cloud Realtime client events](https://www.alibabacloud.com/help/en/model-studio/client-events).

## dev: parallel completeness checks and batch latency

The retained `window` mode on `dev` starts Jev completeness classification **at the same time** as the existing quiet-window timer, once final transcripts are available. Submission waits for both. This changes `window + Jev` into `max(window, Jev)` without shortening the configured window. New speech cancels the entire wait, including the in-flight Jev request. Explicit end commands still bypass Jev, and an incomplete request still cannot auto-execute. The newer default `jev` mode described above removes that timer only for complete classifications.

Each submitted batch gets an independent ID and captures its own speech-stopped event time, final-transcript readiness and submission time. That object follows the serial HTTP queue and all reports from the batch. Later speech cannot replace the earlier batch's timing origin. Search progress is tagged `search_progress`, final answers `reply`, and device tones `tone`; their audio latencies are not combined into one misleading measurement.

The brain receives an optional `request_id`. When present, `/ask/stream` also emits `tool_timing` events for the gateway's actual executor calls: name, operation ID, phase, duration and server-relative elapsed time. No tool arguments or results are included. Duplicate/rejected operations do not generate fake execution spans. These events cover gateway execution, not the Anthropic-hosted search internals or local memory tools. Other HTTP clients without a request ID retain the old stream format. Telemetry failures cannot change a tool's outcome.

Logs show the following stages with the same batch ID:

- Speech-stopped event → final transcripts ready → submission.
- Queue duration before the HTTP request and brain completion time.
- Tool start/end relative to the server request, with execution duration.
- Qwen response request → first audio packet at the client.
- Speech-stopped event → first playback frame, estimated from the output callback and device DAC time/latency.

Only differences measured in the same process clock are compared. Absolute `perf_counter` values are never exchanged over HTTP. “Speech stopped” here is **receipt of Qwen's speech_stopped event**, not a microphone-level measurement of the user's last sound; VAD/network delay before that event is not included. Playback starts are device estimates, not acoustic measurements or detection of the first spoken word. Audio cleared before its predicted playback time is marked not played, rather than assigned a latency of zero. Missing timestamps are reported as unknown. Pause does not produce a playback-start event until audio is actually consumed.

Restart **both** `brain.server` and `voice.qwen_alpha` after switching to `dev`, so the new tool timing events are available. No additional model request is added by latency instrumentation.

Earlier parallel classification is speculative: continuing to speak cancels that check, but the remote Jev request may already have been processed. This can increase classification request counts compared with waiting until silence ends. The timing instrumentation itself makes no model requests. The quiet-window length and unfinished-request safety rules remain unchanged.


## Dedicated TTS trial (Ethan)

The code defaults to `QWEN_ALPHA_SPEAKER=tts` and `QWEN_TTS_VOICE=Ethan`.
This uses `qwen3-tts-flash-realtime` on an independent, persistent WebSocket, with
PCM16 mono 24 kHz output fed into the existing player. Input defaults to independent
ASR; Jev and AlphaBrain are unchanged. TTS receives the exact brain text
through `input_text_buffer.append` and `commit`, without conversation instructions.
Startup preconnects TTS; interrupted synthesis closes its connection so late audio
cannot enter the next report. Errors do not automatically replay reports with Omni.
TTS usage is logged separately from Omni costs (no estimated TTS price is added).

### Qwen-Audio-TTS (`QWEN_TTS_MODEL`)

`QWEN_TTS_MODEL=qwen-audio-3.1-tts-flash` (or `-plus`) switches to `QwenAudioTTS`:
`/api-ws/v1/inference` with `run-task` → `continue-task` → `finish-task`, audio as
binary frames, one task per sentence on a kept connection, per-task usage (not
cumulative). Voices differ between models; in Singapore the 3.1 Chinese voices are
`longanlingxin_v3.1`, `longanhuan_v3.1`, `longanfengyue_v3.1` (female) and
`xunanchuan_v3.1` (male). The server is slow to answer the close handshake, so the
client uses `close_timeout=1`. `devtools/tts_compare.py` writes side-by-side WAVs.

2026-10-23 comparison (first audio after the text was sent, connection excluded):
qwen3-tts Ethan 0.53–0.59 s, 3.1 flash 0.56–0.79 s, 3.1 plus 0.71–1.13 s. Price for a
20-character sentence: qwen3-tts ≈ $0.00026, 3.1 flash ≈ $0.00005 (token-priced).

Start the trial: `./start-alpha`.
Restore the previous Omni speaker for one run:

```bash
QWEN_ALPHA_INPUT=omni QWEN_ALPHA_SPEAKER=omni ./start-alpha
```

For persistent rollback set both `QWEN_ALPHA_INPUT=omni` and `QWEN_ALPHA_SPEAKER=omni` in `.env`.
The original `QWEN_VOICE` is untouched (Zane in the local pre-trial configuration).
A non-secret snapshot of the pre-trial settings is saved locally in
`data/qwen-speaker-before-tts.json` (ignored by git).
Official voice list: https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-list

Two synthetic API samples from the development machine measured approximately
1.50 s to first packet including initial connection, then 0.33 s on the reused
connection. These are individual measurements, not latency guarantees or an audio
quality evaluation. Startup preconnection avoids the initial handshake on the first
normal reply. Example files are saved in `data/tts-preview/` for local listening.

## Unified cost ledger

Restart both brain and voice after upgrading. `[费用账本累计]` now covers Claude
(including caches and web search), memory extraction, Jev, TTS, Omni and an explicit
ASR duration reference estimate. Usage gaps and interrupted paid requests remain
marked as unpriced instead of appearing free. Data persists in `data/costs.db`.
Read it with `.venv/bin/python -m core.costs` or brain's `GET /costs` endpoint.
See [cost accounting details](../docs/COSTS.md) for verified prices and the embedded
ASR billing limitation. The earlier note that TTS is absent from the cost total is
superseded by this unified ledger; the old Omni-only counter remains separate.

## Local VAD upload trial

Local `.env` selects `QWEN_ALPHA_LOCAL_VAD=1`. The microphone still listens locally
at 16 kHz PCM16 mono. WebRTC VAD mode 3 requires 80 ms of consecutive voiced audio above a minimum
normalized RMS of 0.003. It retains 400 ms of pre-roll plus the confirmation
audio, then sends captured
frames continuously through short pauses. After 1,000 ms of non-speech it stops
uploading. Quiet capture continues locally; no audio silence keepalives are sent.
The WebSocket library's normal control-frame ping/pong remains enabled.

Qwen **server_vad** still detects utterance boundaries, Jev still decides whether
text is complete/addressed to Alpha, and AlphaBrain still handles all business
operations. The gate does not decide when to execute tasks. Original local VAD
playback pause/resume continues to observe **all** captured audio. A short answer
such as “对” now needs at least 80 ms of qualifying audio. Very soft/far-field
speech or a quieter recording device can fail this threshold; tune it using the
diagnostics rather than assuming the gate identifies human speech perfectly. TV voices and coughs can trigger upload;
Jev filtering still occurs after transcription. Use headphones for the trial.

Settings (milliseconds must be multiples of 20):

```dotenv
QWEN_ALPHA_LOCAL_VAD=1
QWEN_ALPHA_VAD_MODE=3
QWEN_ALPHA_VAD_PRE_MS=400
QWEN_ALPHA_VAD_ONSET_MS=80
QWEN_ALPHA_VAD_MIN_RMS=0.003
QWEN_ALPHA_VAD_DEBUG=1
# Default tail=max(1000, QWEN_SILENCE_MS+200), rounded up to 20ms.
# QWEN_ALPHA_VAD_TAIL_MS=1000
```

The tail must cover the server silence threshold plus at least 200 ms; invalid
settings fail before opening the microphone. Semantic VAD is not supported by
this trial because a fixed silence tail cannot guarantee semantic completion.
The local pre-trial upload settings were saved in
`data/qwen-before-upload-vad.json` (ignored by git). The speaker configuration,
including the Ethan TTS trial and Omni rollback, is unaffected.

Restore continuous upload for one run:

```bash
QWEN_ALPHA_LOCAL_VAD=0 ./start-alpha
```

Or set `QWEN_ALPHA_LOCAL_VAD=0` in `.env` for persistent rollback. Restart the voice
process after switching. Test silence, a single “对”, soft speech, and a long
instruction with pauses. `[本地VAD]` logs upload start/stop; microphone shutdown or
device restart logs captured/forwarded durations. ASR cost metering receives only
successfully uploaded bytes. This reduces uploaded silence; actual bill savings
still depend on the provider's treatment of embedded transcription.


### Quiet-room false triggers

The first trial opened on one positive WebRTC frame, which repeatedly uploaded
quiet-room noise. The revised gate uses mode 3 plus an 80 ms onset and a minimum
RMS floor. Frames below that floor also count toward the closing tail, even if
WebRTC's raw result remains speech. These are configurable starting settings,
not a guarantee for every microphone. Confirmation audio is preserved alongside
pre-roll, so the stricter onset does not intentionally discard the first syllable.
Local pre-fix settings were saved to `data/qwen-vad-before-noise-fix.json`.

With `QWEN_ALPHA_VAD_DEBUG=1`, `[VAD诊断]` prints once every ten seconds: mean/max
normalized RMS, raw VAD-positive fraction and fraction also passing the volume
threshold. No audio is stored or uploaded for diagnostics. After restarting,
remain quiet for ten seconds, then say “对” and a normal command. If it still
triggers in silence or misses words, use these measurements to adjust
`QWEN_ALPHA_VAD_MIN_RMS` and `QWEN_ALPHA_VAD_ONSET_MS`. Raise the floor cautiously:
a high floor can suppress the user too. Set `QWEN_ALPHA_VAD_DEBUG=0` after tuning.


## 切换音频设备

`REALTIME_INPUT_DEVICE` / `REALTIME_OUTPUT_DEVICE` 留空时，语音层跟随 macOS 的系统默认设备：
在系统设置或菜单栏里切换输出（例如换成 AirPods），1～2 秒内 Alpha 就改用新设备，不用重启。

PortAudio 只在初始化时读取设备列表，所以切换时要先关闭扬声器和麦克风的流，再重新初始化，
再用新设备打开（`voice/audio_devices.py`）。正在播放的那句话会被打断；ASR、TTS 和大脑的连接都不断。

- 在 `.env` 里指定了固定设备的那一边不跟随；
- 旧的 Omni 收音路径（`QWEN_ALPHA_INPUT=omni`）不跟随：它的输入流没有接入协调，
  流开着时重新初始化 PortAudio 可能让进程崩溃；
- 如果有流在 3 秒内没有关闭，这次切换放弃（日志里有"这次不切换"），需要重启语音进程。

## XVF3800 麦克风阵列（reSpeaker Flex，免戴耳机）

板子：reSpeaker Flex XVF3800 Linear-4 with XIAO ESP32S3。出厂是 I2S 固件（给 XIAO 用，Mac 识别不到），
接 Mac 要刷 USB 固件 `xmos_firmwares/usb/v1.0.4/respeaker_flex_usb_l16k2ch_v1.0.4.bin`
（仓库 https://github.com/respeaker/reSpeaker_Flex；按住主板 BOOT 上电进入安全模式，
用靠近 RST 的 USB-C 口，`dfu-util -R -e -a 1 -D <文件>`）。刷完是 16kHz、2 进 2 出的 USB 声卡
"reSpeaker Flex XVF3800 L16K2Ch"。音箱接板子的 3.5mm 口：回声消除的参考信号就是 USB 送进板子的播放内容，
声音从别的扬声器出来，板子就不知道放了什么。

`.env`：

```dotenv
REALTIME_INPUT_DEVICE=reSpeaker Flex
REALTIME_OUTPUT_DEVICE=reSpeaker Flex
REALTIME_INPUT_CHANNEL=2
```

两个输入声道（出厂路由 `AUDIO_MGR_OP_L=[8,0]`、`AUDIO_MGR_OP_R=[7,3]`），2026-10-23 用
`devtools/xvf_echo_probe.py` 实测（音箱较响，Mac 麦克风录到播放峰值约 -20 dBFS）：

| 声道 | Alpha 自己的声音 | Alpha 说话时插话 |
| --- | --- | --- |
| 1（左，会议输出） | 完全压掉 | 你的声音也被压掉，没法打断 |
| 2（右，语音识别波束） | 残留比底噪高约 15～20 dB | 完整保留：你的声音比残留回声响约 20 dB |

对照：Mac 自带麦克风里回声比你的声音还响约 5 dB。所以用第 2 声道。

注意：
- 回声消除要听约 20 秒播放才收敛（`AEC_AECCONVERGED`）。板子重新上电、**调音箱音量**、挪音箱或板子后，
  最开始几句回答的回声消不干净，Alpha 可能把自己的声音当成插话（Jev 判断会挡住大部分）。
- 回声消除没收敛时，Alpha 自己的话会被转写出来，Jev 也会判成"对 Alpha 说话"（实测三次都是）。
  所以播报期间先比对文字：转写里至少 85% 的字能在 Alpha 这一轮的回答里按顺序找到，就当作回声忽略
  （`voice/collection_rules.py` 的 `is_echo`，日志 `[回声忽略]`）。少于 3 个字不比对。
  回声尾巴：播报结束后 0.5 秒内开始的声音，也和刚说完那句话的最后 20 个字比对（允许 2 个字）。
  实测结束后 118ms 转写出"资料。"，正是最后两个字。超过 0.5 秒再说的话不比对。
  代价：播报期间插话说了几乎和 Alpha 原话一样的句子，会被忽略，等它说完再说即可。
- 音箱到麦克风的延迟实测 8 ms，`AUDIO_MGR_SYS_DELAY` 不用改。
- 板子参数可以用仓库里的 `python_control/xvf_host.py` 读写（需要 pyusb 和 libusb）；
  不执行 `SAVE_CONFIGURATION` 就不会写进 flash，重启即恢复。

## 播报期间怎样打断（`QWEN_ALPHA_INTERRUPT_MODE`）

默认 `call`：Alpha 说话期间（包括刚说完的 0.5 秒），只接受三种输入，其余一律忽略（日志 `[播报中忽略]`）：

| 说的话 | 结果 |
| --- | --- |
| “Alpha，……” / “阿尔法，……” | 停下当前播报，把这句话交给大脑（不经过 Jev） |
| “停”“停一下”“别说了”“等一下”“Alpha 停”…… | 只停下（日志 `[停止播报]`），不发给大脑 |
| Alpha 提问后的简短回答（“好”“对”“不用了”……） | 照常接收 |

原因：回声被 ASR 转写出来时常常走样（“我是安好”“你的家居就是”“二十一分了”），
文字比对拦不住，Jev 也会判成“对 Alpha 说话”。去掉名字后的部分如果有 60% 能在 Alpha 正在说的话里找到
（例如回声“Alpha 你的家居就是”），仍然当作回声。`call` 模式下听到人声不再先暂停播放，
也不再把音量压到 20%，避免回声让播报卡顿、忽大忽小；代价是喊停后要等转写出来（约半秒）才停。

名字的识别：
- 中文 ASR 常把夹在中文里的 “Alpha” 漏掉或认成“阿发”“奥特”。ASR 会话现在带上上下文
  （`input_audio_transcription.corpus.text`，官方 SDK 里叫 `corpus_text`），说明助手叫 Alpha。
  2026-10-23 用 TTS 合成的“Alpha，我的购物清单上有什么？”实测：没有上下文 3/3 认成“阿发”，有上下文 3/3 正确。
- 认错过的写法（阿发、奥特、阿尔发……，不含“奥特曼”）也算叫名字（`collection_rules.WAKE`）。
- 带名字的回声：去掉名字后有 60% 能在 Alpha 正在说的话里找到、并且至少连续 3 个字一样，才当作回声。
  只零散碰到几个字（“Alpha，现在几点了”对“现在下午五点……”）仍然是叫 Alpha。

`jev`：原来的做法，播报期间的插话由 Jev 判断是不是对 Alpha 说的。

## 识别语言（`QWEN_ASR_LANGUAGE`）

默认留空，由 ASR 自动判断语言（日志 `[ASR转写] … ｜ 语言 en/zh`）。以前固定 `language=zh`，
用户反馈真人说英文会被写成中文。2026-10-23 用 TTS 合成的句子对照：标准英文在固定 zh 时也原样输出英文，
没有复现“翻译”；不指定语言时中文结果不变，英文识别为 en。所以真人场景（口音、回声、噪音）下的原因
还要看日志里的语言字段确认。需要固定中文时填 `zh`。

## ESPHome 语音卫星（J9，`ALPHA_SATELLITE_HOST`）

设备本地听唤醒词，唤醒后才把音频推给 Alpha；Mac 不再开麦克风。

```
XVF3800（降噪/回声消除/波束）→ I2S → XIAO ESP32S3（ESPHome，microWakeWord）
   唤醒 → API 连接推 16 kHz PCM → voice/satellite.py → AsrInput.process（Silero → ASR → Jev → 大脑）
```

**设备**
- XVF3800 刷 I2S 固件 `respeaker_flex_i2s_l16k2ch_v1.0.4.bin`（和 USB 固件同样的 dfu-util 安全模式刷法；刷回
  `respeaker_flex_usb_l16k2ch_v1.0.4.bin` 就恢复 USB 声卡）。XVF3800 提供 I2S 时钟，XIAO 是 secondary。
- XIAO 刷 `esp-alpha/esphome/alpha-xvf.yaml`（引脚 WS 7、BCK 8、DIN 43、DOUT 44，与 formatBCE 的 4-Mic 配置一致）。
  左声道带 AGC、音量正常，右声道内容相同但只有约 3%，所以麦克风取左声道。
- XIAO ESP32S3 必须接上外置天线，否则 Wi-Fi 只有 -91 dBm。
- 通路先用 `esp-alpha/xvf_udp_probe`（Arduino）+ `devtools/udp_audio_receiver.py` 验证过：20 秒 2502 包、丢包 0。

**一轮的生命周期**（`SatelliteLink.phase`）
1. listening：设备听到唤醒词发 start 请求；Alpha 回端口 0（音频走 API 连接），发 RUN_START/STT_START，
   音频直接喂进断句器，后面的 ASR、Jev、大脑、TTS 都不变。设备在请求里给的 `volume_multiplier` 会照做。
2. awaiting：这句话提交给大脑，发 STT_VAD_END——设备停麦、等回答，这一轮不结束。Alpha 播报时设备不推流，听不到自己。
3. speaking：第一段要播的音频到了，发 TTS_START（要有文字）/TTS_END（要有 url）/TTS_STREAM_START，设备进入播放状态；
   `SatellitePlayer` 把 24 kHz 重采样到 16 kHz，按实时速度推（设备缓冲只有约 0.5 秒，最多领先 0.4 秒），
   播放开始/结束时间按推送进度估算。“我查一下”和最终回答在同一个流里。
4. 最终回答（reply/tone/error）播完，Receiver 调 `player.end_response()`：发 TTS_STREAM_END + RUN_END，设备放完回到空闲。
跟进窗口（`ALPHA_SATELLITE_FOLLOWUP_SECONDS`，默认 6 秒，0 关闭）：第 4 步改发 INTENT_END(continue_conversation=1)
+ TTS_STREAM_END、不发 RUN_END，设备放完自动开新的一轮（没有唤醒词）。Alpha 把紧跟在回答后 5 秒内、没有唤醒词的
start 当作跟进（看请求里有没有 USE_WAKE_WORD 标记；唤醒词名字不可靠，设备开跟进时会沿用上次的），用跟进超时：不开口就发 RUN_END、播退出提示音，设备回去听唤醒词。不发 RUN_END 是因为它晚到一步会结束掉设备新开的一轮。
其他结束：唤醒后超过 `ALPHA_SATELLITE_LISTEN_SECONDS`（默认 8 秒）没有人声（和跟进超时一样播退出提示音）；等回答超过 90 秒；设备自己停止。
设备没有 SPEAKER 能力（旧固件）时，提交就发 RUN_END，回答走 Mac 播放器。

**打断**：播报开始时设备重新启动唤醒词检测。一轮进行中又听到唤醒词，设备发事件 `esphome.alpha_barge_in`
（`api: homeassistant_services: true`），Alpha 停掉播报、结束这一轮，设备回到空闲后马上开新的一轮。
ESPHome 的 `voice_assistant.stop` 在播放阶段既不停播放也不通知服务端，所以不能靠它。

API 连接在 `main()` 里常驻，跨 ASR 重连保留；断线 5 秒后重连。

**设备配置踩过的坑**（2026-10-24，详见 `alpha-xvf.yaml` 注释）
- 扬声器链路照 formatBCE（语音助手 → 重采样 → 混音 → I2S），但混音器输入不能 `timeout: never`：语音助手要等扬声器
  自己停下才回到空闲，never 会让它播完一次就卡住。先改成 500ms，但这段就是回答播完到跟进收音开始之间的空白
  （用户说话录不进），后来改 100ms。混音器至少要两路输入（第二路留给提示音）。
- “唤醒几次后就不灵”：先后怀疑过立体声、检测器状态、I2S 帧对齐、XVF3800 状态，都不是。最后用 UDP 探针连续录音、
  离线逐句打分（pymicro-wakeword 特征 + ai-edge-litert，同一模型、同样 5 窗平均，`devtools` 外的临时脚本）查明：
  用户的 Hey Jarvis 大多压在 0.97～1.00，约 1/4 落在 0.3～0.5，默认门槛 0.97 下一半左右失败。普通对话和视频都 ≤ 0.03。
  门槛降到 0.35（11 次能过 10 次）。注意设备日志里 “Detected … sliding average” 是刚越过门槛那一刻的值，不是最高分，
  门槛设多少它就略高于多少——别拿它当发音分数。根本办法是训练自己的 “Alpha” 唤醒词（加入用户录音）。
- 声道：左声道（通话输出，带 AGC）平时唤醒和识别都好，但播报中降噪狠，打断只能偶尔成功；右声道（ASR 波束）
  播报中人声保留得多，但没有 AGC、小约 25 dB，Hey Jarvis 平时几乎不认。现在唤醒和识别都用左声道。
- 回声消除本身正常：`esp-alpha/xvf_aec_probe`（一边播测试语音一边录）+ `devtools/aec_probe_receiver.py` 实测，
  收敛约 9 秒后播放中的回声压到 -65～-70 dBFS。

**调试工具**
- `devtools/satellite_logs.py`：用 aioesphomeapi 直接订阅设备日志，逐行写文件、断线重连（esphome CLI 接管道会缓冲、会卡）。
- `ALPHA_SATELLITE_DUMP=1`：每一轮收到的原始音频存到 `data/satellite_runs/`；配合设备的“诊断收音”按钮
  （不用唤醒词直接收音），可以把设备此刻听到的声音录下来离线分析。

**启动**（XVF3800 是 I2S 模式时 Mac 看不到 USB 声卡，两个设备名要留空；回答从板载 3.5mm 接的音箱出来）：
```bash
ALPHA_SATELLITE_HOST=192.168.1.115 REALTIME_INPUT_DEVICE= REALTIME_OUTPUT_DEVICE= ./start-alpha 2>&1 | tee -i -a data/voice.log
```

API 已加密（2026-10-24）：设备 YAML 用 `!secret api_encryption_key`，同一个密钥填 `.env` 的 `ALPHA_SATELLITE_KEY`。
不带密钥连接会报 `RequiresEncryptionAPIError`。生成：`openssl rand -base64 32`（追加到 `.env` 前先确认最后一行有换行，
否则会接到上一行末尾）。

提示音：设备检测到唤醒词就在本地播上行“叮-咚”，退下时播下行“咚-叮”（`beep` 脚本，走混音器的提示音输入），不经过 Alpha。

取消词：一句话的最后一段是“退下”“没事了”“拜拜”“可以了”“再见”“结束”“结束对话”“finish”“never mind”（结尾带“吧”也算；“没事的/没事啦”按“没事了”算，ASR 常这样转）时以它为准。
跟进窗口会把这段时间说的话连成一句（“没有用啊。；退下。”），所以看最后一段，不要求整句只有取消词。
前面没内容：不交给大脑，发 RUN_END 结束这一轮（不开跟进窗口），调用设备的 `dismiss` 动作（`api.actions`）播退出提示音。
前面有请求（“把灯关了，拜拜”）：照常处理，回答完发 RUN_END、不开跟进窗口，0.6 秒后（等设备放完回答）播退出提示音。
例外：Alpha 两分钟内播完的上一句回答以问号结尾（“确认发送吗？”）时，“没事了”是在回答它，照常交给大脑。

待做：训练 "Alpha" 唤醒词（加入用户录音）、LED、观察误触发后再定门槛、播报中打断的成功率。

## 本地 Breeze TTS + 云端兜底（`BREEZE_TTS_URL`）

家里 PC（RTX 4070 Ti SUPER 16GB，WSL2 Ubuntu）上跑 Breeze TTS 2 的官方服务，Alpha 通过局域网调用；
不可用时自动用云端 TTS（`voice/breeze_tts.py`）。

**PC 端**（官方仓库 `breezeblue-ai/breeze-tts`，权重 `BreezeBlue/Breeze-TTS-2`；代码 Apache 2.0，
权重和自托管输出仅限研究/非商业用途）：
```bash
cd ~/breeze-tts && source .venv/bin/activate
python -m breeze_infer.api ~/breeze-tts-2 --host 0.0.0.0 --port 7860 --fast-backbone-decode --fast-depth-decoder
```
- 不加速时实时率约 2.2（播放会卡）；这两个分阶段加速（CUDA graph）把实时率降到 0.52～0.54、首包 0.23～0.28 秒，
  预热约 57 秒。没装 flash-attn。
- WSL2 用镜像网络（`%USERPROFILE%\.wslconfig`：`[wsl2]` `networkingMode=mirrored`），防火墙放行 7860
  （Windows 规则限 Private 网络 + Hyper-V 规则 `VMCreatorId {40E0AC32-46A5-438A-A0B2-2B479E8F2E90}`）。
  家里网络要设成 Private。镜像模式下旧的 `netsh portproxy` 规则会占住同名端口（2026-10-25：11434 被占，
  WSL 里的 Ollama 起不来），要删掉。
- 服务一次只合成一句（并发返回 409），没有身份验证，只能在局域网用。

**Alpha 端**：`BREEZE_TTS_URL`、`BREEZE_TTS_REF_AUDIO`（相对项目根目录，5～10 秒）、`BREEZE_TTS_REF_TEXT`
（参考录音一字不差的原文）。音色设计（只给文字描述）每句声音都不一样，所以用克隆：每句都带参考录音，
纯克隆模式不能发 `cfg_scale`（服务端 500）。
- 每句先试本地：连不上（连接超时 1.5 秒）、报错、409 或 3 秒内没出第一块音频，这一句改用云端，并把本地标成
  不可用，之后直接走云端，后台每 30 秒探测一次（健康检查 + 预热），恢复了切回。播到一半出错的那句算失败，
  不拼接两种声音。
- 启动时 `ready()` 先合成一句预热（启动后第一次克隆请求首包约 1.9 秒）。
- 实测：本地首包 0.24～0.41 秒；模拟 PC 关机时第一句 1.0 秒（等连接超时）后改用云端，之后 0.6 秒。
- **朗读标记**（`voice/speech_markup.py`，语音模式 VOICE_RULES 允许大脑写）：
  - 声音事件 `[笑]` `[叹气]` `[咳嗽]` `[清嗓子]`（英文 `(laugh)` `(sigh)` 等）原样交给 Breeze，会真的笑、叹气；
    句首带事件首包 0.5～0.7 秒（普通 0.2 秒）。
  - `[严肃]`：整段用音色指导模式（参考录音 + 指令“语气严肃、冷静，语速稍慢”+ `cfg_scale` 4），首包 0.29 秒、实时率 0.77。
  - 收到回答时拆开：`Report.text` 是去掉标记的文字（回声比对、打断记录、“刚问过用户”都用它），`Report.speech`
    带事件标记交给 TTS，`Report.style` 是语气。云端 TTS 去掉事件标记、忽略语气。
- **短句缓存**：不超过 12 字的句子完整合成后按（文字, 语气）缓存 64 条，再说同一句直接播（首包 0）；
  `ready()` 预先合成“我查一下。”（启动多约 0.5 秒）。预热句不进缓存，探测恢复时一定真的请求服务端。
