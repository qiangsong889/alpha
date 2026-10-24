# Qwen voice + AlphaBrain

**Current dev default: Jev-driven submission.** Final transcripts are classified immediately; a complete result submits without the fixed two-second window. See the mode switch below to restore the previous behavior.

The retired standalone Qwen tool agent and LiveKit entry point have been removed. Shared Omni protocol helpers live in `voice/omni_transport.py`, and Jev routing lives in `voice/search_router.py`. It uses the existing AlphaBrain HTTP server, so HA tools, Google services, weather, web search, persistent memory, email confirmation, operation records and unknown-result handling stay in the brain. No business tools are published to Qwen.

## Start

One command from `alpha/`:

```bash
./start-qwen
```

This uses Alpha's `.venv` automatically, starts the brain, waits for `/health`, then starts Qwen. Press Ctrl+C once to stop both processes it started. A healthy existing brain is reused and is not closed by this launcher. Stop an old standalone Qwen process before using this command. Startup logs from both services share the terminal. Optional: `./start-qwen --startup-timeout 90`.

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
QWEN_ALPHA_INPUT=asr ./start-qwen
# Select the original input (still using dedicated TTS by default):
QWEN_ALPHA_INPUT=omni ./start-qwen
# Select both original Omni input and Omni speaker:
QWEN_ALPHA_INPUT=omni QWEN_ALPHA_SPEAKER=omni ./start-qwen
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
QWEN_ALPHA_SUBMIT_MODE=window ./start-qwen
```

Set `QWEN_ALPHA_SUBMIT_MODE=window` in `.env` for a persistent rollback. Use `QWEN_ALPHA_SUBMIT_MODE=jev ./start-qwen` to explicitly select the trial again. A complete sentence does not prove the user has finished a whole task chain: “加牛奶” may be submitted before a later “还有鸡蛋”. The `window` mode remains available for that tradeoff.

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
QWEN_ALPHA_UNCERTAIN_INTERRUPTION=accept ./start-qwen
```

After a question has actually finished playback, a ten-second window accepts short answers such as “对”“不是”“不用了” without another addressee/completeness request. An interrupted question does not open this window. Late classification cannot cancel a different, newer report. To disable this trial filter for one run:

```bash
QWEN_ALPHA_JEV_INTERRUPTIONS=0 ./start-qwen
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

Start the trial: `./start-qwen`.
Restore the previous Omni speaker for one run:

```bash
QWEN_ALPHA_INPUT=omni QWEN_ALPHA_SPEAKER=omni ./start-qwen
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
QWEN_ALPHA_LOCAL_VAD=0 ./start-qwen
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
  （`voice/collection_rules.py` 的 `is_echo`，日志 `[回声忽略]`）。少于 3 个字不比对；Alpha 说完后再说的话不比对。
  代价：播报期间插话说了几乎和 Alpha 原话一样的句子，会被忽略，等它说完再说即可。
- 音箱到麦克风的延迟实测 8 ms，`AUDIO_MGR_SYS_DELAY` 不用改。
- 板子参数可以用仓库里的 `python_control/xvf_host.py` 读写（需要 pyusb 和 libusb）；
  不执行 `SAVE_CONFIGURATION` 就不会写进 flash，重启即恢复。
