# Qwen voice + AlphaBrain trial

**Current dev default: Jev-driven submission.** Final transcripts are classified immediately; a complete result submits without the fixed two-second window. See the mode switch below to restore the previous behavior.

This entry point is separate from `voice.qwen_realtime` and LiveKit. It uses the existing AlphaBrain HTTP server, so HA tools, Google services, weather, web search, persistent memory, email confirmation, operation records and unknown-result handling stay in the brain. No business tools are published to Qwen.

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

The local `.env` now selects `QWEN_ALPHA_SPEAKER=tts` and `QWEN_TTS_VOICE=Ethan`.
This uses `qwen3-tts-flash-realtime` on an independent, persistent WebSocket, with
PCM16 mono 24 kHz output fed into the existing player. Omni still provides input
transcription; Jev and AlphaBrain are unchanged. TTS receives the exact brain text
through `input_text_buffer.append` and `commit`, without conversation instructions.
Startup preconnects TTS; interrupted synthesis closes its connection so late audio
cannot enter the next report. Errors do not automatically replay reports with Omni.
TTS usage is logged separately from Omni costs (no estimated TTS price is added).

Start the trial: `./start-qwen`.
Restore the previous Omni speaker for one run:

```bash
QWEN_ALPHA_SPEAKER=omni ./start-qwen
```

For persistent rollback set `QWEN_ALPHA_SPEAKER=omni` in `.env`.
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
