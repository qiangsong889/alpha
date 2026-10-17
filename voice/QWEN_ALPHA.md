# Qwen voice + AlphaBrain trial

This entry point is separate from `voice.qwen_realtime` and LiveKit. It uses the existing AlphaBrain HTTP server, so HA tools, Google services, weather, web search, persistent memory, email confirmation, operation records and unknown-result handling stay in the brain. No business tools are published to Qwen.

## Start

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
```

Jev uses the existing `JEV_PROVIDER`, `TYPESAFE_API_KEY`/`AI_GATEWAY_API_KEY`, `JEV_MODEL`, and `JEV_TIMEOUT_MS`. Without credentials, completeness falls back to local unfinished-sentence rules. It is not equivalent to a semantic detector in that configuration.

## Submission and execution

Qwen receives microphone audio continuously but automatic replies and automatic interruptions are disabled and verified against the server's returned configuration. A pause produces transcription, not permission to execute. The client waits for final transcripts for all known segments, then for the collection window. Jev checks the combined text. An unfinished sentence or an `incomplete` decision triggers one follow-up question and retains the request; it does not execute automatically after five seconds. Uncertainty falls back to local sentence rules.

Say `执行吧` or `就这些` as a separate command or a punctuation-separated ending to submit without the additional quiet window. A clearly unfinished phrase still needs completion. Example:

> 把牛奶加到购物清单，然后……明天早上八点提醒我开会，牛奶不要了，换成酱油，执行吧。

All segments, including the correction, reach AlphaBrain as one request. Every accepted batch uses one HTTP submission, and batches execute serially. New speech during execution collects into the next batch. Interrupting playback does not cancel or undo already-started business operations.

No detector can predict a correction made after a complete request and a long silence. For an explicit boundary, finish a long chain with `执行吧`; for pauses longer than the configured two seconds, increase the collection window. Lowering it can reduce delay but admits earlier submission. **This trial does not promise lower end-to-end latency:** it still waits for collection and Claude, then asks Qwen to read the returned answer. It compares Qwen's audio path with OpenAI STT/TTS, rather than bypassing Claude for casual conversation.

## Playback and reconnection

Search progress is forwarded when the brain emits `我查一下。`, before the final result. Device-only `ha_done` plays a local confirmation tone. Other replies are read by Qwen from the brain's text. Actual Qwen output transcription is logged; instruction-based reading is not a guarantee of literal TTS fidelity.

VAD first ducks audio. Sustained local speech pauses it; meaningful final text clears playback and cancels generation. Recognized noise resumes buffered audio. This trial accepts meaningful interruptions without the LiveKit-side Jev addressee filter, so side conversations may be accepted. Use headphones to avoid speaker echo.

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

The `dev` branch starts Jev completeness classification **at the same time** as the existing quiet-window timer, once final transcripts are available. Submission still waits for both. This changes `window + Jev` into `max(window, Jev)` without shortening the configured window. New speech cancels the entire wait, including the in-flight Jev request. Explicit end commands still bypass Jev, and an incomplete request still cannot auto-execute.

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
