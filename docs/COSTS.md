# Runtime cost accounting

Enabled by `brain.server` and `voice.qwen_alpha`. Both processes use the same
`data/costs.db` SQLite ledger (ignored by git). Restart **both** processes after
updating, including an already-running brain that `start-qwen` might reuse.

Logs show per-request estimates and `[费用账本累计]` across all recorded runs.
They do not include historical usage from before this feature was installed.

Read the persisted ledger without making API requests:

```bash
.venv/bin/python -m core.costs
```

Or query the running brain's `GET http://127.0.0.1:8765/costs` endpoint. Each
component has `usd` (decimal string), `events`, and `unpriced` (records requiring
reconciliation). ASR's reference estimate is explicitly excluded from the known
subtotal. This is not an invoice: credits, free quotas, negotiated prices, tax,
SDK retries whose usage is not returned, and supplier-side rounding can differ.

## Recorded components

| Component | Metering |
| --- | --- |
| Claude | Every main-model request, streaming or non-streaming: uncached input, output, cache reads, 5-minute and 1-hour cache writes, and web search requests |
| Memory | Background Haiku extraction requests, separately from main Claude requests |
| Jev | Every classifier request, including cancelled/time-out requests; a combined two-question call is counted once |
| TTS | Server-reported billed `usage.characters`, not Python string length; audio tokens are not charged again |
| Omni | All four server-reported input/output text/audio token counts for the original speaker mode |
| ASR | Actual successfully sent PCM16 mono 16 kHz bytes divided by 32,000; includes silence, flushes every 10 seconds and on normal disconnect/exit |

For Vercel Jev, `provider_metadata.gateway.cost` takes precedence over estimated
token pricing; we verified the actual response contains this USD amount.
Otherwise Jev uses reported input tokens at $0.042 / million, configurable via
`JEV_INPUT_USD_PER_MTOK`. Returned output tokens are not charged for Jev.

### ASR limitation

Omni's embedded `input_audio_transcription` does not give this code an
authoritative billed-duration receipt. We preserve the sent duration and provide
a **reference** using the standalone `qwen3-asr-flash-realtime` duration rate:
Singapore $0.000090/s, Beijing $0.000047/s. It is marked
`embedded_asr_billing_unverified`, not treated as a confirmed separate charge,
and not added to the known subtotal. Do not infer that Omni audio-input token
charges apply when no Omni response is generated; do not add both blindly.
Supplier bills are needed to reconcile this embedded mode. Forced process kill
can lose the last less-than-ten-second audio checkpoint; an in-flight send whose
acknowledgement is lost cannot be measured exactly by the client.

Missing or malformed usage, unrecognized model prices, interrupted synthesis,
and incomplete model streams are **unpriced**, not zero/free. An attempt is
persisted before the paid request so cancellation cannot erase it; completion
updates the same row. ASR checkpoints and duplicate Omni `response.done` events
also update the same event instead of adding charges twice. Failed TTS connection
setup before sending text is not registered as a synthesis request.

Web search is $0.01 per reported search plus token costs. Web fetch has no separate
request fee. If a search is seen but its request usage is absent, token costs can
be subtotalled while the missing search fee is flagged.

## Price snapshot (USD, checked 2026-10-17)

| Model | Input / million | Output / million | Cache read / million |
| --- | --- | --- | --- |
| Claude Sonnet 5 | 2 | 10 | 0.20 |
| Claude Sonnet 5.5 | 2 | 10 | 0.10 |
| Claude Sonnet 4.5 / 4.6 | 3 | 15 | 0.30 |
| Claude Haiku 5.5, input up to 100k | 0.10 | 0.50 | 0.01 |
| Claude Haiku 5.5, input above 100k | 0.50 | 2.50 | 0.05 |
| Claude Haiku 4.5 | 1 | 5 | 0.10 |

Claude cache writes: 1.25x input price for five minutes, 2x for one hour.
Unsupported models are flagged, not silently priced as Sonnet.
Qwen3 TTS Flash Realtime: Singapore $0.13 / 10,000 characters;
Beijing $0.143353 / 10,000 characters.
Omni Singapore input text/audio and output text/audio: $0.23 / $0.93 / $0.70 /
$1.87 per million tokens; Beijing $0.212 / $0.848 / $0.636 / $1.696.

Official references:

- https://platform.claude.com/docs/en/about-claude/pricing
- https://www.alibabacloud.com/help/en/model-studio/model-pricing
- https://www.alibabacloud.com/help/en/model-studio/qwen3-asr-flash-realtime
- https://www.alibabacloud.com/help/en/model-studio/qwen3-8-omni-flash-realtime
- https://vercel.com/ai-gateway/models/jev
- https://typesafe.ai/blog/introducing-system-one-models-and-jev

The current instrumentation covers the Qwen + AlphaBrain route. Local device
playback and local VAD add no model API charges. Existing LiveKit OpenAI STT/TTS
and standalone eval scripts are not included in this runtime ledger. No billing
integration makes extra provider requests or changes tools, prompts, playback,
confirmation rules, or the saved Omni/TTS switch. The ledger stores only model,
component, numeric usage, amount, timestamp and missing-usage reason; never keys,
user messages, transcripts, tool arguments or results.

With `QWEN_ALPHA_LOCAL_VAD=1`, the microphone gate suppresses idle audio before
sending. ASR duration includes only successfully sent speech, pre-roll and tail,
not all locally captured audio. Consequently idle listening no longer grows the
local duration estimate. The same embedded-ASR billing uncertainty still applies.
