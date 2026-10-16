# Alpha voice pipeline

An independent LiveKit environment for microphone → Chinese transcription →
AlphaBrain → Chinese speech. Tool execution and conversation history stay in
`brain.server`; this process sends only the latest user utterance to `/ask/stream`.

Implemented against the installed LiveKit Agents and plugin versions **1.8.8**.

## Run

From `alpha/`, start the brain:

```bash
.venv/bin/python -m brain.server
```

In another terminal:

```bash
cd alpha/voice_pipeline  # or cd voice_pipeline if already in alpha/
.venv/bin/python agent.py download-files  # once; models are cached
.venv/bin/python agent.py console
```

The voice process loads `voice_pipeline/.env`, then fills missing settings from
`alpha/.env`. Set `OPENAI_API_KEY` in either location. Console mode uses local
audio devices and does not require a LiveKit Cloud room.

Defaults:

- Silero VAD with 100 ms minimum speech, including short replies such as “对”.
- Local **MultilingualModel** turn detection, supporting Chinese.
- OpenAI `gpt-4o-mini-transcribe`, with `language="zh"` and realtime audio upload.
  Its transcription VAD uses 350 ms silence; Silero and the multilingual model
  still decide when to send a completed user turn to the brain. Set
  `ALPHA_STT_REALTIME=0` to compare with the previous batch-upload STT.
- OpenAI `gpt-4o-mini-tts`, voice `coral`, speed **1.15**. Optional environment
  variables: `ALPHA_TTS_VOICE` and `ALPHA_TTS_SPEED`. Audio is returned as PCM;
  the sentence adapter allows short Chinese sentences (4-character minimum,
  2-character lookahead) instead of waiting for a larger text buffer.
- Custom `llm_node` → `http://127.0.0.1:8765/ask/stream`. No provider LLM is instantiated
  in this process. `BrainNodeLLM` is a local marker required by 1.8.8 to enable
  automatic replies; its `chat()` is deliberately unusable.

Preemptive generation is disabled: the brain must not execute tools while a user
turn is still tentative. Failed POST requests are not automatically retried.
Interrupting speech does not roll back a tool already executed by the brain.

By default, a completed LiveKit turn goes straight to the brain, without an
additional collection timer or rule-based five-second prompt. Turn boundaries
are handled by `MultilingualModel` with endpointing `min_delay=0.5` and
`max_delay=3.0`. This removes the extra two seconds; it does not guarantee that
every unfinished sentence will be recognized correctly. Tools still must not
run speculatively (`preemptive_generation` stays disabled).

For A/B testing only, set `ALPHA_COLLECTION_ENABLED=1` to restore the extra
two-second collection window (`ALPHA_COLLECTION_SECONDS=2`). It is disabled
even if old duration settings remain in `.env`. In this optional mode, further speech
pauses submission; meaningful segments are combined into one request. The
window restarts after the latest transcript/segment is collected, so it adds
time beyond STT and LiveKit turn detection. Standalone “执行” (or a separated
sentence-ending “，执行”) submits the current collection immediately after
speech/transcription completes. Ordinary phrases such as “提醒我执行” are not
treated as the control command.

An optional **Jev search router** can replace the keyword heuristic. Configure
a Vercel AI Gateway key or TypeSafe official API key in `alpha/.env` (or the overriding
`voice_pipeline/.env`):

```dotenv
AI_GATEWAY_API_KEY=your_vercel_gateway_key
JEV_PROVIDER=vercel
# Optional experiment settings:
ALPHA_SEARCH_ROUTER=jev
# JEV_MODEL=typesafe-ai/jev
JEV_TIMEOUT_MS=300
JEV_MIN_CONFIDENCE=0.8
```

Restart the voice process after changing these values. With a key present,
`jev` is the default; otherwise the default stays `rules`. Explicit modes:
`jev` uses the classifier, `rules` uses the existing heuristic, and `off` skips
early classification/prompting. The brain can still independently decide to
search and emit its search-start prompt in every mode.

With `JEV_PROVIDER=vercel`, the router uses the [Vercel TypeSafe-compatible API](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe)
at `https://ai-gateway.vercel.sh/typesafe/v1/systemone`, with model `typesafe-ai/jev`.
For direct TypeSafe access, use `JEV_PROVIDER=typesafe`, `TYPESAFE_API_KEY` and model
`jev-latest`. Without an explicit provider, a nonempty `AI_GATEWAY_API_KEY` selects
Vercel; otherwise TypeSafe is used. Keys are not shared between providers.

The router calls the [official TypeSafe API](https://docs.typesafe.ai/api) with
one Choice question (`web` / `direct` / `uncertain`). It sends the current
transcript and at most six recent user/assistant messages, capped at 800
characters each, together with the distinction between public web search and
HA, personal-list/calendar and home-weather tools. It does not call business
tools, modify Claude's prompt, or restrict Claude's tools. Only a `web`
decision meeting the confidence threshold triggers “我查一下。” locally.
Low confidence, uncertainty, missing credentials, HTTP/format errors or timeouts
go directly to the brain, without retries or a second keyword-rule decision.
Cancellation propagates and does not start an outdated brain request.

`JEV_TIMEOUT_MS` is a total classification budget, including connection and
reading the response. This extra stage can add up to the budget to each turn;
300 ms is an initial experiment setting, not a measured service latency.
`[Jev] route=... confidence=... ｜ ...ms` records decision timing without raw
prompts or keys. Measure Chinese queries and contextual followups on your
network before choosing a different budget or threshold. No new SDK dependency
is needed; the existing async HTTP client is reused.

A small Vercel probe on this setup classified four Chinese synthetic examples
correctly: restaurant search, contextual cuisine followup, a light command and
home weather. The first request took 1.40 s; subsequent requests took 287–310 ms.
These are a small sample, not a latency guarantee. The local `.env` budget is
600 ms to allow more room than the initial 300 ms experiment; a slow first
connection can still time out and fall through to Claude without an early prompt.

After LiveKit commits the turn (and any optional collection), web requests
(“帮我搜一下”, “上网查”, “帮我查一下…”, “最新新闻”)
first emit “我查一下。” locally, before the brain POST. Local list/device
queries and explicitly negated search requests do not trigger this heuristic.
The prompt is synthesized as a separate finite TTS segment, accepting both the
complete text and the version without its final period produced by LiveKit's
default Markdown filter. The filter's delayed period is not read twice. This
prevents either text filtering or sentence tokenization from holding the prompt
until the final answer arrives. The brain POST and prompt synthesis run in
parallel: the POST does not wait for a TTS first packet, `speaking`, or playback
completion. The TTS node still finishes yielding the prompt's audio before
yielding answer audio, preserving playback order without adding TTS latency
to search latency. These guarantees concern the audio output pipeline, not
an acoustic measurement of the physical speaker.
The request carries `search_announced=true` to suppress the brain's search-start
prompt. For searches missed by this narrow heuristic, the brain still prompts
on its first actual search-start event. Misclassification is possible; this
heuristic announces progress, not proof that a search actually succeeded.
The immediately following search-related question (for example “台湾的呢” or
“有没有别家”) can also trigger the local prompt. An unrelated or local-tool turn
clears this context. Local list/device/weather queries and cancelled searches
do not get a search prompt. This remains a heuristic, not full intent parsing.
When this turn has nonempty successful web-search results, the final spoken
answer starts with “查到了，…”. It is not added for search errors, empty results
or answers explicitly saying no matching information was found; an existing
“查到了” prefix is not repeated.

Only in the optional collection mode, obvious unfinished endings (for example “然后……”, “还有”, “再帮我”, or
“把牛奶” without an action) use a five-second continuation window instead
(`ALPHA_CONTINUATION_SECONDS=5`). Connectors inside complete instructions do
not extend the window. After that window, the voice layer asks once:
“你还没说完，我在听。接下来要做什么？” No brain request is sent. Unfinished
tasks stay in memory until continued or the voice process exits; silence does
not execute them and there is no repeated prompting. “就这些” / “没了” also
submit complete collections immediately, but cannot override a known
unfinished ending. These narrow text rules do not guarantee semantic
completeness; missing tool parameters still require validation by the brain.

In that optional mode, cancelling a waiting LiveKit generation preserves unsubmitted text for the next
turn. Noise transcripts are not appended. If a new speech segment has no final
transcript yet, pending tasks remain held rather than executing prematurely.
Once the POST starts, a later correction cannot undo a tool already dispatched.
During playback, detected speech lowers outgoing audio to 20% while the report
keeps playing. The `stt_node` waits for a final transcript before deciding:

- Empty transcripts and explicit noise/cough markers are dropped **before**
  LiveKit receives them. Volume returns to normal; the report is not cancelled.
- Normal text, including single-character replies such as “对” and “停”, triggers
  an explicit interruption, then enters the normal user-turn pipeline.
- If no transcript arrives within one second after speech stops, volume returns
  to normal. A late meaningful final transcript can still interrupt.

Automatic audio-only interruption is disabled, and incoming audio is retained
while Alpha speaks. Interim transcripts cannot cancel playback. The gate does
not require half a second of speech or a minimum word count; Silero's 100 ms
speech detection still applies. Output already queued at the audio device can
take a short time to reflect a volume change.

This is a conservative text filter, not a separate semantic model. It preserves
ambiguous acknowledgments such as “嗯” and cannot identify a cough that ASR
mistakenly transcribes as an ordinary word. The playback wrapper uses LiveKit's
public audio-output interface; it does not change system-wide speaker volume.

## Streaming and latency

Restart **both** the brain and voice processes after updating. The existing JSON
`/ask` endpoint remains available; the voice process now consumes NDJSON events
(`delta`, `signal`, `done`, `error`) from `/ask/stream` and forwards text increments to TTS.
An interrupted or truncated response is not retried automatically. A disconnected
voice client does not cancel an already-started brain/tool request.

Claude uses a private `alpha_voice_reply` output tool to distinguish final speech
from intermediate tool decisions. Only its validated answer is sent to the speaker;
tool-call preambles are suppressed. This output format is not an MCP tool and is
not published in the business-tool catalog or saved as a tool call in history.
Claude may propose multiple business tools in a single response. The brain
checks the complete response before speaking, executes those tools in their
returned order through the gateway, and sends all results back together.
This batches model decisions; it does not execute side effects concurrently.
If an early final-answer block appears alongside business calls, it is not
spoken. Its call ID receives an explicit rejected-summary result so the model
can regenerate a final answer after the business tools finish.

The brain continues until Claude emits a response with no remaining business
calls. Three independent tasks can now use one batch decision plus one final
response, rather than three individual decisions plus a final response.
Dependent tasks can still need further rounds. The former device fast exit
remains removed because one successful tool does not mean the turn is complete.
For correctness, answer text now waits for response validation instead of being
spoken token by token while that same response may still contain later tools.
The HTTP transport and TTS audio remain streaming, with short-sentence playback.

Turns containing only successful HA device actions (on/off/brightness) use a
local 120 ms confirmation tone instead of spoken results. HA status queries
must speak their answers; any turn with a query or web search is not muted.
One tone is played only after the complete device-action-only turn succeeds.
Failures and unknown outcomes are still explained verbally;
personal-tool results continue to use speech. The tone is delivered as a
`signal` event with `name="ha_done"` and played through LiveKit's existing audio
queue, without a TTS request or an extra conversational message.

Spoken task reports default to one short sentence grouped by category, for
example “购物清单、日历已更新，家居操作成功。” They omit item-by-item recaps, times,
device names and conversational filler unless the user asks to verify details.
Failures, skipped operations and unknown outcomes must remain explicit, for
example “购物清单已更新，日历失败。” Queries still answer the actual question.

Logs now include STT/turn-completion timing, first brain text, complete brain
answer, TTS first-audio timing, and speech-end-to-playout latency. First text is
not the same measurement as first audible audio.

Earlier component checks, before the batch-tool response validation change,
using synthetic audio and no microphone:

- PCM TTS for “对”: first audio in 1.486 s.
- Realtime STT recognized “對” about 0.458 s after the synthetic speech ended.
- Claude streaming greeting: first text at 1.261 s, complete response at 1.268 s.

These are component checks, **not** a new full-conversation latency measurement.
The microphone measurements below were taken before these optimizations.

## Devices and measurement

```bash
.venv/bin/python agent.py console --list-devices
.venv/bin/python agent.py console --input-device "microphone name" --output-device "speaker name"
```

Use headphones during testing. Speak “把客厅灯关了”, wait for its answer, then
say “对” on its own. Check both the transcript and the actual device state.

After each spoken answer, `[延迟] 说完 → 首帧播放` logs LiveKit's `e2e_latency`:
VAD-estimated speech end through turn detection, STT, the brain, TTS, and first
audio playout. This is a software measurement, not an acoustic measurement of
the first audible syllable. `[大脑]` separately reports the HTTP request duration.

Live microphone check on 2026-10-15:

| Utterance | Observed result | Speech end → first playout |
| --- | --- | --- |
| 把客厅灯关了 | HA `intent__HassTurnOff` returned `action_done`; Alpha answered “已关闭 Living Room。” | 4.16 s |
| 对 | Transcribed as “對”; Alpha answered verbally | 3.46 s |

Brain HTTP time for these two turns was 1.99 s and 1.15 s respectively. These
are measurements from this run, not latency guarantees.

This release still supports `agent.py console` and the local multilingual turn
detector, but emits deprecation notices for both. They are intentionally used to
match this project's requested command and local multilingual model.
During the live test, this version's legacy console also printed sounddevice
`Event loop is closed` callbacks during Ctrl+C shutdown. The process exited and
the dialogue checks passed; the warning originates in LiveKit's legacy console
audio cleanup, not in the brain HTTP adapter.

## Tests

Run inside `voice_pipeline/` with its own environment:

```bash
.venv/bin/python -m unittest -v test_agent
```

References: [pipeline nodes](https://docs.livekit.io/agents/logic/nodes/),
[turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector/),
[console mode](https://docs.livekit.io/agents/server/startup-modes/),
[OpenAI realtime transcription](https://developers.openai.com/api/docs/guides/realtime-transcription),
[OpenAI TTS](https://developers.openai.com/api/docs/guides/text-to-speech).

### Jev interruption filtering

With a Jev key configured, interruption filtering is enabled by default. Optional
settings in `alpha/.env` or `voice_pipeline/.env`:

```dotenv
ALPHA_JEV_INTERRUPTIONS=1
ALPHA_ANSWER_WINDOW_SECONDS=10
```

During playback, VAD only lowers the volume; it does not cancel the report.
The final transcript is evaluated before it reaches LiveKit's conversation or
AlphaBrain. An explicit leading `Alpha` / `阿尔法` is accepted locally. Otherwise,
Jev receives the new text, the current generated reply (at most 2,000 characters),
and whether a question-answer window is open. Only an `addressed` decision with
confidence at least `max(0.9, JEV_MIN_CONFIDENCE)` interrupts. Side conversation,
backchanneling, uncertainty, HTTP failures and timeouts leave the report running.
Noise markers are still rejected locally without any model call.

After an uninterrupted reply ending in a question finishes playback, a 10-second
window accepts short answers such as `对` or `不要了` without a model call. Speech
that started before the question finished does not get this shortcut. The next
accepted input consumes the window. The installed LiveKit version emits the
assistant `conversation_item_added` event after playback; interrupted replies do
not open a window.

This is transcript-based addressing detection, not speaker identification or an
acoustic wake-word engine. Outside playback, normal requests are still accepted
as before. It cannot reliably know whom you are looking at. Set
`ALPHA_JEV_INTERRUPTIONS=0` to restore the previous meaningful-transcript policy.
It reuses the Jev endpoint, credentials and total timeout budget configured above.

A synthetic Vercel probe preserved playback for third-party chat and backchannel
examples (both below the confidence threshold), and accepted an explicit task
correction and a direct question answer. Requests took 221–559 ms. Real microphone
and household conversation accuracy still need evaluation.

### Experimental Jev end-of-turn detector

The local experiment is enabled with:

```dotenv
ALPHA_TURN_DETECTOR=jev
ALPHA_TURN_MAX_DELAY=5
```

Restart the voice process to apply it. Set `ALPHA_TURN_DETECTOR=livekit` to return
to the original multilingual detector (its default maximum delay is 3 seconds;
remove `ALPHA_TURN_MAX_DELAY` or set it to `3` when comparing defaults).

`JevTurnDetector` implements the installed LiveKit 1.8.8 turn detector protocol.
It replaces the multilingual predictor; it does not add a second collection
window. The old `ALPHA_COLLECTION_ENABLED` window is disabled in this mode.
VAD/STT still detect pauses and provide text. Jev sees the current accumulated
user turn and up to six recent messages. A high-confidence complete result uses
LiveKit's 0.5-second minimum silence; an incomplete result uses its configured
maximum silence. These are measured from the framework's last speaking anchor,
not an additional delay after Jev. The STT and network latency still apply.

Uncertain, low-confidence, HTTP-error or timed-out results fall back to the
original local multilingual model, interpreted using its own threshold. If the
fallback fails, the long silence limit is used. User speech cancels the old
prediction through LiveKit; later transcripts are merged by the framework.
Preemptive tool execution stays disabled.

The adapter's logged probability is a 0/1 control value, not calibrated model
confidence. The `[Jev] turn_complete` log shows actual classifier confidence and
request time. `[轮次判断]` shows which decision/fallback was used.

A five-example synthetic probe classified `关客厅灯` and a complete reminder as
complete, `把牛奶` and a request ending in `还有` as incomplete, and `对` after a
question as complete. The first request took 981 ms; subsequent requests took
267–449 ms. The diagnostic probe used a five-second timeout; normal requests
retain the local 600 ms budget, so a slow first call can fall back.

This predicts semantic completeness, not future intent. A complete `加牛奶` may
still be followed by `还有鸡蛋` after a long pause. The maximum silence remains a
hard framework limit: at five seconds it submits the text even if Jev called it
incomplete; it does not wait indefinitely or guarantee that partial tasks cannot
execute. Compare real short commands and a compound request with a one-second
pause before deciding whether to keep this mode.
