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

After LiveKit commits the turn (and any optional collection), explicit web requests (“帮我搜一下”, “上网查”, “最新新闻”)
first emit “好的，我查一查。” locally, before the brain POST. Local list/device
queries and explicitly negated search requests do not trigger this heuristic.
The prompt is synthesized as a separate finite TTS segment so the sentence
tokenizer cannot hold it until the final answer arrives. Synthesis/playback and
the brain request can overlap; the POST does not wait for playback completion.
The request carries `search_announced=true` to suppress the brain's search-start
prompt. For searches missed by this narrow heuristic, the brain still prompts
on its first actual search-start event. Misclassification is possible; this
heuristic announces progress, not proof that a search actually succeeded.

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
