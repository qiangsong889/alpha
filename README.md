# Alpha

Alpha is a personal AI assistant under development. Its goal is to control home devices through natural language and connect personal services, such as tasks and calendars, through MCP. The project also provides hands-on practice with asynchronous Python, model tool calling, MCP, and OAuth.

The current implementation includes a command-line agent connected to both Home Assistant and a local Google Tasks MCP server. Calendar support includes authorization, a basic read check, and reminder creation.

## Current Features

### Home Assistant Agent

- Calls Claude using `AsyncAnthropic` and responds briefly in Chinese.
- Connects to Home Assistant's `/api/mcp` endpoint over Streamable HTTP and initializes an MCP session.
- Discovers tools from both sessions, rejects duplicate tool names at startup, filters tools through an allowlist, and converts their schemas to Claude's tool format.
- Routes tool calls to the owning session. The HTTP client remains open for the process lifetime; personal tools are cached after a short discovery connection, and a background `PersonalWorker` holds the personal stdio subprocess/session and serializes calls through a queue; Home Assistant tools and the Assist prompt are cached after an initial discovery connection. Each HA tool call creates, initializes, and closes its own transport and session.
- Reports per-turn connection failures, including grouped background-task errors, and returns to the input prompt without automatically replaying operations.
- Combines Alpha's rules with the server's `Assist` prompt and adds a prompt caching marker.
- Executes tool calls and returns their text and error status to the model.
- Keeps conversation history during the current process, with up to 10 model requests per turn.
- Prints each tool's name, arguments, and the first 100 characters of its result for debugging.

The current allowlist is shown below. Available tools depend on what the Home Assistant server actually exposes.

| Tool | Purpose |
| --- | --- |
| `homeassistant__GetLiveContext` | Retrieve live home context |
| `intent__HassTurnOn` | Turn devices on |
| `intent__HassTurnOff` | Turn devices off |
| `light__HassLightSet` | Adjust lighting |
| `list_items` | Read a Google Tasks list |
| `add_items` | Add tasks to a Google Tasks list |
| `create_reminder` | Create a primary-calendar event with a popup reminder |

### Google Personal Services

- Browser-based OAuth authorization, saved credentials in `token.json`, and automatic refresh of expired credentials.
- Google Tasks API verification: list task list names and IDs.
- Google Calendar API verification: retrieve up to three upcoming events from the primary calendar.
- A local stdio MCP server exposing Google Tasks operations.

| MCP Tool | Input | Output |
| --- | --- | --- |
| `list_items` | `list_name` | Task titles and completion status |
| `add_items` | `list_name`, `items` | A text confirmation of added tasks |
| `create_reminder` | `title`, `local_time`, optional `minutes_before` | Confirmation including the scheduled time and reminder timing |

Task list lookup and task retrieval both handle pagination. If a list cannot be found, the error includes available list names. If a batch insertion fails partway through, the error identifies tasks already added. Successful insertions are not rolled back.

## Communication Structure

```text
User → Alpha CLI → Claude
           │         │
           │ ← Tool call decision
           ├─ session_ha → Home Assistant MCP (Streamable HTTP)
           └─ session_personal → servers/personal_server.py (stdio)
                                         ↓ Google API
                                     Google Tasks
```

The model decides which tools to call; the Python client executes the MCP requests. Stdio connects local processes through standard input and output. The client normally launches the server process, so no HTTP listening port is required.

## Installation

Run the following commands from the `alpha` directory. The current development environment uses Python 3.11 and MCP SDK 2.3.0. This SDK version uses `MCPServer` rather than the older `FastMCP` class.

```bash
cd alpha
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'mcp==2.3.0' anthropic httpx2 python-dotenv \
  google-auth google-auth-oauthlib google-api-python-client OpenCC
```

The project does not currently include a fully pinned dependency file. To record your installed environment:

```bash
python -m pip freeze > requirements.txt
```

## Run the Home Assistant Agent

Prepare an accessible Home Assistant MCP service and a long-lived access token. Create `.env` in this directory:

```dotenv
HA_URL=http://your-home-assistant-host:8123
HA_TOKEN=your-home-assistant-token
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_MODEL=a-claude-model-id-available-to-your-account
```

Set `HA_URL` to the root address; the code appends `/api/mcp`. The server must provide an `Assist` prompt. `ANTHROPIC_MODEL` overrides the code's default, `claude-sonnet-5`; use a model ID available to your account.

Complete Google authorization using the steps below before starting Alpha. Alpha launches `python -m servers.personal_server` from the project root using `.venv/bin/python`; Alpha attempts discovery of both servers at startup, then closes their discovery connections. If either server is unavailable, it continues with the other server and retries the missing catalog on subsequent user turns until successful. Verify the Home Assistant connection, then start a conversation:

```bash
python devtools/ha_client.py
python alpha.py
```

Example inputs, using device names that exist in your home:

```text
你：查看客厅灯的状态
你：打开客厅灯
你：把客厅灯调暗一点
```

These ask Alpha to check the living room light, turn it on, and dim it. The current assistant prompt and CLI messages are in Chinese.

Enter `exit`, `quit`, or `退出`, or press Ctrl+C to stop. Conversation history is stored in memory and is lost when the process exits.

## Google Authorization and API Verification

Enable the Google Tasks API and Google Calendar API in your Google Cloud project. Configure the OAuth consent screen, create a desktop application OAuth client, and save its downloaded configuration as `credentials.json` in this directory. If the app is in testing mode, add the Google account used for authorization as a test user.

The code requests these scopes:

```text
https://www.googleapis.com/auth/tasks
https://www.googleapis.com/auth/calendar.events
```

Run:

```bash
python -m servers.google_auth
```

On the first run, a browser opens for Google sign-in and permission approval. A temporary local server receives the authorization callback on an available port. Credentials are saved to `token.json` and reused or refreshed on subsequent runs.

The credentials belong to the account that grants authorization. The current implementation uses one `token.json` and is intended for local, single-user use.

Names such as `My Tasks` in this script's output represent task lists, not individual tasks. The Calendar check uses `events().list()`; Tasks entries are not returned as Calendar events. The requested `calendar.events` scope does not authorize `calendarList().list()`.

## Test Tasks Tools with MCP Inspector

Run `python -m servers.google_auth` first to complete authorization. Then launch Inspector with Node.js and npm available:

```bash
npx @modelcontextprotocol/inspector
```

Open the web address printed in the terminal and add a server:

| Field | Configuration |
| --- | --- |
| Server ID | `personal_server` |
| Transport | `stdio (local process)` |
| Command | Absolute path to this project's `.venv/bin/python` |
| Arguments | `-m` and `servers.personal_server`, one argument per line; set environment variable `PYTHONPATH` to the project root |
| Working directory | Absolute path to the `alpha` directory |
| Environment | Leave empty |

Inspector launches the server. You do not need to start `servers/personal_server.py` separately.

After connecting, list the tools and call `list_items`:

```json
{
  "list_name": "My Tasks"
}
```

Calling `add_items` creates real tasks:

```json
{
  "list_name": "My Tasks",
  "items": ["Buy milk", "Study Python"]
}
```

The SDK wraps returned text in an MCP tool result. Expected failures use `ToolError`, producing `isError=true` with an explanatory message. In stdio mode, stdout carries protocol messages; send debugging logs to stderr.

## Files

Run every command from the project root (`alpha/`) with the project virtual environment. Entry points run as modules (`python -m package.module`) so that packages import each other.

```
alpha/
├─ alpha.py                  Compatibility launcher → apps.cli
├─ apps/
│  └─ cli.py                 Text / voice terminal loop around AlphaBrain   (python -m apps.cli [--voice])
├─ brain/
│  ├─ agent.py               AlphaBrain: Claude agent loop, tools, history, time context, interruption sync
│  └─ server.py              AlphaBrain as a local HTTP service              (python -m brain.server)
├─ core/
│  ├─ paths.py               Single source of truth for file paths (.env, venv, Google credentials)
│  └─ rules.py               Shared Alpha system rules
├─ integrations/
│  ├─ mcp_client.py          MCP discovery, allowlist, HA calls, prompt assembly
│  └─ personal_worker.py     Background task holding the personal MCP session (queue + reconnect)
├─ servers/
│  ├─ personal_server.py     Google Tasks / Calendar stdio MCP server        (launched by Alpha)
│  └─ google_auth.py         Google OAuth credentials + API check          (python -m servers.google_auth)
├─ voice/
│  ├─ audio.py               Playback, device selection, local VAD, speech filtering
│  ├─ listen.py              Push-to-talk recording + Whisper transcription (python -m voice.listen)
│  ├─ qwen_realtime.py       Qwen Omni realtime voice layer                 (python -m voice.qwen_realtime)
│  └─ openai_realtime.py     OpenAI Realtime version, kept for reference   (python -m voice.openai_realtime)
├─ devtools/
│  ├─ ha_client.py           Verify the HA MCP connection                   (python devtools/ha_client.py)
│  └─ mcp_probe.py           Raw JSON-RPC probe for HA MCP                   (python devtools/mcp_probe.py)
└─ tests/                    Unit tests, no real services                   (python -m unittest discover -s tests -t .)
```

Secrets stay in the project root and are git-ignored: `.env`, `credentials.json`, `token.json`.

## Limitations and Next Steps

- Each server is discovered independently at startup. Unavailable services are noted in the system prompt, and missing catalogs are retried on subsequent turns until successful. Each HA tool call uses a fresh HTTP transport/session; personal calls use a persistent worker session that reconnects after connection failures. Failed operations are not automatically replayed.
- After successful discovery, tool catalogs and the HA Assist prompt remain cached until restart; server-side tool changes are not automatically refreshed. The personal worker retries connections after one second; its calls have a default 60-second timeout including queueing. Failed calls are not replayed, and requests that expire before execution are skipped.
- Tasks supports reading and adding entries; completion and deletion tools are not implemented.
- Calendar reminder creation is exposed through MCP; event modification and deletion are not implemented.
- Tool calls run sequentially. Concurrent scheduling, persistent conversation history, and multi-user credential storage are not implemented.
- If a conversation turn fails, its messages are not saved to history, but device operations or task insertions that already succeeded remain in effect.

`.env`, `credentials.json`, `token.json`, and the virtual environment are excluded by `.gitignore`. Keep these files local rather than committing them with the source code.

## Calendar Reminders and Cross-Server Tool Calls

`create_reminder` creates a 15-minute event with a popup reminder. Pass `local_time` without a timezone or UTC offset. The server uses Python `ZoneInfo` to calculate the correct offset for that date.

Configure the IANA timezone in `.env` (default: `America/Los_Angeles`):

```dotenv
REMINDER_TIMEZONE=America/Los_Angeles
```

```json
{
  "title": "Pay rent",
  "local_time": "2026-11-03T08:00:00",
  "minutes_before": 10
}
```

Invalid dates, timezone-bearing inputs, empty titles, and minutes outside 0–40320 are rejected. Local times that do not exist or occur twice during daylight saving transitions are also rejected rather than silently selecting an instant. Event duration is calculated in UTC to preserve 15 elapsed minutes across transitions.

For relative dates, Alpha prefixes each user message with the current local timestamp and configured IANA timezone. The system prompt instructs the model to use that message's timestamp as its reference. Home Assistant's time tool is not exposed. Python computes the reminder's UTC offset using the same `REMINDER_TIMEZONE` configuration; ambiguous user requests require clarification.

The `when` parameter has been replaced by `local_time`. Restart Alpha or reconnect Inspector to refresh the cached tool schema. Only absolute local-date conversion is handled by code; interpreting relative phrases remains the model's responsibility.

## Voice Mode (macOS)

Install the audio dependencies in the project environment:

```bash
python -m pip install sounddevice numpy faster-whisper webrtcvad-wheels
```

Start Alpha with voice input and spoken answers:

```bash
python alpha.py --voice
```

Press Enter to start recording. VAD mode 2 stops after the configured `SILENCE_MS` of consecutive non-speech following detected speech (default: 1500 ms; rounded up to a 30 ms frame boundary). Recording is discarded after 5 seconds without speech and capped at 15 seconds after speech begins. Alpha transcribes the recording, prints the recognized text and timing, runs the agent, and reads its answer using macOS `say`. Enter `q` at the recording prompt to exit. Without `--voice`, Alpha retains its text input mode. Allow microphone access for your terminal when macOS requests it; the system speech settings determine the voice used for playback.

`listen.py` exposes `record() -> numpy.ndarray` and `transcribe(audio) -> str`. Audio is mono float32 at 16 kHz. The small Whisper model runs on CPU with int8 computation and is loaded lazily, then reused; its first use downloads the model if needed. Recording and transcription run in worker threads in Alpha. Run `python -m voice.listen` to test speech recognition independently.

Set `SILENCE_MS=1000` in `.env` to adjust VAD stopping. To compare 1300 ms, run `SILENCE_MS=1300 python -m voice.listen`. Test both “关客厅灯” and “把牛奶……还有鸡蛋，加到购物清单”. Each recording prints the wall-clock wait since the last detected speech frame, silence duration in audio samples, and stop reason.

### Two-stage Speech End Detection

After 400 ms of silence (420 ms at 30 ms frame boundaries), a background transcription checks simple ending rules. One- or two-character text, trailing conjunctions, and incomplete “把…” fragments continue waiting; other text can end recording early once transcription finishes. Resumed speech invalidates the earlier preview. `SILENCE_MS=1500` is the fallback silence limit. Preview inference does not block VAD processing, but it adds inference work and may not finish before the fallback limit. Shared model inference is serialized; final transcription may wait for an unfinished preview.

## Realtime Interruption Confirmation

Run `python -m voice.openai_realtime` with `OPENAI_API_KEY` configured. The script uses streaming microphone input and output, with a two-stage interruption policy:

- `speech_started` lowers output volume to 20% while playback continues.
- At 400 ms of local VAD speech, playback pauses without consuming or clearing queued audio. Generation continues until transcription resolves the turn.
- Meaningful transcription clears playback, cancels active generation, and truncates the unheard response. Noise such as “咳” resumes playback from the saved position; transcription failure also resumes the old response.
- Short noise restores volume without cancelling the response. Local VAD is used because semantic VAD may delay its end-of-speech event.

Set `INTERRUPT_MS` to an integer from 300 to 500 (default: 400). Local VAD processes 20 ms frames and tolerates gaps shorter than 100 ms. It is a heuristic, not a guarantee that every noise burst will be rejected. Input transcription uses `gpt-4o-mini-transcribe`; automatic response creation and interruption are disabled, so the client triggers responses only when audio is committed and transcription is meaningful. The latency log measures receipt of `speech_stopped` to the first returned audio chunk, rather than acoustic end-of-speech to audible playback.

### Home Assistant Tools in Realtime

`realtime.py` loads HA tools and the Assist prompt at startup using `HA_URL` and `HA_TOKEN`. It exposes only `intent__HassTurnOn`, `intent__HassTurnOff`, `light__HassLightSet`, and `homeassistant__GetLiveContext`. Tool schemas use Realtime's `type: function` and `parameters` format. If HA discovery fails, the conversation continues with no home tools.

Completed `response.done` events containing function calls dispatch HA operations in a background task so audio events remain responsive. Calls are checked against both the allowlist and discovered tools. Results, including errors, are sent as `function_call_output` items tied to their `call_id`; one continuation response is requested after the batch. Cancelled responses are not executed, and repeated call IDs are not replayed. Already executed device operations are not undone by a voice interruption.

## Qwen Omni Voice MVP

`realtime_qwen.py` tests `qwen3.8-omni-flash-realtime` independently of the OpenAI conversation pipeline. Both scripts import playback, device selection, local VAD, and speech filtering from `audio.py`. Configure `.env`:

```dotenv
DASHSCOPE_API_KEY=your-regional-model-studio-key
QWEN_WORKSPACE_ID=your-workspace-id
QWEN_REGION=singapore
QWEN_VOICE=Tina
INTERRUPT_MS=400
QWEN_SESSION_SECONDS=1800
# Optional: reuse the same HA credentials as the CLI/OpenAI voice layer.
HA_URL=http://your-home-assistant:8123
HA_TOKEN=your-long-lived-access-token
```

The key and workspace must belong to the same selected region. `QWEN_REGION` accepts `singapore` or `beijing`. This model requires a workspace-specific WebSocket endpoint, which the script constructs. Existing `REALTIME_INPUT_DEVICE` and `REALTIME_OUTPUT_DEVICE` settings also apply.

```bash
python -m voice.qwen_realtime
```

Speak directly in Mandarin; press Ctrl+C to exit. Input is mono PCM16 at 16 kHz, output is 24 kHz. Wear headphones to reduce feedback.

At startup it discovers `list_items`, `add_items`, `complete_items`, and `create_reminder` from the personal MCP server and starts a persistent `PersonalWorker`, stopped on exit. Google credentials use the same setup as the CLI. Qwen function definitions use the nested `function` object and `parameters` schema. Tool calls run in background tasks, so incoming voice events remain responsive. Each call returns a `function_call_output` with its original `call_id`, including errors. The model continues after all results in a completed response have arrived. Repeated function events are deduplicated to avoid executing a write twice.

With `HA_URL` and `HA_TOKEN` configured, it also discovers `intent__HassTurnOn`, `intent__HassTurnOff`, `light__HassLightSet`, and `homeassistant__GetLiveContext`. The HA Assist prompt supplies the static device context. Both catalogs are checked for duplicate names before filtering, and tools are routed to the server they belong to. Personal tools use the worker; HA tools use `call_ha_tool`, with a fresh MCP session per call and a shared HTTP client. The same catalogs, Assist prompt, and HTTP client are reused after Qwen reconnection. Missing HA configuration or failed startup discovery leaves chat and personal tools available; rediscovering HA tools then requires restarting the script. Tool failures are returned to the model and are not automatically replayed.

Successful on/off/brightness batches finish silently: when every tool in the response is a device action with `is_error=False`, HA `response_type=action_done`, and no failure details, the client returns tool outputs and requests a follow-up response to finish processing them, but discards that response's audio and transcript. This still incurs model generation cost. No additional current-time message is inserted for this silent follow-up. If it unexpectedly asks for another tool, the client does not execute that request and reconnects instead. Queries, errors, uncertain results, and batches containing personal tools continue to the model for a spoken answer. Instructions also ask the model to omit pre-action narration.

Current time is injected as a separate text message before response creation, including tool continuations, using `REMINDER_TIMEZONE` (default `America/Los_Angeles`). Instructions remain static. Tool logs show the name, arguments, and first 100 characters of the result. A new meaningful user turn suppresses a stale tool continuation; already-running writes are not replayed automatically.

Tool transitions may be silent or use one short conversational phrase such as “嗯，我看看”; instructions discourage repetitive narration. Shared playback queues consecutive audio items rather than clearing the previous item's unplayed tail when a fast tool result arrives. Item playback counters switch when the callback actually starts the next item. Confirmed user interruption still clears queued audio. Acoustic naturalness and echo-related false interruptions require microphone testing.

The client disables automatic responses and interruption. Speech onset ducks playback to 20%; sustained local VAD (default 400 ms, configurable from 300 to 500) pauses playback without clearing it. Meaningless transcriptions resume the buffered answer and delete the noise item. Meaningful transcriptions clear playback, cancel an active response, and truncate its audio history to the estimated heard duration. A new response is requested only after meaningful transcription and audio commitment; cancellation completes before another response starts.

Logs include transcripts, end-of-speech-to-first-audio latency, usage, per-response cost, and cumulative cost in USD. Prices use the official Qwen3.8 table checked on 2026-10-13: Singapore input text/audio $0.23/$0.93 and output text/audio $0.70/$1.87 per million tokens; Beijing $0.212/$0.848 and $0.636/$1.696. Both output text and audio are charged. Estimates exclude free quotas and discounts. Missing usage details are reported as unpriced, rather than treated as free.

Closed connections, network failures, and session-expiration errors trigger reconnection and a fresh `session.update`. The script also deliberately rotates sessions every 1800 seconds by default; `QWEN_SESSION_SECONDS` controls this interval. New connections start fresh conversation history; cumulative cost survives reconnections. Invalid configuration/authentication errors are reported instead of retried indefinitely.

`Turn queue overflow` also triggers reconnection without replaying device writes. This is recovery from a full server queue, not proof of its underlying cause. Numbered `response.create` logs distinguish normal replies from silent follow-ups; live event logs are needed to confirm whether repeated tool turns, background speech, or another server behavior caused accumulation.

Compatibility needs live verification: the official server example includes `create_response` and `interrupt_response`, but the client parameter reference does not explicitly document these switches, `conversation.item.delete`, or `conversation.item.truncate`. Delete/truncate are compatibility attempts, not confirmed Qwen capabilities. The script checks that the server echoes both switches as false and reports rejected commands rather than silently claiming noise recovery/history truncation works. Offline tests verify the client state machine, not server support or microphone behavior.

```bash
.venv/bin/python -m unittest tests.test_realtime_audio tests.test_brain
```

Official references: [client events](https://www.alibabacloud.com/help/en/model-studio/client-events), [server events](https://www.alibabacloud.com/help/en/model-studio/server-events), [model and prices](https://www.alibabacloud.com/help/en/model-studio/qwen3-8-omni-flash-realtime).

## Reusing AlphaBrain

The CLI calls a reusable brain; importing it does not initialize audio, clients, or models. Conversation history, cached catalogs, HA availability, routing, and last-request timings belong to the brain instance.

```python
from brain import AlphaBrain

async def handle_requests():
    brain = AlphaBrain()
    try:
        await brain.start()
        answer = await brain.ask("关闭客厅灯")
        print(answer)
    finally:
        await brain.close()
```

`ask()` injects current local time, retries catalogs that failed discovery, and preserves the existing tool loop and successful-device shortcut. `close()` stops the personal worker and closes the clients. Calls are serialized to keep history consistent. Use `python -m apps.cli` or the existing `python alpha.py`; both retain `--voice`.

Brain and MCP debug output uses the `alpha.brain`, `alpha.mcp`, and `alpha.personal_worker` logging namespaces. CLI timing uses `alpha.cli`. The CLI enables INFO logs on stderr; library imports do not configure global logging. Other voice layers can configure these loggers independently.

Regression tests (no external API requests):

```bash
.venv/bin/python -m unittest test_brain -v
```
