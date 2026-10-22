# Alpha

Alpha is a personal AI assistant under development. Its goal is to control home devices through natural language and connect personal services, such as tasks and calendars, through MCP. The project also provides hands-on practice with asynchronous Python, model tool calling, MCP, and OAuth.

The current implementation connects AlphaBrain and its voice/CLI frontends to Home Assistant and a local Google personal-services MCP server, including Tasks, Calendar, Docs, Drive, Gmail and YouTube Analytics.

## Quick Start (current version)

Everything runs from the `alpha/` directory with the project's `.venv` (see [Installation](#installation)
and [Google Authorization](#google-authorization-and-api-verification) for first-time setup).
The sections further down also document earlier paths (Qwen Omni, LiveKit, push-to-talk); the
commands below are the ones in daily use.

### 1. Check the configuration

All settings live in `.env`; their names, defaults, and valid ranges are listed in
[docs/SETTINGS.md](docs/SETTINGS.md). Required keys:

| Process | Required |
| --- | --- |
| Brain | `HA_URL`, `HA_TOKEN`, `ANTHROPIC_API_KEY` |
| Voice | `DASHSCOPE_API_KEY`, `QWEN_WORKSPACE_ID` (`TYPESAFE_API_KEY` recommended for Jev; without it Jev falls back to local rules) |

```bash
.venv/bin/python -m core.settings      # errors, warnings and current values; secrets shown only as set/unset
```

Both processes run the same check at startup and refuse to start with a list of every problem.

### 2. Start Alpha (voice)

```bash
./start-qwen
```

The launcher starts AlphaBrain (`brain.server`, `http://127.0.0.1:8765`), waits until it is healthy,
then starts the voice layer (`voice.qwen_alpha`: Silero segmenter → Qwen ASR → Jev → AlphaBrain → Qwen TTS).
Press Ctrl+C once to stop both. If a healthy brain is already running, it is reused and left running.
Wear headphones: there is no echo cancellation.

To run the two processes separately (for example, to read each one's logs in its own terminal):

```bash
.venv/bin/python -m brain.server        # terminal 1: brain on 127.0.0.1:8765
.venv/bin/python -m voice.qwen_alpha    # terminal 2: voice, after the brain is ready
```

Text-only, without voice or `brain.server` (the CLI runs its own AlphaBrain in-process):

```bash
.venv/bin/python -m apps.cli            # add --voice for macOS record + `say` playback
```

### 3. Open the console (optional, read-only)

The console shows process status, today's conversations and costs, and every setting
(saved value vs. the value the running process is using). It needs Node.js only to build the page once.

```bash
cd console/web && npm install && npm run build && cd ../..   # first time, or after frontend changes
.venv/bin/python -m apps.console                             # 127.0.0.1:8770
```

Open the `http://127.0.0.1:8770/login?token=...` link printed in the terminal. The token changes
on every start. The console can run whether or not Alpha is running. Details: [docs/CONSOLE.md](docs/CONSOLE.md).

### 4. Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .   # unit tests, no real services, no model calls
```

The eval in `eval/` calls real models and costs money; run only the cases affected by a change.

## Current Features

### Home Assistant Agent

- Calls Claude using `AsyncAnthropic` and responds briefly in Chinese.
- Enables Anthropic's server-side `web_search_20260209` tool with its default
  dynamic filtering for current information and
  explicit search requests (up to five searches per API request). No additional
  MCP server or search API key is needed. Search queries, errors and cited sources
  appear in brain logs; text answers include source links, while voice answers
  stay brief and do not read URLs. For explicit requests such as “帮我搜一下”,
  “上网查” or “最新新闻”, LiveKit emits “我查一下。” before the brain
  POST. It sends `search_announced=true` so the brain does not repeat it.
  Otherwise, the first streamed search-start event triggers the same phrase.
  This fixed progress phrase is emitted once and is
  separate from the final answer. Search results stay in conversation history.
  Search has additional Anthropic usage charges. If the API reports that search
  is disabled, enable web search for your organization in the Claude Console.
  Home weather and outdoor advice now prefer `get_weather(day="today"|"tomorrow")`,
  using Open-Meteo rather than web search or a home weather sensor. The tool
  reads home coordinates and timezone from HA's authenticated `GET /api/config`
  with existing `HA_URL`/`HA_TOKEN`, caches the valid location until restart,
  and refreshes weather on each call. Missing coordinates and `(0, 0)` are errors.
  Both services have five-second request timeouts. Results contain only selected
  weather data, never coordinates; tomorrow's summary does not mix in current
  temperatures. Errors distinguish location lookup from weather-service failure.
  Web search uses the optional `HOME_CITY`, `HOME_REGION` and `HOME_COUNTRY`
  environment settings as an approximate location, with the configured
  `REMINDER_TIMEZONE`. The tool definition is built after `.env` loads; without
  `HOME_CITY`, no `user_location` is sent. These search settings do not override
  the weather tool's fixed home location. The model has no separate home-location
  tool: lookup remains internal to `get_weather`. HA Assist instructions apply to home devices, not to
  external-information queries. See [Open-Meteo](https://open-meteo.com/en/docs)
  for forecast data and WMO weather codes.
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
- A local stdio MCP server exposing Google personal-services tools (see the tool list below).

| MCP Tool | Input | Output |
| --- | --- | --- |
| `list_items` | `list_name` | Task titles and completion status |
| `add_items` | `list_name`, `items` | A text confirmation plus structured IDs for tasks created by this call |
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
  google-auth google-auth-oauthlib google-api-python-client OpenCC jsonschema
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

Enable Google Tasks API, Google Calendar API, Google Docs API, Google Drive API, Gmail API and YouTube Analytics API in your Google Cloud project. Configure the OAuth consent screen, create a desktop application OAuth client, and save its downloaded configuration as `credentials.json` in this directory. If the app is in testing mode, add the Google account used for authorization as a test user.

Scopes are requested in **two separate OAuth flows**, using the same desktop
client configuration. Google rejects `drive.file` combined with YouTube scopes
in one request.

Workspace (`token.json`):

```text
https://www.googleapis.com/auth/calendar
https://www.googleapis.com/auth/tasks
https://www.googleapis.com/auth/drive.readonly
https://www.googleapis.com/auth/drive.file
https://www.googleapis.com/auth/gmail.readonly
https://www.googleapis.com/auth/gmail.send
```

YouTube (`youtube_token.json`):

```text
https://www.googleapis.com/auth/yt-analytics.readonly
https://www.googleapis.com/auth/yt-analytics-monetary.readonly
https://www.googleapis.com/auth/youtube.readonly
```

Run:

```bash
python -m servers.google_auth
```

On the first run, the script authorizes Workspace and YouTube sequentially in the browser. A temporary local server receives each authorization callback on an available port. Credentials are saved separately to `token.json` and `youtube_token.json`, then reused or refreshed independently on subsequent runs.

Credentials belong to the account that grants authorization. This is a local, single-user setup; the Workspace account and the YouTube channel are selected separately.

Names such as `My Tasks` in this script's output represent task lists, not individual tasks. The Calendar check uses `events().list()`; Tasks entries are not returned as Calendar events. The full `calendar` scope now also permits listing calendars. Running the module performs authorization only; its optional `main()` function contains the Tasks/Calendar read checks. If saved credentials lack the new scopes, the authorization flow opens again without manually deleting `token.json`.

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
QWEN_VAD=server_vad
QWEN_SILENCE_MS=800
# Optional: reuse the same HA credentials as the CLI/OpenAI voice layer.
HA_URL=http://your-home-assistant:8123
HA_TOKEN=your-long-lived-access-token
```

The key and workspace must belong to the same selected region. `QWEN_REGION` accepts `singapore` or `beijing`. This model requires a workspace-specific WebSocket endpoint, which the script constructs. Existing `REALTIME_INPUT_DEVICE` and `REALTIME_OUTPUT_DEVICE` settings also apply.

```bash
python -m voice.qwen_realtime
```

Speak directly in Mandarin; press Ctrl+C to exit. Input is mono PCM16 at 16 kHz, output is 24 kHz. Wear headphones to reduce feedback.

VAD defaults to `server_vad`, with `QWEN_SILENCE_MS=800` (allowed range 200–6000 ms).
Set `QWEN_VAD=semantic_vad` to restore semantic detection; the silence setting is
only sent for `server_vad`. Acoustic VAD avoids semantic backchannel filtering,
but a pause around the configured silence duration may split a sentence. The
client verifies the returned VAD configuration before opening the microphone.
See the [official client events](https://help.aliyun.com/en/model-studio/client-events).

For a 60-second microphone-only check without business tools:

```bash
QWEN_VAD=server_vad QWEN_SILENCE_MS=800 .venv/bin/python -u -m devtools.qwen_vad_probe
```

At startup it discovers `list_items`, `list_reminders`, `add_items`, `complete_items`, and `create_reminder` from the personal MCP server and starts a persistent `PersonalWorker`, stopped on exit. Google credentials use the same setup as the CLI. Qwen function definitions use the nested `function` object and `parameters` schema. Tool calls run in background tasks, so incoming voice events remain responsive. Each call returns a `function_call_output` with its original `call_id`, including errors. The model continues after all results in a completed response have arrived. Repeated function events are deduplicated to avoid executing a write twice.

Qwen tool execution now goes through `ToolGateway` and the injected executor in `integrations/tool_executor.py`. A tool must be loaded and have a local policy. The gateway stores operations in `data/operations.db` (ignored by Git), shared across Qwen reconnects and closed on exit. At startup, leftover `executing` operations are changed to `unknown` and printed without retrying. A missing/closed personal worker or unavailable HA client is definitely not sent; HA connection/initialization failures are also classified as not sent. Once a tool call begins, transport failures and timeouts are unknown. HA device actions require a clean `action_done`; other tools use MCP `is_error`.

**Automatic batches:** the Qwen entry point uses `SimpleCoordinator`. HA on/off/brightness commands execute immediately and successful device actions stay silent. Other personal mutations (including deletion, undo and restoration) collect in a two-second window. User instructions authorize execution: there is no separate “yes” step and no `confirmed` parameter. Ambiguous targets still require clarification.

Code seals a batch after the response and all tool results have arrived and the silence window has elapsed. A batch executes sequentially, then receives one factual spoken report. Only after that report completes does the next queued batch start. New instructions can be interpreted and collected while an earlier batch executes. A larger number of tasks does not add confirmation. When a response contains only pending writes, no extra acknowledgment response is requested; the model resumes for the actual execution report. Reads and rejected tool calls can still request a normal continuation.

New speech pauses collection; noise resumes it. Recognized unfinished expressions such as “等等”, “还有”, or “把” wait up to five seconds and then prompt for clarification. This is a local heuristic, not a general completeness detector. `COLLECTION_WINDOW_S=2` and `COLLECTION_INCOMPLETE_S=5` configure the delays, measured from speech end.

Four local control tools are exposed:

- `cancel_pending()`: cancel all unsent operations, including queued batches. A request already in flight is allowed to finish and is not described as cancelled.
- `undo_recent(title=...)`: register an exact deletion of an item/reminder Alpha created during this program session. No extra confirmation is needed.
- `delete_item(kind, item_id, list_id)`: register deletion of an exact existing resource, without asking for agreement again. IDs come from queries; unclear targets prompt for clarification.
- `restore_recent(title=...)`: register restoration of a resource Alpha deleted during this program session, using the saved snapshot. No extra confirmation is needed. Restored resources get new Google IDs.

`list_items` and `list_reminders` supply structured IDs, which should not be spoken. Immediately before deletion, code reads the original resource and persists its snapshot in SQLite, independently of the operation's final result. If a backup cannot be obtained, deletion is not sent. Task restoration retains title, notes, due date and completion data; ordinary calendar restoration retains title, description, location, timing and reminders. Automatic deletion rejects recurring/special calendar events and events with participants or conferencing because their original behavior cannot be faithfully restored by this implementation. Task IDs/order and externally linked task metadata are not restored exactly.

Undo and restoration have separate operation records; original successes remain recorded. Repeated pending/executing/unknown/successful deletions or restorations are not blindly sent again. Unknown Google writes receive read-only verification where available; unproven results remain unknown. SQLite facts and deletion snapshots survive process restarts, but the voice `undo_recent`/`restore_recent` selectors currently cover this program session. WebSocket reconnect retains work and reports; unsent work remains paused with a spoken recovery prompt until a new mutation or explicit continuation. Shutdown cancels unsent operations and leaves interrupted in-flight requests unknown.

Plans, dependency tools, required collection controls, recap comparisons and fallback TTS are absent from the production Qwen path. Earlier batch modules remain for regression tests. See [execution and undo](docs/task-batches.md) for the current flow.

With `HA_URL` and `HA_TOKEN` configured, it also discovers `intent__HassTurnOn`, `intent__HassTurnOff`, `light__HassLightSet`, and `homeassistant__GetLiveContext`. The HA Assist prompt supplies the static device context. Both catalogs are checked for duplicate names before filtering, and tools are routed to the server they belong to. Personal tools use the worker; HA tools use `call_ha_tool`, with a fresh MCP session per call and a shared HTTP client. The same catalogs, Assist prompt, and HTTP client are reused after Qwen reconnection. Missing HA configuration or failed startup discovery leaves chat and personal tools available; rediscovering HA tools then requires restarting the script. Tool failures are returned to the model and are not automatically replayed.

Successful on/off/brightness operations are silent when HA returns a clean `action_done`. Queries, errors, uncertain results and personal-task reports can receive a spoken answer. Pending-write acknowledgments and unsupported completion claims are held or suppressed locally; their generation can still incur cost. Final execution reports use actual operation states.

Current time is injected as a separate `system` text item before response creation, including tool continuations, using `REMINDER_TIMEZONE` (default `America/Los_Angeles`). It no longer becomes the newest user message. This is an explicit compatibility attempt: the current Omni client reference does not document generic system messages or history insertion with `previous_item_id`; server acceptance still requires live verification. A role rejection is reported rather than silently falling back to a timestamp as a user message. Instructions remain static. Tool logs show the name, arguments, and first 100 characters of the result. A new meaningful user turn suppresses a stale tool continuation; already-running writes are not replayed automatically.

At `response.done`, completion phrases such as “已经设好”, “已添加”, and “标记好了” are checked against successful write-tool calls for the same user request, including earlier tool responses in that request. Read-only calls cannot establish write success. The check waits for pending tools; unverified claims print `[⚠️ 疑似虚构]` and inject a system correction before one additional `response.create`. Corrections are capped at one per user request; failed or timed-out writes must be checked rather than blindly repeated. Final response text and streamed transcripts are both supported. This is a keyword heuristic, not proof that every detail of a statement matches the operation. It may miss paraphrases or flag benign descriptions of earlier actions; spoken audio may already have played before the check runs. System correction items share the same unverified Omni role compatibility as time items.

Tool transitions may be silent or use one short conversational phrase such as “嗯，我看看”; instructions discourage repetitive narration. Shared playback queues consecutive audio items rather than clearing the previous item's unplayed tail when a fast tool result arrives. Item playback counters switch when the callback actually starts the next item. Confirmed user interruption still clears queued audio. Acoustic naturalness and echo-related false interruptions require microphone testing.

The client disables automatic responses and interruption. Speech onset ducks playback to 20%; sustained local VAD (default 400 ms, configurable from 300 to 500) pauses playback without clearing it. Meaningless transcriptions resume the buffered answer and delete the noise item. Meaningful transcriptions clear playback, cancel an active response, and truncate its audio history to the estimated heard duration. A new response is requested only after meaningful transcription and audio commitment; cancellation completes before another response starts.

Logs include transcripts, end-of-speech-to-first-audio latency, usage, per-response cost, and cumulative cost in USD. Prices use the official Qwen3.8 table checked on 2026-10-13: Singapore input text/audio $0.23/$0.93 and output text/audio $0.70/$1.87 per million tokens; Beijing $0.212/$0.848 and $0.636/$1.696. Both output text and audio are charged. Estimates exclude free quotas and discounts. Missing usage details are reported as unpriced, rather than treated as free.

Closed connections, network failures, and session-expiration errors trigger reconnection and a fresh `session.update`. The script also deliberately rotates sessions every 1800 seconds by default; `QWEN_SESSION_SECONDS` controls this interval. New connections start fresh conversation history; cumulative cost survives reconnections. Invalid configuration/authentication errors are reported instead of retried indefinitely.

`Turn queue overflow` also triggers reconnection without replaying device writes. This is recovery from a full server queue, not proof of its underlying cause. Numbered `response.create` logs distinguish normal replies from silent follow-ups; live event logs are needed to confirm whether repeated tool turns, background speech, or another server behavior caused accumulation.

Compatibility needs live verification: the official server example includes `create_response` and `interrupt_response`, but the client parameter reference does not explicitly document these switches, `conversation.item.delete`, or `conversation.item.truncate`. Delete/truncate are compatibility attempts, not confirmed Qwen capabilities. The script checks that the server echoes both switches as false and reports rejected commands rather than silently claiming noise recovery/history truncation works. Offline tests verify the client state machine, not server support or microphone behavior.

```bash
.venv/bin/python -m unittest discover -s tests
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

`ask()` injects current local time, retries catalogs that failed discovery, and continues the tool loop until Claude explicitly finishes the user turn. A successful device tool is one step, not an early exit: later device or personal tasks in the same request must still run. `close()` stops the personal worker and closes the clients. Calls are serialized to keep history consistent. Use `python -m apps.cli` or the existing `python alpha.py`; both retain `--voice`.

All brain tool calls pass through `ToolGateway` and are recorded in
`data/operations.db`. Only loaded, policy-approved public tools are exposed.
Deferred operations execute immediately after registration in this pipeline;
there is no additional two-second window. Lights retain their immediate execution
and factual device summaries. Completing even a single device request requires
a final Claude completion decision; the former one-request shortcut could
silently discard the remaining tasks and has been removed.

The voice brain now accepts multiple business tool calls per Claude response,
executes them in order, and returns their tool results together. It validates the
entire response before speaking; a summary accompanying unfinished business calls
is rejected rather than played. This reduces model round trips for independent
tasks without executing writes concurrently or ending after the first tool.

Identical `add_items` and `create_reminder` calls within ten minutes are recorded
as cancelled without execution. Claude may request an intentional repeat using
the brain-only boolean `confirm_duplicate: true`; this parameter is removed
before recording or executing tool arguments. An identical `unknown` write that
is unsafe to retry blocks another attempt for ten minutes, even with that flag;
query its actual outcome first. Blocked attempts do not extend that window.
After it expires, a new explicit user request may execute; expiration does not
prove the previous request failed, so duplicate creation remains possible.
Old records remain `unknown`. Retry-safe operations such as HA lights accept a
new user turn immediately, while automatic retries in the same turn stay blocked.

`brain.server` uses `AlphaBrain(style="voice")`: short spoken answers, oral time
and number phrasing, and no IDs. The factual device brightness summary also uses
spoken percentages. Text CLI behavior is unchanged.

The brain and Qwen entrypoints take an exclusive lock on the operation database
before startup recovery. Run one of these entrypoints at a time; a second one
fails with an explanation instead of changing an active operation to `unknown`.
On startup, abandoned `executing` records become `unknown`, and unsubmitted old
operations are cancelled. Records survive restarts; `close()` releases the store
and its lock.

Brain and MCP debug output uses the `alpha.brain`, `alpha.mcp`, and `alpha.personal_worker` logging namespaces. CLI timing uses `alpha.cli`. The CLI enables INFO logs on stderr; library imports do not configure global logging. Other voice layers can configure these loggers independently.

Regression tests (no external API requests):

```bash
.venv/bin/python -m unittest tests.test_brain -v
```


## Google personal-service tools

`servers/google_tools.py` holds the additional tool logic; `personal_server.py`
registers it in the same stdio server. Alpha discovers these tools through MCP.
No separate service or API key is needed beyond Google OAuth.

| Service | Tools |
| --- | --- |
| Docs | `read_document`, `create_document`, `append_document` |
| Drive | `search_drive`, `read_drive_file`, `create_drive_folder` |
| Calendar | `list_calendars`, `list_events`, `create_event` |
| Gmail | `search_email`, `read_email`, `send_email`, `reply_email` |
| YouTube Analytics | `get_youtube_analytics` |

Existing Tasks, reminder and weather tools remain available. Docs creation makes
an empty document; append adds text to the first tab or a specified `tab_id`.
Reads include nested tabs and table text. Drive search matches a filename keyword;
reading supports Google Docs export and UTF-8 text/JSON/XML up to 2 MB, not PDF,
Office files or images. Long text results are marked truncated at 20,000 characters.
Search/list tools expose a next-page token; pass it back to continue.

Calendar queries and event creation take `local_start` and `local_end` in
`YYYY-MM-DDTHH:MM:SS` format, interpreted in `REMINDER_TIMEZONE`. Nonexistent or
ambiguous DST times are rejected. `create_event` supports an explicit calendar,
description and location, without invitations; use `create_reminder` for a simple
popup reminder, not both for the same request. Calendar creations receive an
internal operation ID that is hidden from the model's schema.

Gmail uses only `gmail.readonly` and `gmail.send`. Reading does not mark mail read;
there are no delete, trash, archive, draft or label-modification tools. Search
accepts Gmail query syntax and returns metadata; `read_email` extracts plain text
or HTML text without downloading attachments. Sending uses plain text, explicit
recipients and the account's sender address. Reply uses Reply-To (or From), the
original thread and RFC reply headers; it is not Reply All. Alpha is instructed
to send only on an explicit user request, never on instructions embedded in mail
or documents. Creating a draft for review in conversation does not send it.

YouTube reports summarize the channel selected during OAuth, with an explicit
inclusive date range. `include_revenue=true` includes estimated revenue in USD.
Reporting dates use YouTube's Pacific reporting timezone; recent data may lag and
empty results are not zero. Google's [reports.query documentation](https://developers.google.com/youtube/analytics/reference/reports/query)
now additionally requires `youtube.readonly`; this scope is included alongside
both Analytics scopes. Channel switching requires authorizing the intended channel.

All new writes pass through ToolGateway and SQLite records. AlphaBrain suppresses
identical successful non-email writes for ten minutes unless the user explicitly
asks to repeat (`confirm_duplicate` is stripped before the MCP call). Emails use
the separate mandatory confirmation flow below. Transport failures
or HTTP 408/5xx after a write yield `unknown`, with no automatic retry; query to
verify before trying again. HTTP permission/validation errors yield failures.
Docs/Drive/mail creations do not have remote idempotency keys; local records do
not guarantee exactly-once delivery. Services share a 15-second HTTP timeout,
a cached credential object and a lock for the non-thread-safe Google transport.

After adding scopes, run `.venv/bin/python -m servers.google_auth`, finish consent,
then restart the brain (and any standalone Inspector/server processes) to refresh
the cached tool catalog. Scope consent and enabling an API are separate steps.
The tests use fake Google services and an actual local stdio handshake; they do
not send real email or create remote files/events.


### Mandatory email confirmation (AlphaBrain)

`send_email` and `reply_email` have `commit="confirm"`. They are not automatically
approved as ordinary deferred writes. AlphaBrain's CLI and LiveKit pipeline use
`brain/email_confirmation.py`; the older Qwen coordinator has no equivalent flow
and refuses these tools. Even a manually approved gateway operation cannot send
without a code-owned confirmation token.

1. The first model call prepares a local snapshot through internal `prepare_email`
   (a read-only operation recorded by the gateway). It resolves the account sender,
   actual recipient and, for replies, subject/thread/reply headers. It does not
   create a Gmail draft or send a message. The server freezes the MIME bytes.
2. Code directly outputs the actual recipient, subject, complete body and
   `确认发送吗？`, without letting the model shorten or rewrite this preview.
3. Only after another nonempty user request can the model call the original tool
   with identical arguments and `confirmed=true`. The coordinator checks the new
   user turn, exact arguments, a two-minute lifetime and an unused confirmation.
   Model-internal tool iterations do not count as new user input. Whether the
   latest user sentence means agreement still relies on Claude; a new sentence
   alone does not prove consent. Questions, rejection and ambiguity must not be
   treated as agreement.
4. Code approves the original pending operation, consumes confirmation before
   awaiting execution, and supplies a private token to the executor. The model
   cannot set this token. The MCP server checks the same snapshot and consumes
   its token before issuing a single send request. Addresses are not resolved again.
5. Parameter changes prepare a new snapshot and cancel the previous pending
   operation. The changed snapshot needs another user turn before sending.
   Only one email can await confirmation at a time; preparing a different one
   replaces the previous unsent draft. Expiration is checked on access. Close
   cancels pending mail; neither drafts nor confirmation survive process restart.
   If the worker restarts, its frozen snapshot is lost and confirmation fails closed.

Timeout or cancellation after sending begins produces `unknown` and consumes
confirmation. Query sent mail before any new attempt. A successful previous
identical mail is explicitly distinguished from a new draft; another send still
needs a fresh preview and confirmation. `confirm_duplicate` is not a mail override.
No real mail is sent by the tests. They cover same-turn auto-confirmation, altered
parameters, frozen reply addresses, expiration, restart, replay and unknown results.


### Narrow Drive / Docs authorization

The app now requests `drive.readonly` (search/read all accessible files) plus
`drive.file` (create and modify files within the application's per-file access).
It no longer requests full `drive` or `documents` access. Docs reading accepts
`drive.readonly`; Docs creation and append accept `drive.file`.

Per-file access includes files created by Alpha and files explicitly opened or
shared with the application through an authorization flow such as Google Picker;
it is not literally an only-created-by-Alpha scope. No Picker is implemented here.
Searching or reading an existing file does not grant permission to modify it.
Existing documents outside this scope remain read-only; Google rejects writes.
See [Drive scopes](https://developers.google.com/workspace/drive/api/guides/api-specific-auth)
and [Docs batchUpdate scopes](https://developers.google.com/workspace/docs/api/reference/rest/v1/documents/batchUpdate).

Changing the requested scope list does not revoke already-issued broad grants.
The auth loader rejects cached credentials carrying the removed full scopes,
requests desktop-app consent without the unsupported `include_granted_scopes` parameter, validates the actual returned
grants, and saves actual scopes instead of merely requested scopes. If Google
still returns broad permissions, it fails without overwriting the token.

To fully migrate an existing broad grant: stop Alpha and standalone personal
servers, remove this app's old connection at
[Google account connections](https://myaccount.google.com/connections), then run
`.venv/bin/python -m servers.google_auth` and approve the narrower permissions.
Restart Alpha afterward so cached services and credentials are replaced. Removing
the connection also invalidates the app's other Google API grants; the same auth
flow requests their configured scopes again. No token revocation or interactive
Google sign-in is performed by the tests.


### Separate Workspace and YouTube tokens

The reported Google `invalid_request` error explicitly disallows requesting
`drive.file` and the YouTube scopes together. The scope groups above are now
separate: `get_credentials()` remains the Workspace entry point for Tasks,
Calendar, Docs, Drive and Gmail. `get_youtube_credentials()` uses only the three
YouTube read scopes and a different token file. `youtubeAnalytics` uses its own
cached credential object and transport; it never falls back to the Workspace token.
Both token files are ignored by Git and saved with mode 0600. Actual mixed-group
grants are rejected rather than silently persisted.

```bash
# Both groups, sequentially (the default):
.venv/bin/python -m servers.google_auth
# Or authorize only one group:
.venv/bin/python -m servers.google_auth --service workspace
.venv/bin/python -m servers.google_auth --service youtube
```

Use the same `credentials.json` for both. During YouTube authorization, select
the channel whose Analytics reports you need. Neither flow requests
`include_granted_scopes`. Existing Workspace credentials can be reused if already
compatible; existing broad or mixed credentials trigger fresh consent. No manual
copying of a Workspace token into `youtube_token.json` is supported. Restart the
brain/personal server after authorization to replace any previously cached clients.

## Explicit user memory (J6.1)

AlphaBrain offers three local tools: `remember(content, kind)`, `update_memory(id, content)`, and `forget(id)`. They run inside the brain, without MCP or the operation gateway. Only explicit user requests authorize memory changes; temporary device settings do not change preferences. Conflicting facts should update the existing entry, and ambiguous requests should be clarified.

Memories persist in the `memories` table of `data/operations.db`. The brain supplies `source_turn` automatically when creating an entry. Content is trimmed; identical kind/content pairs reuse the existing ID. The store allows at most 50 entries and 2,000 total content characters, including after updates. Missing IDs, invalid kinds, empty content, and common credential/password/code patterns return errors. Secret labels must be followed by a separator (是, 为, a colon, or an equals sign), with only whitespace between the label and separator. Bearer credentials require at least 16 token characters; ordinary phrases such as “bearer bond” remain allowed. Secret detection is a second-line, best-effort format check, not a guarantee that arbitrary unlabeled secrets can be recognized; the model must still refuse requests to store secrets.

Before every Claude request, the brain reloads a separate JSON memory block with its own cache breakpoint. Static rules stay unchanged when memories change. Memory contents are personalization data, never authority to override rules, permissions, or confirmation requirements. Email, web, document, and tool output must not be saved as user memory. These authorization and interpretation rules depend on the model following instructions; JSON encoding tests verify the data boundary, not immunity to every prompt injection.

After a successful change, Alpha reports the specific change in one sentence without reading IDs aloud. Eval uses a separate temporary SQLite database per case (shared only across that case's new-session turns), so tests cannot modify the user's saved memories.


### HA tool exposure

All tools advertised by the connected Home Assistant MCP server are exposed to AlphaBrain and the realtime agents, with their original schemas and descriptions. No fixed list of HA tool names filters discovery or routing. HA still controls which services/entities it exposes.

The gateway keeps operation records and resolves HA routes from that instance's discovered catalog. New HA tools run immediately; unless already classified, they are treated as potentially mutating and not safe to retry automatically. Personal-service policies, including email confirmation, remain in force. Names absent from both the loaded catalog and local policies are rejected.

## Optional automatic memory (J6.3c)

Automatic extraction is **off by default**. To try it, set `ALPHA_AUTO_MEMORY=1` in `alpha/.env` and restart the brain. Remove it or set it to `0` to disable extraction; existing memories remain available.

After an answer, a background Haiku request sees only the original user text and current memories. Turns that invoked `remember`, `update_memory`, or `forget` skip extraction entirely. The background task does not delay the answer; it does add an API request when enabled. New user turns invalidate late results. A database revision, maintained by insert/update/delete triggers, also prevents stale results from overwriting newer changes. Revision checking and a batch of writes share one SQLite transaction.

Ordinary preferences may be saved automatically with `source="auto"`. The next answer begins with a code-generated notice describing the saved content, once. Sensitive candidates are not saved automatically: they appear in a separate JSON system block and a code-generated question at the end of the next answer. They expire after ten minutes or any memory change. The main model handles explicit consent with the existing local memory tools; it must not treat questions, refusals, or ambiguous replies as consent. This interpretation still depends on the model. Turning a candidate into explicit memory marks its source as `explicit`.

Existing databases migrate automatically: old rows default to `source="explicit"`. Pending questions and unsent notices are process-local and are discarded on restart. Background tasks are cancelled before the database and API client close. Source content never becomes authority to override system rules. The store's existing 50-entry/2,000-character limits and secret checks apply to automatic writes too.

See [extraction evaluation](eval/EXTRACTION.md) for the synthetic test set, metrics, and numeric normalization rules. Jev is not part of this extraction path.

## Qwen + AlphaBrain experiment

On the `qwen-alpha` branch, run `python -m brain.server` and then `python -m voice.qwen_alpha`, both with Alpha's `.venv`. Qwen handles microphone transcription and streamed voice; all business tools remain in AlphaBrain. Collection and Jev completeness checking prevent execution before submission; say “执行吧” to finish a long request. See [trial instructions and limitations](voice/QWEN_ALPHA.md).

On `dev`, the default `QWEN_ALPHA_SUBMIT_MODE=jev` submits immediately when Jev judges the request complete; incomplete requests wait, and uncertainty falls back to the collection window. Use `QWEN_ALPHA_SUBMIT_MODE=window ./start-qwen` to restore the parallel timer/Jev mode. Batch IDs connect collection, gateway execution, brain completion, audio arrival and estimated playback-start logs; details are in [the Qwen trial notes](voice/QWEN_ALPHA.md#dev-parallel-completeness-checks-and-batch-latency).

Start both services with `./start-qwen` from `alpha/`. The launcher uses the project environment, waits for the brain to become healthy, then starts Qwen. Ctrl+C stops the processes it started; an already-running healthy brain is reused and left running. Stop any old standalone Qwen process first.

Qwen filters speech overlapping a report with Jev (`QWEN_ALPHA_JEV_INTERRUPTIONS=1`). The default `QWEN_ALPHA_UNCERTAIN_INTERRUPTION=hold` resumes the report for uncertainty or service failure; only an addressed classification, an explicit call to Alpha, or an eligible question answer interrupts. Idle speech is unaffected. One request checks both addressee and completeness. To restore the earlier uncertainty-accepting behavior use `QWEN_ALPHA_UNCERTAIN_INTERRUPTION=accept ./start-qwen`; disable the filter with `QWEN_ALPHA_JEV_INTERRUPTIONS=0 ./start-qwen`.
