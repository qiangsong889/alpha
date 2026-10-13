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
           └─ session_personal → personal_server.py (stdio)
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

Complete Google authorization using the steps below before starting Alpha. Alpha launches `personal_server.py` using this directory's `.venv/bin/python`; Alpha attempts discovery of both servers at startup, then closes their discovery connections. If either server is unavailable, it continues with the other server and retries the missing catalog on subsequent user turns until successful. Verify the Home Assistant connection, then start a conversation:

```bash
python ha_client.py
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
python google_auth.py
```

On the first run, a browser opens for Google sign-in and permission approval. A temporary local server receives the authorization callback on an available port. Credentials are saved to `token.json` and reused or refreshed on subsequent runs.

The credentials belong to the account that grants authorization. The current implementation uses one `token.json` and is intended for local, single-user use.

Names such as `My Tasks` in this script's output represent task lists, not individual tasks. The Calendar check uses `events().list()`; Tasks entries are not returned as Calendar events. The requested `calendar.events` scope does not authorize `calendarList().list()`.

## Test Tasks Tools with MCP Inspector

Run `google_auth.py` first to complete authorization. Then launch Inspector with Node.js and npm available:

```bash
npx @modelcontextprotocol/inspector
```

Open the web address printed in the terminal and add a server:

| Field | Configuration |
| --- | --- |
| Server ID | `personal_server` |
| Transport | `stdio (local process)` |
| Command | Absolute path to this project's `.venv/bin/python` |
| Arguments | Absolute path to `personal_server.py`, one argument per line |
| Working directory | Absolute path to the `alpha` directory |
| Environment | Leave empty |

Inspector launches the server. You do not need to start `personal_server.py` separately.

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

| File | Purpose |
| --- | --- |
| `alpha.py` | Claude + Home Assistant CLI agent |
| `ha_client.py` | Verify the Home Assistant MCP connection, tools, and Assist prompt |
| `personal_server.py` | Google Tasks and Calendar stdio MCP server |
| `personal_worker.py` | Queued personal MCP calls, persistent session, timeout, and reconnect loop |
| `google_auth.py` | Google credential management and Tasks / Calendar API verification |
| `mcp_probe.py` | Probe Home Assistant MCP using raw JSON-RPC |
| `mcp_probe_browser.js` | Browser-side MCP probe script |
| `.gitignore` | Exclude local credentials, environments, caches, and logs |

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
python -m pip install sounddevice numpy faster-whisper
```

Start Alpha with voice input and spoken answers:

```bash
python alpha.py --voice
```

Press Enter to start recording, then Enter again to stop. Alpha transcribes the recording, prints the recognized text and timing, runs the agent, and reads its answer using macOS `say`. Enter `q` at the recording prompt to exit. Without `--voice`, Alpha retains its text input mode. Allow microphone access for your terminal when macOS requests it; the system speech settings determine the voice used for playback.

`listen.py` exposes `record() -> numpy.ndarray` and `transcribe(audio) -> str`. Audio is mono float32 at 16 kHz. The small Whisper model runs on CPU with int8 computation and is loaded lazily, then reused; its first use downloads the model if needed. Recording and transcription run in worker threads in Alpha. Run `python listen.py` to test speech recognition independently.
