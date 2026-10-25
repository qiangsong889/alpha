# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Alpha is a single-user personal voice assistant: speech in → Claude agent (AlphaBrain) → Home Assistant + Google services over MCP → speech out. The model's prompts and replies are in Chinese. So are most code comments, docs under `docs/` and `eval/`, and commit messages.

## Commands

Run everything from the repo root with the project venv (Python 3.11, `mcp==2.3.0`). Entry points are run as modules (`python -m pkg.module`) so the packages can import each other. The repo has no pinned requirements file. The install line is in README.md.

```bash
.venv/bin/python -m unittest discover -s tests -t .      # all unit tests (no real services, no model calls)
.venv/bin/python -m unittest tests.test_brain -v          # one module
.venv/bin/python -m unittest tests.test_brain.SomeClass.test_name   # one test

.venv/bin/python -m core.settings      # validate .env per process: errors / warnings / effective values
./start-alpha                          # brain + voice (apps.run_alpha); reuses an already-healthy brain
.venv/bin/python -m brain.server       # brain alone, 127.0.0.1:8765
.venv/bin/python -m voice.qwen_alpha   # voice alone (needs the brain running)
.venv/bin/python -m apps.cli           # text REPL with an in-process AlphaBrain (--voice for macOS mic/say)
.venv/bin/python -m apps.console       # read-only web console on :8770 (build first: cd console/web && npm install && npm run build)
.venv/bin/python -m core.costs         # print the cost ledger
.venv/bin/python -m servers.google_auth [--service workspace|youtube]   # browser OAuth; runtime never opens a browser
```

The eval in `eval/` calls the real Claude API and costs money. Run only the cases a change affects:
`.venv/bin/python -m eval.run_eval --list`, `--case <id>` (repeatable), `--repeat N`, `--judge` (run `eval.calibrate_judge` first). `tests.test_eval` checks the runner itself offline. Case format and scoring rules are in `eval/README.md`.

## Architecture

**Two processes, one HTTP boundary.** `brain.server` (FastAPI) wraps one `AlphaBrain(style="voice")`, with endpoints `/ask`, `/ask/stream`, `/interrupted`, `/delivered`, `/status`, `/costs` and `/health`. `voice.qwen_alpha` holds no business tools. It turns audio into a complete request and submits it to the brain exactly once. Each process takes a single-instance lock: the voice process locks `data/run/voice.lock`, and both brain and voice lock the operations DB. A second instance fails fast. It must never steal or recover a live instance's operations.

**Voice pipeline (`voice/`).** Audio comes from the Mac mic or the XVF3800 array, or from an ESPHome satellite that does on-device wake-word detection (`voice/satellite.py`, enabled by `ALPHA_SATELLITE_HOST`; device YAML in `esp-alpha/esphome/`). The chain is: Silero segmenter (`segmenter.py`) → serial Qwen ASR uploads (`asr_client.py`, `upload_gate.py`) → `TurnCollector` in `qwen_alpha.py` with Jev (`search_router.py`, a classifier that judges completeness and addressee) deciding when to submit → `BrainBridge` streams the answer → Qwen TTS (`qwen_tts.py`). Jev also filters barge-in speech while a reply is playing. Omni (`omni_transport.py`) is a legacy fallback that must be selected explicitly. `voice/QWEN_ALPHA.md` documents the behaviour and env flags.

**AlphaBrain (`brain/agent.py`).** This is the Claude tool loop (`AsyncAnthropic`, streaming, server-side `web_search`). The system prompt comes from `core/rules.py` plus HA's `Assist` prompt. The JSON memory block has its own cache breakpoint, so editing memories doesn't invalidate the static prompt cache. `ask()` is serialized by a lock and injects the local time into each user message. The loop keeps going until Claude explicitly finishes, and a successful device call is never an early exit. Other parts of the brain:
- `context.py`: idle reset of the session, plus Haiku compaction of history in the background (`docs/CONTEXT.md`).
- `memory_tools.py`: explicit memory (`remember` / `update_memory` / `forget`).
- `auto_memory.py` + `memory_extractor.py`: optional extraction (`ALPHA_AUTO_MEMORY`).
- `email_confirmation.py`: two-turn send confirmation backed by a frozen MIME snapshot and a code-owned token.
- `device_guard.py`: per-turn limits on device writes.
- `history_tools.py`: `search_history` over past turns and operations.

**Tool execution boundary (`integrations/`).** Every model tool call goes through `ToolGateway` (`tool_gateway.py`). The gateway records the call in SQLite (`operations.py` → `data/operations.db`) and then calls the executor (`tool_executor.py`). The executor routes the call to HA (a fresh Streamable-HTTP MCP session per call, `mcp_client.py`) or to the personal stdio MCP server, through a long-lived `PersonalWorker` queue. `tool_policy.py` is the allowlist. Each entry sets `mutates`, `retry_safe` and `commit` (`immediate` / `deferred` / `confirm`). A name with no policy is rejected, except for tools from HA's live catalog. Those default to mutating and not retry-safe. Outcome semantics:
- `ToolNotSent` means the call failed.
- `ToolOutcomeUnknown` means the status is `unknown`. Nothing retries automatically.
- Identical non-email writes within 10 minutes are suppressed unless the model passes `confirm_duplicate`. The brain strips that flag before recording or executing the call.
- `INTERNAL_TOOLS` are hidden from the model.

**Personal MCP server (`servers/`).** `personal_server.py` is the stdio server that Alpha launches. It covers Google Tasks, Calendar and reminders, and registers the Docs/Drive/Gmail/YouTube tools from `google_tools.py` and weather from `weather.py`. In stdio mode stdout carries the protocol, so log to stderr. `google_auth.py` manages two separate token files: `token.json` (Workspace) and `youtube_token.json` (YouTube). Google rejects the two scope groups in one request.

**Shared core (`core/`).**
- `paths.py` is the only place file paths are defined.
- `settings.py` defines every env setting in its `SETTINGS` registry: name, type, default, range, which process reads it, and whether the console may edit it. `require_settings(process)` validates at startup.
- `trace.py` holds the session/turn context vars used to attribute costs.
- `costs.py` is the SQLite cost ledger (`data/costs.db`). It converts cumulative ASR/TTS counters into per-request deltas.
- `archive.py` is the conversation archive (`data/archive.db`). Only `brain.server` and the CLI enable it. Tests and eval never write to it.
- `runtime.py` handles heartbeat/status files and compares saved settings with the values a running process uses (shown by the console).

`docs/ARCHIVE.md` explains how `session_id` and `turn_id` link the three DBs.

## Conventions

- Read settings with `from core.settings import setting; setting('NAME')`, never `os.getenv`. A new setting must be added to `SETTINGS` in `core/settings.py` and documented in `docs/SETTINGS.md`. `setting()` re-reads the environment on every call, so `patch.dict(os.environ, ...)` works in tests.
- Build paths from `core/paths.py` constants rather than ad-hoc joins.
- Tests use stdlib `unittest` (`IsolatedAsyncioTestCase` for async code) with fake HA, worker and Google services. They must never touch real devices, send real mail or call models.
- Importing `brain` or `AlphaBrain` must have no side effects: no audio, clients or models start until `start()`.
- Library modules log under `alpha.*` namespaces and don't configure global logging.
- Commit messages are short Chinese summaries, often prefixed with a milestone tag such as `J9：`, `D2：` or `P1：`.
- `data/`, `.env`, `credentials*.json`, `token.json`, `youtube_token.json` and `esp-alpha/` (except `esphome/alpha-xvf.yaml`) are git-ignored.
