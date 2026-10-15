# Automatic batches, deletion and correction

The production Qwen flow uses `core/simple_coordinator.py` and
`voice/simple_receiver.py`. CLI/Claude behavior is unchanged.

```sh
.venv/bin/python -m voice.qwen_realtime
```

## Collect, execute, report

HA on/off/brightness executes immediately and clean successes stay silent.
Reads execute immediately. Other writes, including deletion, undo and restoration,
collect in a two-second silence window. User instructions authorize execution;
there is no additional approval, plan version, playback qualification or
`confirmed` parameter. An unclear target still requires clarification.

Code waits for the response and all its tools to finish, then seals a batch when
the silence window expires. Each batch executes one operation at a time and
produces one factual report. The next sealed batch starts after the current
report completes. While execution is running, new instructions can be understood
and collected into later batches; they cannot change an already-sent request.

All-pending tool responses do not trigger a model acknowledgment. Their outputs
are stored in conversation and the model resumes for the actual execution report.
Queries and rejected calls can continue normally. Success, failure and unknown
results are reported from operation facts, rather than inferred from promises.
If a completed response promises to perform an action without a tool call,
code withholds its buffered audio and requests a tool correction. This shares
the existing completion-check retry budget: at most one automatic correction per
user utterance. The phrase detector is a heuristic, not a semantic guarantee.
It never executes a deletion merely because a phrase matched.
Simple shopping phrases have a deterministic entry point: “买牛奶” registers
`add_items` for `购物清单`, while “购物清单里有什么” and
“现在购物清单里有什么” call `list_items` for fresh data. These routes do not
depend on the model deciding to call a tool. Local queries run as session-managed
background tasks so speech events remain responsive; a superseded query does
not produce a late answer. The model receives the actual query result for speech.
Ambiguous, negated, interrogative or multiple-action purchase phrases use normal
interpretation. Business operations still pass through Gateway and the write window.
Startup prints the loaded source path and “本地购物路由 v1”.
Completion detection includes deletion, removal, restoration and cancellation,
and permits up to 12 intervening characters after “已/已经”, e.g.
“面包已从购物清单里撤掉了”. Negated statements are excluded.
It also recognizes “已加到 / 已加进 / 已加入”. Ordinary answers and execution
corrections both hold audio until their final text passes the same checks;
correction instructions alone do not make a response trusted.
If that retry again promises work or falsely claims completion without tools,
the client logs a factual failure and requests one final spoken status notice.
This notice cannot execute tools and cannot start another retry cycle. Pending,
successful or unknown operations receive their actual status rather than an
incorrect instruction to repeat a possibly effective write. Notice audio still
comes from Qwen; the deterministic terminal message does not depend on its wording.

A cancelled report remains active and is sent again after the new input is
handled. The next batch remains blocked until a report response completes.
All tools requested inside a report response are blocked in code, including local
undo/cancel controls and calls found only in `response.done.output`. Their outputs
state that no tool was executed and include the original batch's actual facts.
A tool-only report does not count as delivered. One factual re-report is allowed;
if the model again requests tools, code prints the facts as a terminal fallback
and releases the next batch, rather than looping or replaying an operation.
Here completion means a successful, uninterrupted `response.done`, not physical
speaker drain. A regression test covers interruption, re-reporting and subsequent
batch execution without replaying the first batch.

Speech pauses collection; noise resumes it. Unfinished-expression rules extend
the window and ask for clarification. Configure `COLLECTION_WINDOW_S=2` and
`COLLECTION_INCOMPLETE_S=5`. Silence starts at speech end, not tool completion.

## Four local controls

- `cancel_pending()` cancels all unsent work, including queued batches and the
  unsent remainder of the executing batch. In-flight requests retain real results.
- `undo_recent(title=...)` registers deletion of an exact resource Alpha created
  in this program session. It uses saved creation IDs and needs no confirmation.
- `delete_item(kind, item_id, list_id)` registers deletion of an exact queried
  task or calendar event. It does not ask the user to say “yes”. Multiple possible
  targets need clarification, rather than an invented ID.
- `restore_recent(title=...)` registers restoration of a recently deleted resource
  from its saved snapshot, without confirmation. Unknown deletion outcomes must
  be verified before restoration; unknown restoration is not blindly repeated.

Deletion, undo and restoration use the same automatic batch queue as additions.
For the exact phrase “刚才那个不要了” (and a small set of equivalents), code
registers undo directly if the latest mutation is a successful single-resource
creation with a saved exact ID. It bypasses model confirmation. Multiple targets,
missing receipts or a more recent different mutation use normal interpretation.
The model must call the tool rather than promise “I'll delete it”. Before execution,
pending results cannot be described as completed.

## Delete snapshots and recovery limits

Immediately before deleting, code queries the exact resource and persists the
full snapshot in SQLite's `deletion_backups` table. A backup failure prevents
sending the deletion. Cancellation during backup also prevents deletion. Final
operation results do not overwrite backups.

Restoration creates a new Google resource, with a new ID. Tasks retain title,
notes, due date and completion fields (and parent where still valid), without
promising original position or external links. Ordinary events retain title,
description, location, start/end, popup reminders and supported display settings.
Automatic deletion rejects recurring/special events and events containing
participants or conferencing because this restoration path cannot faithfully
recreate those semantics. Rejection is a factual failure, not a confirmation gate.

Snapshots and execution facts survive restart. The voice undo/restore selectors
currently reference operations from this program session, surviving WebSocket
reconnect but not rebuilding selectors after process restart. Internal raw
restore/delete/inspect tools are not exposed directly to the model.

## Connection and lifecycle

Model snapshots include compact pending state and the last three creation/deletion
records, with essential IDs and shortened result text. Full timestamps, nested
results and resource backups stay in SQLite rather than being repeated to the
model. An unchanged snapshot is not appended again for every `response.create`.
Time context is still injected. This reduces future input growth; it does not
delete conversation items already stored by the server or cap total history.

The coordinator and queued work live outside a WebSocket connection. In-flight
requests can complete during reconnect. Unsent work pauses with a recovery prompt
and resumes on a new mutation or explicit “继续” / “继续执行” / “执行吧”.
Reports remain queued until deliverable; interrupted reports are retained.
Microphone queue overflow discards buffered audio and ends only the current
session with a reconnectable error. Each audio send also has a two-second timeout,
so a stalled WebSocket cannot leave the sender waiting indefinitely. The old
stream is closed and its queued audio is never replayed into the new connection.
This recovery does not replay business-tool operations.
Reconnect logs flatten nested task-group exceptions and show connection duration.
Audio queue errors also show received/sent block counts and maximum callback
dispatch delay; send timeouts show their limit and queued blocks. These distinguish
the recovery trigger from the generic task-group wrapper, without asserting that
all congestion is a network fault.

Startup marks leftover executing operations unknown and cancels discarded unsent
operations. It does not replay writes. Shutdown cancels unsent operations and
marks interrupted in-flight results unknown.

Legacy plan/batch modules remain for regression tests. Production does not use
those confirmation, dependency, recap comparison or fallback TTS paths.

## Validation

`QWEN_VAD=server_vad` and `QWEN_SILENCE_MS=800` configure acoustic VAD (the
defaults). `semantic_vad` remains selectable. The actual server-echoed type and
silence are verified and printed. Test short answers and a one-second pause using
`python -m devtools.qwen_vad_probe`; it never executes business tools.

An online `qwen3.8-omni-flash-realtime` probe tried both
`response.create.response.tool_choice="required"` and top-level
`response.create.tool_choice="required"`. Neither request returned an error,
but both returned a greeting without the available no-op function. This does not
establish enforced tool calling, so neither form is enabled in production.
`python -m devtools.qwen_tool_choice_probe` reproduces the isolated check without
MCP or microphone access. It uses the configured API account and incurs model usage.

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

Tests use fake workers/WebSockets/Google services, direct audio callbacks and
in-memory or temporary SQLite. No real credentials, devices or paid models are
used. Live speech/model adherence and service latency still require manual tests.

Google API references: [Tasks resource](https://developers.google.com/tasks/reference/rest/v1/tasks),
[Calendar event insertion](https://developers.google.com/workspace/calendar/api/v3/reference/events/insert).
