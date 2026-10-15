# Execution, cancellation and undo

The production Qwen voice flow uses `core/simple_coordinator.py` and
`voice/simple_receiver.py`. The CLI/Claude flow is unchanged.

```sh
.venv/bin/python -m voice.qwen_realtime
```

## Execution

HA on/off/brightness actions execute immediately. Successful device actions are
silent. Reads also execute immediately. Personal additions, task completion and
reminders remain pending until two seconds of silence have elapsed and code has
received every tool result in the response. No `finish_collection` call is
required. Multiple ordinary tasks do not trigger a plan or confirmation.

Speech onset pauses submission. Noise resumes the previous window. Meaningful
speech waits for the new response to finish. Simple unfinished-expression rules
extend the delay to five seconds and prompt for clarification. Configure
`COLLECTION_WINDOW_S=2` and `COLLECTION_INCOMPLETE_S=5`.

The coordinator lives outside WebSocket connections. Already-running operations
can finish during reconnection. Unsent work is paused after reconnect until a
new mutation or explicit execution request. After reconnecting, an idle receiver
asks once, “刚才断线了，还有 N 项操作没执行，要继续吗？” without releasing
the window. Say “继续” / “继续执行” / “执行吧” to resume, or request cancellation.
There is no prompt when no operations are waiting. Reports wait until the receiver can
request another response. Tool work does not block receiving audio events.

## Three local controls

`cancel_pending()` cancels all unsent operations. It cannot recall a request
already sent; that request keeps its actual result.

`undo_recent(title=...)` deletes an item or reminder created by Alpha during this
program session, using its exact saved IDs. No extra confirmation is needed.
When several targets are possible, the user must identify one. Reading a list
does not confer ownership, and ownership does not survive a process restart.
It does survive WebSocket reconnection.

`delete_item(kind, item_id, list_id, confirmed)` handles exact existing resources.
The first call verifies the target and asks for confirmation without deleting.
The next call needs both `confirmed=true` and a new meaningful utterance that
began after the confirmation request. Confirmation is bound to that exact kind,
list and item. Noise and the same utterance cannot satisfy this requirement.
The model remains responsible for understanding explicit agreement; a question
or ambiguous statement should not produce `confirmed=true`.

No version, playback-complete signal or transcript comparison is required.
Reconnect clears outstanding deletion confirmation requests. An exact target
created in this program session uses the immediate undo rule instead.

## IDs, facts and uncertain results

`list_items` returns readable text and structured list/task IDs. `list_reminders`
returns upcoming main-calendar events with IDs. These IDs are for tool arguments,
not spoken output. Creation receipts retain the exact new task/event IDs.
Internal delete/inspect tools remain behind the coordinator and Gateway.

Gateway checks policies and records every operation in SQLite. Undo records a
new deletion operation and preserves the original creation fact. Repeated
executing, unknown or successful deletions are not blindly resent. Unknown
creation/deletion results can be checked against operation markers or exact IDs;
missing evidence does not prove a creation failed.

On startup, leftover executing operations become unknown and discarded pending
work is cancelled. Execution records survive restart. Shutdown cancels unsent
operations; interrupted in-flight requests remain unknown.

## Retained legacy code

The previous task-batch/plan modules and offline regression paths remain in the
repository. Production `main` creates `SimpleCoordinator`, without dependency
controls, mandatory collection tools, action-count confirmations, recap playback
gates, or fallback `say` narration.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

Tests use fake WebSockets/workers, direct player callbacks, fake Google services,
and in-memory or temporary SQLite. They do not use real credentials, audio
hardware, paid models, or Google/HA requests. Real microphone and model tests
remain necessary to assess speech recognition and conversational behavior.
