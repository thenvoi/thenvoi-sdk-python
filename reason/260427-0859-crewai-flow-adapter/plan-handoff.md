# Plan Handoff: CrewAIFlowAdapter

The reason loop converged on a cautious yes: build an experimental `CrewAIFlowAdapter`, keep `CrewAIAdapter` as the default, and scope v1 to message-scoped orchestration.

The implementation should not model CrewAI Flow as a long-lived external-event subscriber. `SimpleAdapter.on_event` awaits `on_message`, so v1 should run a Flow during one inbound message, emit any delegation side effects, persist pending state through task events, and return. Later peer replies arrive as normal messages and trigger another `on_message` call, where the adapter reconstructs state from task events and decides whether to synthesize.

Suggested implementation shape:

1. Add `CrewAIFlowAdapter` as a separate class and export it without changing `CrewAIAdapter` behavior.
2. Add a flow-state converter that scans history for task events under a metadata namespace such as `crewai_flow`.
3. Define metadata records for `run_id`, `delegation_id`, `room_id`, `parent_message_id`, target participant/mention, status, reply message id if available, and finalization status.
4. In `on_message`, reconstruct state from history, derive or load the current `run_id`, execute the Flow for the current inbound message, and emit task events for state transitions.
5. Use `send_message` only for visible room messages and delegation messages. Use `send_event(..., message_type="task", metadata=...)` for coordination metadata.
6. Avoid indefinite waits inside `on_message`. A delegated-and-waiting outcome is a completed turn with pending task-event state.
7. Add idempotency checks before sending delegation messages and before final synthesis.
8. Treat per-room locks as local protection only. Do not claim distributed exactly-once semantics without platform support.

Initial tests should cover direct response, single delegation, multi-delegation join, duplicate inbound retry, restart reconstruction from task events, later peer reply re-entry, cleanup behavior, and the explicit non-support of suspended waits.
