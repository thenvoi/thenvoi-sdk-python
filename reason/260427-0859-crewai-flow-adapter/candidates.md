# Final Candidates

## Round 1 Winner

Add an experimental second `CrewAIFlowAdapter`, not a replacement. The current `CrewAIAdapter` remains default. The Flow adapter handles deterministic orchestration; Thenvoi is the source of truth, and Flow state is derived. Configuration includes a flow factory, optional state model, routing policy, delegation timeout, synthesis, restart/fail-safe behavior, explicit runtime object instead of ContextVar, correlation metadata or pending records, per-room serialization, visible terminal states, and a custom event-to-flow-state layer. Promote after tests prove deterministic routing, two-room isolation, overlapping turns, late events, missing runtime handling, cleanup, dependency compatibility, and restart/resume.

## Final Winner

Build a new `CrewAIFlowAdapter`, but only if Flow is the local execution plan for one inbound Thenvoi message, not a long-lived subscription mechanism.

The current adapter contract awaits `on_message`, so v1 should not suspend inside `on_message` waiting for future peer replies. If the Flow delegates, it sends messages and records pending state via `send_event(..., message_type="task", metadata=...)`, because `send_message` cannot carry metadata. The adapter then returns. Later peer replies arrive as normal messages and trigger new `on_message` calls. The adapter reconstructs pending state from task events, classifies the reply, records updated task-event state, and synthesizes when the join policy is satisfied.

Durable state belongs in Thenvoi task events first. CrewAI Flow persistence may be used internally during a turn, but it should not become the canonical state for room orchestration. Use stable `run_id` and `delegation_id` values for idempotency. Before sending delegation or final synthesis, reconstruct state and check whether that side effect already happened.

In-process room locks are allowed only as local duplicate-work protection. They are not distributed coordination. Multi-worker exactly-once behavior should be deferred until the platform has idempotency keys, conditional append, advisory locks, or a room-run state endpoint.

V1 supports message-scoped Flow execution, deterministic delegation planning, task-event-backed pending state, later re-entry through normal messages, configurable join policy, direct-answer fallback, and idempotent delegation/finalization. V1 does not support paused Flow execution across external events, cross-room joins, distributed locking, indefinite waits inside `on_message`, or metadata attached directly to chat messages.
