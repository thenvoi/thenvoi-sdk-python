# Reason Lineage

## Round 1

Initial candidate: keep the current `CrewAIAdapter` as the default and add an experimental Flow adapter only for explicit per-room orchestration. The candidate identified source-of-truth ambiguity, live-room/Flow semantic mismatch, duplicate synthesis, late replies, and two-adapter user confusion.

Critique: the initial candidate was directionally right but too loose. It did not define re-entrancy, persistence, delegation correlation keys, adapter contract changes, or whether ContextVars would survive persisted/background Flow execution.

Revised candidate: add a separate experimental `CrewAIFlowAdapter`. Thenvoi owns state and correlation; Flow is a derived execution graph. Use explicit runtime injection, run/delegation IDs, per-room serialization, visible terminal states, and an event-to-flow-state layer. Do not reuse the existing CrewAI converter unchanged.

Synthesis: keep the existing adapter stable and add an opt-in Flow adapter for deterministic routing/delegation/join/synthesis. Treat Flow state as derived, not canonical. Promote only after deterministic routing, isolation, late-event, cleanup, dependency, and restart behavior are proven.

Judge result: 3 votes for the concise synthesis, 2 votes for the more detailed revised candidate. The judges agreed on the direction but split on how much implementation specificity belonged in the decision.

## Round 2

Critique of the Round 1 winner: it still assumed a long-lived Flow could wait for external events. The critic asked for proof against CrewAI lifecycle and Thenvoi's adapter contract, and called out `delegated-and-waiting` as potentially conflicting with an awaited `on_message` turn.

Code-grounding pass:

- `SimpleAdapter.on_event` awaits `on_message` directly.
- Framework adapters process messages only.
- `AgentTools.send_message` has no metadata parameter.
- `AgentTools.send_event` does support metadata.
- A2A and Letta already reconstruct external session state from task event metadata.

Final candidate: build `CrewAIFlowAdapter`, but Flow v1 runs per inbound message. Delegation sends messages, records pending state via task event metadata, and returns. Later peer replies re-enter normal `on_message`; the adapter reconstructs pending state from task events, records reply state, and synthesizes when the join policy is satisfied. Do not suspend `on_message` waiting for future events. Use `run_id` and `delegation_id` for idempotency. Treat local locks as local only; distributed exactly-once needs platform support.

Judge result: 5-0 for the code-grounded candidate. Convergence achieved.
