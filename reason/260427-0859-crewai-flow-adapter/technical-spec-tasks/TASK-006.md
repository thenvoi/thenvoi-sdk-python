---
id: TASK-006
phase: 5
status: pending
depends-on: [TASK-005]
---

# TASK-006: Implement router safety policies

## Objective
Implement the [Router safety policies](../technical-spec.md#router-safety-policies) as typed state transitions on top of Phase 4.

## Spec reference
> **Goal:** Implement the [Router safety policies](../technical-spec.md#router-safety-policies) as typed state transitions on top of Phase 4.

## Changes
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `tagged_peer_policy="require_delegation_before_final"`. Detection: extract `@{handle}` tokens from the inbound message text, normalize each, intersect with the room participant snapshot. Block finalization until each detected handle has a recorded delegation.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `sequential_chains` selectors resolved against the structured participant snapshot at `on_message` time. When the upstream key replies, finalization is blocked until the downstream key has a recorded delegation.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `buffered_syntheses` for partial synthesis text produced before the join is satisfied; concatenate (in order of `source_message_id` arrival) when the join completes and the run finalizes.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `text_only_behavior` for malformed text-only output exactly per the [Malformed output handling](../technical-spec.md#malformed-output-handling) section.
- [ ] `tests/adapters/test_crewai_flow_adapter.py` — ordered-trace fixture for parallel fan-out: two delegations are reserved and sent, the first reply records replied state with no final visible message, the second reply triggers one final reservation and one final visible message.
- [ ] `tests/adapters/test_crewai_flow_adapter.py` — ordered-trace fixture for sequential composition: parent request, upstream delegation reservation, upstream visible delegation, waiting with no final, upstream reply via state-source replay, downstream delegation reservation, downstream visible delegation, downstream reply, one final reservation, one final visible message.
- [ ] `tests/adapters/test_crewai_flow_adapter.py` — **end-to-end fixture covering the full v1 user promise**: turn 1 parent request → two delegations reserved+sent (turn 1) → no final → turn 2 first peer reply → reply recorded, no final → turns 3-5 unrelated chatter (no state mutation) → turn 6 second peer reply → join satisfied → exactly one final reservation + visible message → turn 7 a duplicate of the original parent request from a different user message id is treated as a new run (separate `run_id`) → turn 8 a late reply from one of the original peers (after `finalized`) is discarded per rule 8 with no second final. Test asserts the exact ordered list of visible messages and task events matches a hand-written fixture file `tests/fixtures/crewai_flow_e2e_trace.json`.
- [ ] `tests/adapters/test_crewai_flow_adapter.py` — tagged-peer enforcement, event-only waiting, identity normalization, buffered synthesis, malformed text output, non-bootstrap replay tests.

## Acceptance criteria
- [ ] Unit: tagged peer cannot finalize before the tagged peer has a delegation event.
- [ ] Unit: under `sequential_chains={"upstream": "downstream"}`, an upstream reply blocks finalization until downstream has a delegation.
- [ ] Unit: parallel fan-out under `join_policy="all"` waits for both replies before finalizing.
- [ ] Unit: waiting decisions after delegation emit no visible message.
- [ ] Unit: UUID, namespaced handle, bare handle, and display-name forms all resolve to the same normalized key for the same participant.
- [ ] Unit: malformed text-only output records failed by default.
- [ ] Unit: ordered-trace fixtures for parallel and sequential scenarios match the expected visible-message and task-event sequence exactly.
- [ ] **Integration: the end-to-end ordered-trace fixture (`tests/fixtures/crewai_flow_e2e_trace.json`) — full multi-turn run covering parent request, two parallel delegations, two replies on different turns, one final synthesis, one duplicate-request that becomes a new run, and one stale peer reply that is discarded — matches the recorded fixture exactly.**
- [ ] Pass criterion: `uv run pytest tests/adapters/test_crewai_flow_adapter.py -v -k "safety or sequential or tagged or identity or ordered_trace or e2e_trace"`
- [ ] Acceptance criterion 64 from spec: With `tagged_peer_policy="require_delegation_before_final"`, a tagged-peer request cannot finalize before the tagged peer has a delegation event.
- [ ] Acceptance criterion 65: With `sequential_chains={"upstream": "downstream"}`, an upstream reply blocks finalization until the downstream key has a delegation.
- [ ] Acceptance criterion 66: Parallel two-target fixtures finalize only after both replies under `join_policy="all"`.
- [ ] Acceptance criterion 67: Waiting decisions after delegation emit no visible message.
- [ ] Acceptance criterion 68: UUID, namespaced handle, bare handle, and display-name forms normalize to the same participant key when they refer to the same room participant.
- [ ] Acceptance criterion 69: Malformed text-only output records failed by default.
- [ ] Acceptance criterion 70: Ordered-trace fixtures for parallel and sequential scenarios match the expected visible-message and task-event sequence exactly. The adapter passes these fixtures without monkeypatching `AgentTools` or `CrewAIAdapter`.
- [ ] Acceptance criterion 71: End-to-end ordered-trace fixture covering the full v1 user promise matches `tests/fixtures/crewai_flow_e2e_trace.json` exactly.
- [ ] Acceptance criterion 72: `uv run pytest tests/adapters/test_crewai_flow_adapter.py -v -k "safety or sequential or tagged or identity or ordered_trace or e2e_trace"` passes.

## Out of scope
- Public registration (Phase 6).
- Runnable example documentation (Phase 7).
