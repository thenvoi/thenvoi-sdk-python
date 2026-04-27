---
id: TASK-002
phase: 1
status: completed
depends-on: []
---

# TASK-002: Add the orchestration state contract and state source

## Objective
Add the versioned metadata models, converter, and state-source protocol that reconstruct CrewAI Flow state from task events on every inbound message.

## Spec reference
> **Goal:** Add the versioned metadata models, converter, and state-source protocol that reconstruct CrewAI Flow state from task events on every inbound message.
>
> This phase is purely additive and creates no adapter. It proves state reconstruction does not depend on bootstrap history.

## Changes
- [ ] `src/thenvoi/converters/crewai_flow.py` — create the Pydantic models from the [decomposition table](../technical-spec.md#pydantic-model-decomposition): `CrewAIFlowMetadata`, `CrewAIFlowDelegationState`, `CrewAIFlowSequentialChainState`, `CrewAIFlowBufferedSynthesis`, `CrewAIFlowError`, `CrewAIFlowParticipantSnapshot`, plus the aggregate `CrewAIFlowSessionState`.
- [ ] `src/thenvoi/converters/crewai_flow.py` — create StrEnums `CrewAIFlowRunStatus`, `CrewAIFlowStage`, `CrewAIFlowDelegationStatus`, `CrewAIFlowJoinPolicy`, `CrewAIFlowTextOnlyBehavior`.
- [ ] `src/thenvoi/converters/crewai_flow.py` — implement `CrewAIFlowStateConverter.convert(raw)` applying the merge semantics from [Merge semantics](../technical-spec.md#merge-semantics), including terminal-state absorption.
- [ ] `src/thenvoi/converters/crewai_flow.py` — implement `normalize_participant_key` and `CrewAIFlowAmbiguousIdentityError`.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — define the `CrewAIFlowStateSource` protocol with `async load_task_events(*, room_id, metadata_namespace, tools, history)`.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `RestCrewAIFlowStateSource`. Read items by paginating `tools.fetch_room_context(room_id=room_id, page=N, page_size=100)` (the new `AgentToolsProtocol` method from [Protocol additions](../technical-spec.md#protocol-additions)). Filter by `message_type == "task"` and presence of `metadata[metadata_namespace]`. Sort by `(inserted_at, message_id)` ascending. Maintain the LRU cache from [Caching and read amplification](../technical-spec.md#caching-and-read-amplification). On REST failure, retry once with backoff, then raise `ThenvoiToolError`.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `HistoryCrewAIFlowStateSource(*, acknowledge_test_only: bool)`. Constructor raises `ThenvoiConfigError` if `acknowledge_test_only` is not exactly `True`. On first non-bootstrap use (history empty), emits a single `WARNING`-level log: `"HistoryCrewAIFlowStateSource: AgentInput.history is empty on a non-bootstrap turn. State will be lost. If you see this in production, switch to RestCrewAIFlowStateSource."` Used only by tests and intentional bootstrap-only deployments; never the default.
- [ ] `src/thenvoi/runtime/tools.py` — add `AgentTools.fetch_room_context(*, room_id, page=1, page_size=50)` calling `self.rest.agent_api_context.get_agent_chat_context(chat_id=room_id, page=page, page_size=page_size)` and returning `{"data": [<message dict>...], "meta": {...}}`. Reuse the Fern-to-dict shim shape from `src/thenvoi/runtime/execution.py:551-567` (extract a small helper into `src/thenvoi/runtime/_context_serialization.py` and import from both `execution.py` and `tools.py`; do not duplicate).
- [ ] `src/thenvoi/core/protocols.py` — add `fetch_room_context` to `AgentToolsProtocol`.
- [ ] `src/thenvoi/testing/fake_tools.py` — implement `FakeAgentTools.fetch_room_context` returning a configurable list of message dicts. Constructor accepts `room_context: list[dict] | None = None`; if set, the method paginates over that list.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — leave `CrewAIFlowAdapter` as a stub class for later phases so the test fixtures in this phase can import the state source without depending on Phase 2 work.
- [ ] `tests/converters/test_crewai_flow.py` — fixtures: empty history, unrelated task events, reservations, pending delegations, replies, ambiguous replies, buffered syntheses, finalized, failed, indeterminate, malformed metadata, duplicate normalized keys, REST read failure, non-bootstrap peer-reply replay, newest-event merge.
- [ ] `tests/converters/test_crewai_flow.py` — restart fixture that reconstructs state from raw task-event dicts without adapter memory.
- [ ] `tests/adapters/test_crewai_flow_state_source.py` — non-bootstrap peer-reply fixture where `AgentInput.history` is empty but `RestCrewAIFlowStateSource` returns prior task events from a `FakeAgentTools` whose `fetch_room_context` returns a preloaded list. Plus a fixture asserting cache early-termination behavior on a second turn.

## Acceptance criteria
- [ ] Unit: `tests/converters/test_crewai_flow.py` covers all v1 fields, reservation states, terminal absorption, merge rules, normalization.
- [ ] Unit: non-bootstrap replay uses `RestCrewAIFlowStateSource` with the fake REST client, never `AgentInput.history`.
- [ ] Unit: REST failure surfaces `ThenvoiToolError`, not silent empty state.
- [ ] Pass criterion: `uv run pytest tests/converters/test_crewai_flow.py tests/adapters/test_crewai_flow_state_source.py -v`
- [ ] Acceptance criterion 11 from spec: `src/thenvoi/converters/crewai_flow.py` exports the v1 models, enums, `normalize_participant_key`, and `CrewAIFlowAmbiguousIdentityError`.
- [ ] Acceptance criterion 12: `src/thenvoi/adapters/crewai_flow.py` defines `CrewAIFlowStateSource`, `RestCrewAIFlowStateSource`, and `HistoryCrewAIFlowStateSource(*, acknowledge_test_only)`.
- [ ] Acceptance criterion 13: `src/thenvoi/core/protocols.py` adds `fetch_room_context` to `AgentToolsProtocol`. `src/thenvoi/runtime/tools.py` implements it on `AgentTools`. `src/thenvoi/testing/fake_tools.py` implements it on `FakeAgentTools`. The Fern-to-dict shim is shared with `ExecutionContext.hydrate()` (no duplication).
- [ ] Acceptance criterion 14: `CrewAIFlowStateConverter.convert([])` returns an empty `CrewAIFlowSessionState`.
- [ ] Acceptance criterion 15: Task events without `metadata["crewai_flow"]` are ignored. Task events under a different `metadata_namespace` are ignored.
- [ ] Acceptance criterion 16: v1 task events reconstruct `observed`, `side_effect_reserved`, `delegated_pending`, `waiting`, `reply_recorded`, `reply_ambiguous`, `finalized`, `failed`, and `indeterminate`.
- [ ] Acceptance criterion 17: Two task events with identical `inserted_at` apply in `message_id` order.
- [ ] Acceptance criterion 18: Later events for the same `run_id` apply merge semantics; `finalized`, `failed`, `indeterminate` are absorbing. Performance test verifies terminal-run short-circuit on a 1,000-event log.
- [ ] Acceptance criterion 19: Malformed metadata produces a converter warning and a failed-state record without crashing conversion.
- [ ] Acceptance criterion 20: A non-bootstrap peer-reply test reconstructs pending state from `RestCrewAIFlowStateSource` while `AgentInput.history` is empty.
- [ ] Acceptance criterion 21: A non-bootstrap peer-reply test where `tools.fetch_room_context` raises records `failed` with `state_source_unavailable` and sends no visible message.
- [ ] Acceptance criterion 22: `RestCrewAIFlowStateSource` cache: on the second turn for the same room, only events with `inserted_at > latest_inserted_at_seen` are added. Verified by counting `tools.fetch_room_context` calls.
- [ ] Acceptance criterion 23: `HistoryCrewAIFlowStateSource()` raises `ThenvoiConfigError`. With `acknowledge_test_only=True` succeeds. First non-bootstrap use with empty history emits one WARNING-level log message.
- [ ] Acceptance criterion 24: A non-terminal run whose `parent_message_id` timestamp is older than `now - max_run_age` closes with `failed` / code `run_aged_out` on the next turn.
- [ ] Acceptance criterion 25: Required task-event write failures (reservation, waiting, ambiguous) stop the turn after up to 2 retries.

## Out of scope
- The `CrewAIFlowAdapter` constructor and lifecycle (Phase 2).
- Flow execution (Phase 3).
- Delegation, reply matching, or join handling (Phase 4).
