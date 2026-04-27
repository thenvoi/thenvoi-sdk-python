---
id: TASK-004
phase: 3
status: completed
depends-on: [TASK-001, TASK-002, TASK-003]
---

# TASK-004: Execute message-scoped Flows

## Objective
Run one Flow per inbound message. Support `direct_response`, `waiting`, `failed`, and malformed output. Delegation is in Phase 4.

## Spec reference
> **Goal:** Run one Flow per inbound message. Support `direct_response`, `waiting`, `failed`, and malformed output. Delegation is in Phase 4.

## Changes
- [ ] `src/thenvoi/adapters/crewai_flow.py` — define `CrewAIFlowRuntimeTools` per the [API table](../technical-spec.md#crewaiflowruntimetools-api) above. `create_crewai_tools(*, capabilities=None, custom_tools=None)` builds a `CrewAIFlowSubCrewReporter` (the `CrewAIToolReporter` subclass that bridges to `SideEffectExecutor`, see [Sub-Crew side-effect ordering](../technical-spec.md#sub-crew-side-effect-ordering)) and calls `thenvoi.integrations.crewai.tools.build_thenvoi_crewai_tools` from Phase 0 with that reporter, plus a context getter that returns the current Flow runtime's `CrewAIToolContext`.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — define `CrewAIFlowSubCrewReporter`. It holds the active `SideEffectExecutor` reference and a per-`run_id` counter. `report_call(tools, name, input)` is the hook where the reporter intercepts `thenvoi_send_message` calls, derives the sub-Crew `side_effect_key`, and routes through the executor.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — define a private module-level `ContextVar[CrewAIFlowRuntimeTools | None]` and the `get_current_flow_runtime()` helper.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement the message processing pipeline: acquire per-room lock; load task events from `CrewAIFlowStateSource`; convert to `CrewAIFlowSessionState`; build the input dict from the message, state, participant snapshot, agent metadata, and participants/contacts messages; call `flow_factory()` once per turn (wrapped in try/except — any exception from `flow_factory()` records `failed` with code `flow_factory_error`, emits an error event, and returns); set the runtime `ContextVar` for the duration of `await flow.kickoff_async(inputs)`; clear the `ContextVar` in `finally`; release the lock.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — define Pydantic discriminated-union models for the [Decision shape](../technical-spec.md#decision-shape). Validate Flow output against them before any side effect.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `direct_response`, `waiting`, `failed`, malformed-output handling using the reserve-send-confirm sequence. `FlowStreamingOutput` records `failed`.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — define `SideEffectExecutor` as a private class. It is the only holder of a reference to `tools.send_message`. `CrewAIFlowRuntimeTools` does not import or wrap it.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — per-room `asyncio.Lock` cache. On `on_cleanup(room_id)` the lock entry is removed.
- [ ] `tests/adapters/test_crewai_flow_adapter.py` — direct-response, waiting, failed, malformed-output, `FlowStreamingOutput` rejection, raw-tool-bypass rejection, duplicate-finalization, flow-factory-exception, event-bus-listener-bound, nest-asyncio-not-invoked, sub-Crew tool routing tests.

## Acceptance criteria
- [ ] Unit: mocked Flow returns `direct_response` → exactly one final reservation, one visible message, one finalized task event.
- [ ] Unit: mocked Flow returns `waiting` → no visible message, one waiting task event.
- [ ] Unit: mocked Flow returns malformed shape → error event + failed task event.
- [ ] Unit: mocked Flow returns `FlowStreamingOutput` → failed; no visible message.
- [ ] Unit: `flow_factory()` raises `RuntimeError("eager init failed")` → adapter records `failed` with code `flow_factory_error`, emits an error event, and `on_message` returns cleanly without propagating.
- [ ] Unit: 100 sequential `kickoff_async` cycles in the same process do not grow `crewai_event_bus`'s registered listener count beyond a small bounded set (assert listener count after 100 turns ≤ listener count after 10 turns + small constant).
- [ ] Unit: importing `CrewAIFlowAdapter` and running one `direct_response` turn does NOT invoke `nest_asyncio.apply` (verified by patching `nest_asyncio.apply` and asserting it is never called).
- [ ] Unit: `CrewAIFlowRuntimeTools` does not expose `send_message`, `send_event`, `add_participant`, or `remove_participant`. (Test introspects the public attribute set.)
- [ ] Unit: `runtime.create_crewai_tools()` returns a `list[BaseTool]` whose `thenvoi_send_message` invocations write a `subcrew:{counter}` reservation event, send through the `SideEffectExecutor`, and write a `subcrew:{counter}` confirmation event. Verified by patching the executor and asserting the side_effect_key format and the surrounding event sequence.
- [ ] Unit: a Flow that mutates `self.state` mid-`kickoff_async` and returns a decision dict — the next turn's reconstructed state ignores the `self.state` mutation entirely; only the persisted task events are reflected.
- [ ] Unit: same inbound message with already-finalized state does not send a second final response.
- [ ] Pass criterion: `uv run pytest tests/adapters/test_crewai_flow_adapter.py -v -k "direct or waiting or malformed or streaming or runtime_tools or idempotent or flow_factory or event_bus or nest_asyncio or subcrew or flow_state"`
- [ ] Acceptance criterion 35 from spec: A `direct_response` decision sends one visible message and records `status=finalized`.
- [ ] Acceptance criterion 36: A `waiting` decision sends no visible message and records `status=waiting`.
- [ ] Acceptance criterion 37: A `failed` decision emits an error event and records `status=failed`.
- [ ] Acceptance criterion 38: Malformed output with default `text_only_behavior="error_event"` emits an error event and records `status=failed`.
- [ ] Acceptance criterion 39: `FlowStreamingOutput` is rejected and records `status=failed`.
- [ ] Acceptance criterion 40: `flow_factory()` raising `RuntimeError` records `status=failed` with code `flow_factory_error`, emits an error event, and `on_message` returns cleanly without propagating.
- [ ] Acceptance criterion 41: Flow input contains only JSON-serializable state. `AgentTools` is reachable only through `get_current_flow_runtime()` during the active call.
- [ ] Acceptance criterion 42: A Flow that mutates `self.state` mid-`kickoff_async` — next turn ignores the mutation.
- [ ] Acceptance criterion 43: `CrewAIFlowRuntimeTools` does not expose `send_message`, `send_event`, `add_participant`, `remove_participant`. `create_crewai_tools()` is exposed.
- [ ] Acceptance criterion 44: `runtime.create_crewai_tools()` returns a `list[BaseTool]`. A sub-Crew tool invocation of `thenvoi_send_message` writes a reservation event with key `{run_id}:subcrew:{counter}`, sends through `SideEffectExecutor`, and writes a confirmation event with the same key.
- [ ] Acceptance criterion 45: 100 sequential `kickoff_async` cycles do not grow `crewai_event_bus` listener count beyond a small bounded set.
- [ ] Acceptance criterion 46: Importing `CrewAIFlowAdapter` and running one `direct_response` turn does NOT invoke `nest_asyncio.apply`.
- [ ] Acceptance criterion 47: A retry whose reconstructed state already shows `status=finalized` for the same `run_id` sends no second final message.
- [ ] Acceptance criterion 48: `on_message` returns after one local `kickoff_async(inputs)` call.
- [ ] Acceptance criterion 49: `uv run pytest tests/adapters/test_crewai_flow_adapter.py -v -k "direct or waiting or malformed or streaming or runtime_tools or idempotent or flow_factory or event_bus or nest_asyncio or subcrew or flow_state"` passes.

## Out of scope
- Delegation (Phase 4).
- Reply matching and joins (Phase 4).
- Router safety policies — tagged peer, sequential chains, buffered syntheses (Phase 5).
- Public registration (Phase 6).
