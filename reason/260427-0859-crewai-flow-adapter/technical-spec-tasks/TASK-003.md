---
id: TASK-003
phase: 2
status: completed
depends-on: []
---

# TASK-003: Add the adapter API skeleton

## Objective
Promote the Phase 1 stub into `CrewAIFlowAdapter` with the full constructor contract, validation, and SimpleAdapter lifecycle methods. No Flow execution yet.

## Spec reference
> **Goal:** Promote the Phase 1 stub into `CrewAIFlowAdapter` with the full constructor contract, validation, and SimpleAdapter lifecycle methods. No Flow execution yet.

## Changes
- [ ] `src/thenvoi/adapters/crewai_flow.py` — import `Flow`, `start`, `listen`, `router`, `and_`, `or_` from `crewai.flow.flow` only where needed for typing or user re-export. Import behind a `try/except ImportError` block matching `src/thenvoi/adapters/crewai.py:22-32` so users without the `crewai` extra get a clear error.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `CrewAIFlowAdapter` extending `SimpleAdapter[CrewAIFlowSessionState]` with the constructor from [Public adapter API](../technical-spec.md#public-adapter-api). Constructor never calls `flow_factory()`. Constructor validation raises `ThenvoiConfigError` per the [Validation rules](../technical-spec.md#validation-rules) table.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `on_started` storing `agent_name` and `agent_description`. Do not invoke the Flow.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — implement `on_cleanup(room_id)` that removes per-room async locks and transient caches scoped to that room only.
- [ ] `src/thenvoi/adapters/crewai_flow.py` — never call synchronous `Flow.kickoff()`. Use `kickoff_async` only.
- [ ] `tests/adapters/test_crewai_flow_adapter.py` — mock CrewAI modules using the `_get_crewai_adapter_cls` pattern from `tests/framework_configs/adapters.py:134-193`. Tests for constructor defaults, invalid config, `on_started`, `on_cleanup`.

## Acceptance criteria
- [ ] Unit: constructor validation, lifecycle behavior.
- [ ] Regression: `tests/adapters/test_crewai_adapter.py` still passes without modification.
- [ ] Pass criterion: `uv run pytest tests/adapters/test_crewai_flow_adapter.py tests/adapters/test_crewai_adapter.py -v -k "init or cleanup or started or validation or test_crewai"`
- [ ] Acceptance criterion 27 from spec: `CrewAIFlowAdapter` validates every constructor field. Constructor never calls `flow_factory()`.
- [ ] Acceptance criterion 28: Invalid constructor values raise `ThenvoiConfigError`. `max_run_age=timedelta(0)` and `max_run_age=timedelta(seconds=-1)` both raise.
- [ ] Acceptance criterion 29: `state_source` validation rejects objects without an awaitable `load_task_events(*, room_id, metadata_namespace, tools, history)` method.
- [ ] Acceptance criterion 30: `metadata_namespace=None` resolves to `f"crewai_flow:{agent_id}"` in `on_started`. Two adapters with different `agent_id` values produce different default namespaces.
- [ ] Acceptance criterion 31: `on_started` stores agent metadata and does not call the Flow.
- [ ] Acceptance criterion 32: `on_cleanup(room_id)` removes locks and transient caches scoped to that room only.
- [ ] Acceptance criterion 33: `tests/adapters/test_crewai_adapter.py` continues to pass with no edits, verified by `uv run pytest tests/adapters/test_crewai_adapter.py -v`.
- [ ] Acceptance criterion 34: `uv run pytest tests/adapters/test_crewai_flow_adapter.py -v -k "init or cleanup or started or validation or namespace"` passes.

## Out of scope
- Calling the Flow (Phase 3).
- Reply matching or delegation (Phase 4).
- Public registration via `__init__.py` lazy imports (Phase 6).
