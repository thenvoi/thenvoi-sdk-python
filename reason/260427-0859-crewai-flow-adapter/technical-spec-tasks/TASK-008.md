---
id: TASK-008
phase: 7
status: completed
depends-on: [TASK-007]
---

# TASK-008: Add the runnable example

## Objective
Add one runnable example that shows when to use `CrewAIFlowAdapter` versus the existing `CrewAIAdapter`.

## Spec reference
> **Goal:** Add one runnable example that shows when to use `CrewAIFlowAdapter` versus the existing `CrewAIAdapter`.

## Changes
- [ ] `examples/crewai/08_flow_router.py` — PEP 723 runnable example using `CrewAIFlowAdapter`, a toy Flow factory with two start methods (delegate and synthesize), `join_policy="all"`, and `sequential_chains={"data-fetcher": "presenter"}`. The example uses generic peer names (`data-fetcher`, `presenter`, `ticket-bot`, `task-bot`) and demonstrates direct response, parallel delegation, sequential handoff, waiting turn, and final synthesis.
- [ ] `examples/crewai/README.md` — add an "Adapter choice" section that names `CrewAIAdapter` for normal CrewAI agent turns (single-hop delegation, bounded one-peer synthesis) and `CrewAIFlowAdapter` for room-router orchestration (parallel join, sequential composition, tagged-peer routing).
- [ ] `src/thenvoi/adapters/crewai_flow.py` — module docstring states experimental status, v1 scope, the safety policies offered, the task-event state log, and that this adapter does not replace `CrewAIAdapter`.
- [ ] `tests/adapters/test_crewai_flow_adapter.py` — import-path and default-namespace test matching the example.

## Acceptance criteria
- [ ] Example syntax check: `uv run python -m py_compile examples/crewai/08_flow_router.py`.
- [ ] Unit: default `metadata_namespace` and import path match the example.
- [ ] Pass criterion: `uv run python -m py_compile examples/crewai/08_flow_router.py && uv run ruff check examples/crewai/08_flow_router.py examples/crewai/README.md src/thenvoi/adapters/crewai_flow.py src/thenvoi/converters/crewai_flow.py tests/adapters/test_crewai_flow_adapter.py tests/converters/test_crewai_flow.py`
- [ ] Acceptance criterion 79 from spec: `examples/crewai/08_flow_router.py` compiles with `uv run python -m py_compile examples/crewai/08_flow_router.py`.
- [ ] Acceptance criterion 80: `examples/crewai/08_flow_router.py` shows direct response, parallel delegation, sequential handoff, waiting state, and final synthesis using generic peer names.
- [ ] Acceptance criterion 81: `examples/crewai/README.md` adds an "Adapter choice" section that names `CrewAIAdapter` for normal agent turns and `CrewAIFlowAdapter` for room-router orchestration.
- [ ] Acceptance criterion 82: The example does not imply suspended Flow execution across future events.
- [ ] Acceptance criterion 83: PEP 723 metadata follows the project rules.
- [ ] Acceptance criterion 84: `uv run ruff check examples/crewai/08_flow_router.py examples/crewai/README.md src/thenvoi/adapters/crewai_flow.py src/thenvoi/converters/crewai_flow.py tests/adapters/test_crewai_flow_adapter.py tests/converters/test_crewai_flow.py` passes.

## Out of scope
- Removing the experimental marker on `CrewAIFlowAdapter` — gated on a later minor release per the migration plan.
- Live LLM or Thenvoi-platform end-to-end tests against the example.
- A separate example for sub-Crew tooling via `runtime.create_crewai_tools()`.
