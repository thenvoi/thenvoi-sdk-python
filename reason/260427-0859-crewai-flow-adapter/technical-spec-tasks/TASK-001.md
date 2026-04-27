---
id: TASK-001
phase: 0
status: completed
depends-on: []
---

# TASK-001: Extract shared CrewAI tool wrappers

## Objective
Move the CrewAI `BaseTool` wrappers and the sync-to-async tool-execution plumbing out of `src/thenvoi/adapters/crewai.py` and into a new shared module, so both `CrewAIAdapter` and `CrewAIFlowAdapter` consume one set of wrappers and so Flow authors who spawn sub-Crews inside `@listen` methods get platform tools without copying code.

## Spec reference
> **Goal:** Move the CrewAI `BaseTool` wrappers and the sync-to-async tool-execution plumbing out of `src/thenvoi/adapters/crewai.py` and into a new shared module, so both `CrewAIAdapter` and `CrewAIFlowAdapter` consume one set of wrappers and so Flow authors who spawn sub-Crews inside `@listen` methods get platform tools without copying code.

## Changes
- [x] `src/thenvoi/integrations/crewai/__init__.py` — new package init re-exporting the public surface.
- [x] `src/thenvoi/integrations/crewai/runtime.py` — move `_ensure_nest_asyncio`, `_nest_asyncio_lock`, `_nest_asyncio_applied`, `_run_async` from `adapters/crewai.py:47-110`. Rename `_run_async` to `run_async` (public). Module docstring carries the existing process-global `nest_asyncio` warning verbatim.
- [x] `src/thenvoi/integrations/crewai/tools.py` — move all `*Input` Pydantic models, all `*Tool` `BaseTool` subclasses, the custom-tool factory, and the `_execute_tool` helper into this module. Implement `CrewAIToolContext` and `CrewAIToolReporter` (protocol + `EmitExecutionReporter` + `NoopReporter`). Implement `build_thenvoi_crewai_tools` per the signature above.
- [x] `src/thenvoi/adapters/crewai.py` — replace `_create_crewai_tools` with one call to `build_thenvoi_crewai_tools(get_context=self._get_context, reporter=EmitExecutionReporter(self.features), capabilities=self.features.capabilities, custom_tools=self._custom_tools, fallback_loop=self._tool_loop)`. Remove the now-extracted helpers (`_execute_tool`, `_report_tool_call`, `_report_tool_result`, `_serialize_success_result`, `_run_async`, `_ensure_nest_asyncio`, the module-level `nest_asyncio` lock). The module shrinks by ~700 lines.
- [x] `src/thenvoi/adapters/crewai.py` — keep its own `_current_room_context` ContextVar (it is the legacy adapter's binding mechanism). The new helper `_get_context(self) -> CrewAIToolContext | None` reads that ContextVar and returns it as a `CrewAIToolContext`.
- [x] **Pre-extraction audit step:** before deleting any code, run `rg "from thenvoi.adapters.crewai import _" --type py` and `rg "thenvoi.adapters.crewai\._" --type py` across the entire repo. Audit reported zero internal references at the audit timestamp.
- [x] `tests/adapters/test_crewai_adapter.py` — updated tests that reached into extracted private names to use `thenvoi.integrations.crewai.runtime` for `_ensure_nest_asyncio`/`run_async`/`_nest_asyncio_lock`, and `EmitExecutionReporter` for `_report_tool_call`/`_report_tool_result`. Fixture evicts integration modules from sys.modules so mocks apply on reimport.
- [x] `tests/integrations/test_crewai_tools.py` — new file. Unit tests for `build_thenvoi_crewai_tools`: capability filtering selects the right tool list; `EmitExecutionReporter` only emits when `Emit.EXECUTION` is set; missing context returns the documented error JSON; custom tools are appended; `run_async` patches `nest_asyncio` lazily and only once across calls.
- [x] `tests/adapters/test_crewai_adapter_soak.py` — new file, marked `@pytest.mark.slow`. Drives 100 sequential `on_message` calls across 3 simulated rooms with mocked CrewAI, asserts no exceptions, no event-loop policy mutations beyond the initial `nest_asyncio` patch, and that the per-room state in `_message_history` does not leak between rooms.

## Acceptance criteria
- [x] Unit: `build_thenvoi_crewai_tools(capabilities=frozenset())` returns the seven base tools (no contacts, no memory).
- [x] Unit: adding `Capability.CONTACTS` adds the five contact tools; adding `Capability.MEMORY` adds the five memory tools.
- [x] Unit: `EmitExecutionReporter` with `Emit.EXECUTION` not in `features.emit` does not call `tools.send_event`.
- [x] Unit: a tool invoked with `get_context()` returning `None` returns the documented `{"status": "error", "message": "No room context available..."}` JSON.
- [x] Soak: 100 sequential turns across 3 rooms, no exceptions, no event-loop policy mutations after the first.
- [x] Regression: every existing test in `tests/adapters/test_crewai_adapter.py` passes.
- [x] Pass criterion: `uv run pytest tests/integrations/test_crewai_tools.py tests/adapters/test_crewai_adapter.py tests/adapters/test_crewai_adapter_soak.py -v` — 79 passed.
- [x] Acceptance criterion 1 from spec: `src/thenvoi/integrations/crewai/tools.py` exports `build_thenvoi_crewai_tools`, `CrewAIToolContext`, `CrewAIToolReporter`, `EmitExecutionReporter`, and `NoopReporter`.
- [x] Acceptance criterion 2: `src/thenvoi/integrations/crewai/runtime.py` exports `run_async` and the lazy `nest_asyncio` patch.
- [x] Acceptance criterion 3: `src/thenvoi/adapters/crewai.py` no longer defines `*Input` Pydantic models, `*Tool` `BaseTool` subclasses, `_create_crewai_tools`, `_execute_tool`, `_report_tool_call`, `_report_tool_result`, `_serialize_success_result`, `_run_async`, or `_ensure_nest_asyncio`. The legacy `_current_room_context` ContextVar stays. (File shrank from 1322 → 415 lines.)
- [x] Acceptance criterion 7: pre-extraction audit reported zero internal references.
- [x] Acceptance criterion 8: Phase 0 soak test passes.

## Out of scope
- Any work on `CrewAIFlowAdapter` itself (Phases 1–7).
- Changing the public signature of `CrewAIAdapter`.
- Adding a new pip extra.
