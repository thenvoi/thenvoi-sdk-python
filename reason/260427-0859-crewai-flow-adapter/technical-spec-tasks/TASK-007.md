---
id: TASK-007
phase: 6
status: pending
depends-on: [TASK-006]
---

# TASK-007: Register the adapter surface

## Objective
Expose `CrewAIFlowAdapter` and `CrewAIFlowStateConverter` through the SDK lazy-import surfaces. Registration happens last so users cannot import a half-working orchestration adapter.

## Spec reference
> **Goal:** Expose `CrewAIFlowAdapter` and `CrewAIFlowStateConverter` through the SDK lazy-import surfaces. Registration happens last so users cannot import a half-working orchestration adapter.

## Changes
- [ ] `src/thenvoi/adapters/__init__.py` — add `CrewAIFlowAdapter` to `TYPE_CHECKING`, `__all__`, and `__getattr__`.
- [ ] `src/thenvoi/converters/__init__.py` — add `CrewAIFlowStateConverter` and `CrewAIFlowSessionState` to `TYPE_CHECKING`, `__all__`, and `__getattr__`.
- [ ] `tests/framework_configs/adapters.py` — add `_get_crewai_flow_adapter_cls`, `_crewai_flow_factory`, `_build_crewai_flow_config`. Append `_build_crewai_flow_config` to `_ADAPTER_CONFIG_BUILDERS`. Reuse the `_get_crewai_adapter_cls` mock pattern (`tests/framework_configs/adapters.py:134-193`); guard runtime methods (`on_message` and any flow-execution method) with the same `_CONFORMANCE_ONLY` pattern at lines 196-216.
- [ ] `tests/framework_configs/converters.py` — add `crewai_flow` to `CONVERTER_EXCLUDED_MODULES` (the precedent at lines 256-268 for `letta`, `codex`, `opencode`). The converter is metadata-only and does not implement the standard `convert() -> framework-format` contract that the harness validates. `test_config_drift.py::TestConverterConfigDrift` then passes.
- [ ] `pyproject.toml` — keep CrewAI Flow under the existing `crewai` and `dev` extras. CrewAI 1.14.3 is already pinned at `:69` and `:152`. No new extra.
- [ ] `tests/adapters/test_crewai_flow_adapter.py` — import-shape test: call `inspect.signature` on `crewai.flow.flow.Flow.__init__` and `Flow.kickoff_async` and assert parameter names (`persistence`, `tracing`, `**kwargs`; `inputs`). Do not assert on `crewai.__version__`.

## Acceptance criteria
- [ ] Unit: lazy imports work without importing real CrewAI until requested.
- [ ] Conformance: config drift passes; converter exclusion is explicit.
- [ ] Pass criterion: `uv run pytest tests/framework_conformance/test_config_drift.py tests/framework_conformance/test_adapter_conformance.py -v -k "crewai_flow or config_drift"`
- [ ] Acceptance criterion 73 from spec: `thenvoi.adapters.CrewAIFlowAdapter` lazy import works.
- [ ] Acceptance criterion 74: `thenvoi.converters.CrewAIFlowStateConverter` lazy import works.
- [ ] Acceptance criterion 75: `tests/framework_configs/adapters.py` includes `crewai_flow` with mocked CrewAI imports and `_CONFORMANCE_ONLY` runtime guards.
- [ ] Acceptance criterion 76: `crewai_flow` appears in `CONVERTER_EXCLUDED_MODULES` with a one-line reason comment matching the precedent for `letta`/`codex`/`opencode`. `tests/framework_conformance/test_config_drift.py::TestConverterConfigDrift::test_all_converter_modules_are_covered` and `test_no_stale_exclusions` both pass.
- [ ] Acceptance criterion 77: `pyproject.toml` keeps CrewAI Flow under the existing `crewai` and `dev` extras.
- [ ] Acceptance criterion 78: `uv run pytest tests/framework_conformance/test_config_drift.py tests/framework_conformance/test_adapter_conformance.py -v -k "crewai_flow or config_drift"` passes.

## Out of scope
- Runnable example (Phase 7).
- Removing the experimental marker (a later minor release per the migration plan).
