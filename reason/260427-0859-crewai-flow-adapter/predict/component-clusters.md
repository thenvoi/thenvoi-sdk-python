---
commit_hash: 06b20f78045481f11ca8a150719c7201f856db21
analyzed_at: 2026-04-27 09:46 PDT
---

# Component Clusters

| Cluster | Files | Key entities | Risk areas |
|---------|-------|--------------|------------|
| Runtime hydration | `src/thenvoi/preprocessing/default.py`, `src/thenvoi/core/simple_adapter.py` | `DefaultPreprocessor.process`, `SimpleAdapter.on_event` | Non-bootstrap peer replies do not receive persisted task-event history. |
| Platform side effects | `src/thenvoi/runtime/tools.py` | `send_message`, `send_event` | No atomic visible-message-plus-state write; no idempotency keys. |
| CrewAI adapter surface | `src/thenvoi/adapters/crewai.py`, `src/thenvoi/converters/crewai.py` | `CrewAIAdapter`, `CrewAIHistoryConverter` | Current user mental model is agent/tool based, not decision-envelope based. |
| Test harness | `src/thenvoi/testing/fake_tools.py`, `tests/adapters/test_crewai_adapter.py`, `tests/framework_configs/*.py` | `FakeAgentTools`, CrewAI import mocks, conformance configs | Existing fakes do not model failures, retries, concurrent instances, or state replay. |
| Spec artifacts | `reason/260427-0859-crewai-flow-adapter/*.md` | `technical-spec.md`, `plan-handoff.md` | Spec promises safety-net replacement while excluding live E2E and platform idempotency. |
