# Hypothesis Queue

| Rank | ID | Hypothesis | Confidence | Location | Source |
|------|----|------------|------------|----------|--------|
| 1 | H-01 | On non-bootstrap peer replies, `CrewAIFlowStateConverter` will not see prior task events unless a new hydration path is added. | HIGH | `src/thenvoi/preprocessing/default.py:78` | Confirmed 5/5 |
| 2 | H-02 | A crash after `send_message` but before task-event recording can duplicate delegation or final synthesis on retry. | HIGH | `src/thenvoi/runtime/tools.py:1194` | Confirmed 5/5 |
| 3 | H-03 | If Flow methods can access raw `AgentTools`, they can bypass join/finalization safety checks by sending visible messages directly. | MEDIUM | `reason/260427-0859-crewai-flow-adapter/technical-spec.md:388` | Confirmed 4/5 |
| 4 | H-04 | Sender-only ambiguous reply matching can attach a stale peer reply to the wrong run. | HIGH | `reason/260427-0859-crewai-flow-adapter/technical-spec.md:273` | Confirmed 5/5 |
| 5 | H-05 | Optional `AdapterFeatures.emit` can conflict with mandatory task-event persistence unless the adapter forces or separates state events. | HIGH | `src/thenvoi/core/types.py:29` | Confirmed 4/5 |
| 6 | H-06 | The Flow output contract is too implicit and users will return malformed decisions without an exact discriminator schema. | MEDIUM | `reason/260427-0859-crewai-flow-adapter/technical-spec.md:255` | Probable 3/5 |
| 7 | H-07 | Participant identity normalization cannot be implemented reliably from formatted participant text alone. | MEDIUM | `src/thenvoi/runtime/formatters.py:95` | Confirmed 4/5 |
| 8 | H-08 | The planned fake/unit tests will miss crash, replay, concurrency, and ordered Genpact trace failures. | HIGH | `src/thenvoi/testing/fake_tools.py:58` | Confirmed 5/5 |
| 9 | H-09 | The spec is implementing a framework-neutral router inside a CrewAI adapter, increasing complexity and support burden. | HIGH | `reason/260427-0859-crewai-flow-adapter/technical-spec.md:143` | Confirmed 4/5 |
| 10 | H-10 | The planned example filename conflicts with existing examples and compile-only validation can overstate readiness. | HIGH | `examples/crewai/README.md:18` | Probable 3/5 |

## Suggested first fixes

1. Rewrite the spec around a new `CrewAIFlowStateSource`/hydration contract.
2. Replace send-then-record side effects with reserve-send-confirm transitions or require platform idempotency keys.
3. Change `CrewAIFlowRuntime.tools` to a restricted facade.
4. Change ambiguous reply matching from newest-run heuristic to fail-closed plus explicit correlation token.
5. Add exact Pydantic models for Flow decisions.
