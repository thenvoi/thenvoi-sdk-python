---
commit_hash: 06b20f78045481f11ca8a150719c7201f856db21
analyzed_at: 2026-04-27 09:46 PDT
---

# Codebase Analysis

## Scope

| File | Relevant entities | Notes |
|------|-------------------|-------|
| `reason/260427-0859-crewai-flow-adapter/technical-spec.md` | `CrewAIFlowAdapter`, `CrewAIFlowStateConverter`, task-event metadata, reply matching, safety policies | Draft implementation plan under review. |
| `src/thenvoi/preprocessing/default.py` | `DefaultPreprocessor.process` | Hydrates history only on session bootstrap at lines 78-85, then passes `HistoryProvider(raw=raw_history)` at lines 96-103. |
| `src/thenvoi/core/simple_adapter.py` | `SimpleAdapter.on_event` | Converts only the supplied `AgentInput.history` and awaits `on_message` at lines 137-155. |
| `src/thenvoi/runtime/tools.py` | `AgentTools.send_message`, `AgentTools.send_event` | `send_message` accepts content and mentions only at lines 1194-1256; `send_event` accepts metadata at lines 1258-1293. |
| `src/thenvoi/adapters/crewai.py` | `CrewAIAdapter`, `_process_message`, ContextVar tool bridge | Current adapter stores per-room message history in memory and calls `CrewAIAgent.kickoff_async(messages)` at lines 1201-1285. |
| `src/thenvoi/core/types.py` | `AdapterFeatures`, `Emit.TASK_EVENTS` | Task events are modeled as an optional emit feature at lines 29-49. |
| `src/thenvoi/testing/fake_tools.py` | `FakeAgentTools` | Fake sends always succeed and append deterministic local records at lines 58-82. |
| `examples/crewai/README.md` | CrewAI examples guidance | Current examples teach `CrewAIAdapter` role/goal/backstory and credential-heavy live examples. |

## Key constraints discovered

The current runtime does not give every message a full room history. It only loads history on bootstrap. A Flow state converter that expects prior task events on every later peer reply will receive an empty raw history unless the implementation adds a new state hydration path.

Visible messages cannot carry metadata or idempotency keys through the current `AgentTools.send_message` surface. Task events can carry metadata, but they are separate writes after visible side effects in the current spec.

CrewAI Flow's `kickoff_async(inputs)` copies top-level input keys into `flow.state`, so top-level Flow inputs must be serializable. Runtime objects need a separate access path.
