# CrewAI Flow Adapter Reasoning Overview

Task: determine whether Thenvoi should build a new CrewAI adapter around CrewAI Flows instead of `crewai.Agent.kickoff_async(...)` for deterministic per-room routing, delegation, join, and synthesis.

Domain: software architecture
Mode: convergent
Judges: 5
Convergence: achieved after 2 rounds. Round 2 was unanimous for the code-grounded design.

Final decision: build a new experimental `CrewAIFlowAdapter`, but do not replace the existing `CrewAIAdapter`. Treat CrewAI Flow as a per-inbound-message execution graph, not as a long-lived external-event subscription or suspended workflow in v1.

Core rationale: the current Thenvoi adapter contract awaits `on_message` for each inbound message, and framework adapters process messages only. A Flow adapter can delegate and record pending orchestration state, but it should return instead of blocking for future peer replies. Later peer replies should re-enter normal `on_message` handling, where pending state is reconstructed from task events and synthesis happens once the join policy is satisfied.

Key code evidence:

- `src/thenvoi/core/simple_adapter.py:137-155`: `on_event` awaits `on_message` directly.
- `src/thenvoi/core/protocols.py:194-220`: framework adapters process messages only; lifecycle/presence events are filtered.
- `src/thenvoi/runtime/tools.py:1194-1256`: `send_message` accepts content and mentions only, with no metadata parameter.
- `src/thenvoi/runtime/tools.py:1258-1293`: `send_event` accepts metadata and supports task/error/thought events.
- `src/thenvoi/converters/a2a.py:16-76` and `src/thenvoi/adapters/letta.py:804-829`: existing integrations already use task-event metadata to reconstruct external session state.
- `src/thenvoi/adapters/crewai.py:1163-1284`: the current adapter binds ContextVar room/tool state around `on_message` and calls `Agent.kickoff_async(messages)`.

Final recommendation: yes, this is possible and a good fit for deterministic orchestration if scoped honestly. CrewAI Flow should be the local execution plan for one incoming Thenvoi message. Thenvoi task events should be the durable state log. Anything requiring true paused/resumed external-event execution should be a later runner/runtime feature, not part of the v1 adapter.
