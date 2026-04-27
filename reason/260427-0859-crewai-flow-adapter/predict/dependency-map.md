---
commit_hash: 06b20f78045481f11ca8a150719c7201f856db21
analyzed_at: 2026-04-27 09:46 PDT
---

# Dependency Map

## Import and lifecycle flow

| Step | Source | Destination | Evidence | Risk area |
|------|--------|-------------|----------|-----------|
| WebSocket message -> preprocessor | platform event | `DefaultPreprocessor.process` | `src/thenvoi/preprocessing/default.py:78-85` | History hydration only happens on bootstrap. |
| Preprocessor -> adapter input | `HistoryProvider(raw=raw_history)` | `SimpleAdapter.on_event` | `src/thenvoi/preprocessing/default.py:96-103`, `src/thenvoi/core/simple_adapter.py:137-155` | Converted state can only see supplied history. |
| Adapter -> visible message | `AgentTools.send_message` | REST `create_agent_chat_message` | `src/thenvoi/runtime/tools.py:1194-1256` | No metadata/idempotency field. |
| Adapter -> state event | `AgentTools.send_event` | REST `create_agent_chat_event` | `src/thenvoi/runtime/tools.py:1258-1293` | Separate write from visible message. |
| Existing CrewAI behavior | `CrewAIAdapter._process_message` | `CrewAIAgent.kickoff_async(messages)` | `src/thenvoi/adapters/crewai.py:1201-1285` | In-memory history and prompt/tool-loop control remain separate from proposed Flow contract. |

## Data flows

| Source | Transform | Sink | Risk |
|--------|-----------|------|------|
| Prior task events | `CrewAIFlowStateConverter.convert(raw)` | `CrewAIFlowSessionState` | Fails on non-bootstrap turns unless task events are explicitly hydrated. |
| Flow decision | Adapter normalization | `send_message` then `send_event` | Visible side effect can happen without state being recorded. |
| Peer reply sender | participant-key normalization | pending delegation match | Sender-only matching is ambiguous when the same peer has multiple pending runs. |
| Runtime tools | `get_current_flow_runtime()` | Flow method code | If raw `AgentTools` is exposed, Flow code can bypass adapter policy checks. |
