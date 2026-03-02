You are a focused subagent reviewer for a single holistic investigation batch.

Repository root: /Users/pp/thenvoi/thenvoi-sdk-python
Blind packet: /Users/pp/thenvoi/thenvoi-sdk-python/.desloppify/review_packet_blind.json
Batch index: 4
Batch name: Design coherence — Mechanical Concern Signals
Batch dimensions: design_coherence
Batch rationale: mechanical detectors identified structural patterns needing judgment; concern types: design_concern, duplication_design, mixed_responsibilities, systemic_pattern; truncated to 80 files from 89 candidates

Files assigned:
- .github/actions/GithubToken/generate_token.py
- docker/codex/runner.py
- examples/a2a_bridge/01_basic_agent.py
- examples/a2a_bridge/02_with_auth.py
- examples/a2a_gateway/01_basic_gateway.py
- examples/a2a_gateway/02_with_demo_agent.py
- examples/claude_sdk_docker/runner.py
- examples/coding_agents/create_agents.py
- examples/langgraph/02_custom_tools.py
- examples/pydantic_ai/02_custom_instructions.py
- examples/run_agent.py
- src/thenvoi/adapters/langgraph.py
- src/thenvoi/client/streaming/client.py
- src/thenvoi/config/loader.py
- src/thenvoi/converters/a2a.py
- src/thenvoi/converters/a2a_gateway.py
- src/thenvoi/core/adapter_base.py
- src/thenvoi/integrations/a2a/gateway/adapter.py
- src/thenvoi/integrations/base.py
- src/thenvoi/integrations/codex/rpc_base.py
- src/thenvoi/integrations/codex/stdio_client.py
- src/thenvoi/integrations/codex/websocket_client.py
- src/thenvoi/integrations/langgraph/graph_tools.py
- src/thenvoi/integrations/langgraph/langchain_tools.py
- src/thenvoi/preprocessing/default.py
- src/thenvoi/preprocessing/participants.py
- src/thenvoi/runtime/formatters.py
- src/thenvoi/runtime/presence.py
- src/thenvoi/runtime/shutdown.py
- src/thenvoi/runtime/tools.py
- src/thenvoi/testing/example_logging.py
- tests/adapters/test_codex_adapter.py
- tests/bridge/test_router.py
- tests/integration/test_dynamic_agent.py
- tests/integration/test_history_converters.py
- tests/integration/test_hub_room_strategy.py
- tests/integration/test_multi_agent.py
- tests/integration/test_participant_permissions.py
- tests/preprocessing/test_default.py
- examples/a2a_gateway/with_demo_agent.py
- examples/crewai/02_role_based_agent.py
- examples/crewai/03_coordinator_agent.py
- examples/langgraph/03_custom_personality.py
- examples/langgraph/04_calculator_as_tool.py
- examples/langgraph/06_delegate_to_sql_agent.py
- examples/langgraph/08_jerry_agent.py
- examples/parlant/02_with_guidelines.py
- examples/parlant/04_tom_agent.py
- src/thenvoi/core/__init__.py
- src/thenvoi/core/nonfatal.py
- src/thenvoi/runtime/compat/__init__.py
- thenvoi-bridge/bridge_core/__init__.py
- docker/claude_sdk/runner.py
- docker/shared/repo_init.py
- examples/anthropic/02_custom_instructions.py
- examples/anthropic/04_jerry_agent.py
- examples/langgraph/standalone_rag.py
- src/thenvoi/adapters/anthropic.py
- src/thenvoi/adapters/claude_sdk.py
- src/thenvoi/adapters/codex.py
- src/thenvoi/adapters/crewai.py
- src/thenvoi/adapters/parlant.py
- src/thenvoi/adapters/pydantic_ai.py
- src/thenvoi/core/protocols.py
- src/thenvoi/integrations/a2a/adapter.py
- src/thenvoi/integrations/a2a/gateway/server.py
- src/thenvoi/integrations/a2a_bridge/bridge.py
- src/thenvoi/integrations/claude_sdk/session_manager.py
- src/thenvoi/integrations/claude_sdk/tools.py
- src/thenvoi/integrations/parlant/tools.py
- src/thenvoi/platform/link.py
- src/thenvoi/runtime/contacts/contact_handler.py
- src/thenvoi/runtime/execution.py
- src/thenvoi/runtime/platform_runtime.py
- src/thenvoi/runtime/tool_bridge.py
- src/thenvoi/testing/example_runners.py
- src/thenvoi/testing/fake_tools.py
- src/thenvoi/testing/runner_core.py
- tests/adapters/test_crewai_adapter.py
- tests/docker/test_repo_init.py

Task requirements:
1. Read the blind packet and follow `system_prompt` constraints exactly.
1a. If previously flagged issues are listed above, use them as context for your review.
    Verify whether each still applies to the current code. Do not re-report fixed or
    wontfix issues. Use them as starting points to look deeper — inspect adjacent code
    and related modules for defects the prior review may have missed.
1c. Think structurally: when you spot multiple individual issues that share a common
    root cause (missing abstraction, duplicated pattern, inconsistent convention),
    explain the deeper structural issue in the finding, not just the surface symptom.
    If the pattern is significant enough, report the structural issue as its own finding
    with appropriate fix_scope ('multi_file_refactor' or 'architectural_change') and
    use `root_cause_cluster` to connect related symptom findings together.
2. Evaluate ONLY listed files and ONLY listed dimensions for this batch.
3. Return 0-10 high-quality findings for this batch (empty array allowed).
3a. Do not suppress real defects to keep scores high; report every material issue you can support with evidence.
3b. Do not default to 100. Reserve 100 for genuinely exemplary evidence in this batch.
4. Score/finding consistency is required: broader or more severe findings MUST lower dimension scores.
4a. Any dimension scored below 85.0 MUST include explicit feedback: add at least one finding with the same `dimension` and a non-empty actionable `suggestion`.
5. Every finding must include `related_files` with at least 2 files when possible.
6. Every finding must include `dimension`, `identifier`, `summary`, `evidence`, `suggestion`, and `confidence`.
7. Every finding must include `impact_scope` and `fix_scope`.
8. Every scored dimension MUST include dimension_notes with concrete evidence.
9. If a dimension score is >85.0, include `issues_preventing_higher_score` in dimension_notes.
10. Use exactly one decimal place for every assessment and abstraction sub-axis score.
11. Ignore prior chat context and any target-threshold assumptions.
12. Do not edit repository files.
13. Return ONLY valid JSON, no markdown fences.

Scope enums:
- impact_scope: "local" | "module" | "subsystem" | "codebase"
- fix_scope: "single_edit" | "multi_file_refactor" | "architectural_change"

Output schema:
{
  "batch": "Design coherence — Mechanical Concern Signals",
  "batch_index": 4,
  "assessments": {"<dimension>": <0-100 with one decimal place>},
  "dimension_notes": {
    "<dimension>": {
      "evidence": ["specific code observations"],
      "impact_scope": "local|module|subsystem|codebase",
      "fix_scope": "single_edit|multi_file_refactor|architectural_change",
      "confidence": "high|medium|low",
      "issues_preventing_higher_score": "required when score >85.0",
      "sub_axes": {"abstraction_leverage": 0-100 with one decimal place, "indirection_cost": 0-100 with one decimal place, "interface_honesty": 0-100 with one decimal place}  // required for abstraction_fitness when evidence supports it
    }
  },
  "findings": [{
    "dimension": "<dimension>",
    "identifier": "short_id",
    "summary": "one-line defect summary",
    "related_files": ["relative/path.py"],
    "evidence": ["specific code observation"],
    "suggestion": "concrete fix recommendation",
    "confidence": "high|medium|low",
    "impact_scope": "local|module|subsystem|codebase",
    "fix_scope": "single_edit|multi_file_refactor|architectural_change",
    "root_cause_cluster": "optional_cluster_name_when_supported_by_history"
  }],
  "retrospective": {
    "root_causes": ["optional: concise root-cause hypotheses"],
    "likely_symptoms": ["optional: identifiers that look symptom-level"],
    "possible_false_positives": ["optional: prior concept keys likely mis-scoped"]
  }
}
