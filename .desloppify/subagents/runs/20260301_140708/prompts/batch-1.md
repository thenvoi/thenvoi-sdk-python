You are a focused subagent reviewer for a single holistic investigation batch.

Repository root: /Users/pp/thenvoi/thenvoi-sdk-python
Blind packet: /Users/pp/thenvoi/thenvoi-sdk-python/.desloppify/review_packet_blind.json
Batch index: 1
Batch name: Abstractions & Dependencies
Batch dimensions: abstraction_fitness
Batch rationale: abstraction hotspots (wrappers/interfaces/param bags), dep cycles

Files assigned:
- src/thenvoi/adapters/langgraph.py
- tests/adapters/test_crewai_adapter.py
- tests/integrations/a2a/gateway/test_adapter.py
- tests/bridge/test_bridge.py
- tests/example_agents/a2a_gateway/test_orchestrator_executor.py
- tests/bridge/test_session.py
- tests/platform/test_link.py
- tests/adapters/test_parlant_adapter.py
- src/thenvoi/runtime/execution.py
- src/thenvoi/compat/shims.py
- tests/test_platform_runtime.py
- tests/adapters/test_pydantic_ai_adapter.py
- src/thenvoi/integrations/a2a_bridge/bridge.py
- tests/adapters/test_langgraph_adapter.py
- tests/integrations/a2a/test_adapter.py
- tests/integration/test_a2a_gateway_context.py
- tests/test_agent_contacts.py
- src/thenvoi/runtime/contacts/contact_handler.py
- src/thenvoi/runtime/platform_runtime.py
- tests/runtime/test_contact_handler.py
- src/thenvoi/adapters/claude_sdk.py
- src/thenvoi/integrations/a2a/gateway/adapter.py
- src/thenvoi/testing/example_runners.py
- src/thenvoi/adapters/codex.py
- tests/support/events.py
- tests/core/test_simple_adapter.py
- src/thenvoi/adapters/crewai.py
- src/thenvoi/core/simple_adapter.py
- src/thenvoi/runtime/tools.py
- src/thenvoi/adapters/pydantic_ai.py
- tests/support/integration/config.py
- src/thenvoi/core/protocols.py
- src/thenvoi/client/streaming/client.py
- src/thenvoi/adapters/parlant.py
- src/thenvoi/adapters/anthropic.py
- src/thenvoi/integrations/a2a_bridge/route_dispatch.py
- src/thenvoi/testing/fake_tools.py
- tests/adapters/test_codex_adapter.py
- tests/preprocessing/test_default.py
- tests/integrations/codex/test_adapter_e2e.py

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
  "batch": "Abstractions & Dependencies",
  "batch_index": 1,
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
