# Hypothesis Queue — CrewAI Flow Adapter Spec

Ranked hypotheses for downstream chain consumption. Each is a testable assertion derived from a finding.

| Rank | ID | Hypothesis | Confidence | Location | Source |
|------|----|----|---|----|----|
| 1 | H-01 | The reserve-send-confirm protocol leaves runs permanently stuck when the confirmation send_event fails for any transient reason; no retry path exists in v1 | HIGH | technical-spec.md:359-366 | RE-1, 5/5 consensus, escalated to CRITICAL |
| 2 | H-02 | Sub-Crew BaseTools returned by `runtime.create_crewai_tools()` cannot reach the adapter's private SideEffectExecutor — the wiring from the framework-agnostic Phase 0 builder to the Phase 3 run state is not specified | HIGH | technical-spec.md:485-498, :546-548 | AR-2, 5/5 consensus, merged with DA-3 |
| 3 | H-03 | RestCrewAIFlowStateSource refetches the full room context every turn — for a room with M messages over N turns, total cost is O(N×M) and the spec has no v1 mitigation | HIGH | technical-spec.md:241-249 | PE-1, 5/5 consensus |
| 4 | H-04 | A parallel fan-out with N peers writes 2N+2 task events per run; combined with O(M) per-turn reads this is multiplicative read-write amplification | HIGH | technical-spec.md:359-366 | DA-2, escalated by PE in debate |
| 5 | H-05 | Two task events with identical `inserted_at` apply in REST-response order (not platform-monotonic order), which can misclassify a delegated_pending state as indeterminate | HIGH | technical-spec.md:223 | RE-3, escalated by SA in debate |
| 6 | H-06 | A `@listen` method that mutates `flow.state` mid-run loses those mutations — the spec does not document this lifetime contract | MEDIUM | technical-spec.md:308-316 | AR-1 |
| 7 | H-07 | Two CrewAIFlowAdapter agents in the same room with default `metadata_namespace="crewai_flow"` reconstruct each other's state on every turn | MEDIUM | technical-spec.md:295, :245 | AR-3, 5/5 consensus |
| 8 | H-08 | A peer reply arriving after run finalization is misclassified as a new user input — sender role (Agent vs User) is not filtered in rule 8 | MEDIUM | technical-spec.md:380 | SA-4, 5/5 consensus |
| 9 | H-09 | `flow_factory()` raising at construction time produces an opaque turn failure — no `failed` task event is written, no error event reaches the user | MEDIUM | technical-spec.md:570 | RE-5, 5/5 consensus |
| 10 | H-10 | A user who copies a test config using HistoryCrewAIFlowStateSource into production gets silent state loss on every non-bootstrap turn | MEDIUM | technical-spec.md:266-267 | RE-4, 5/5 consensus |
| 11 | H-11 | The 68 acceptance criteria pass while composed multi-turn behavior breaks — no end-to-end ordered trace fixture covers the full v1 user promise | MEDIUM | technical-spec.md:881-975 | DA-5, escalated to MEDIUM |
| 12 | H-12 | Two near-simultaneous on_message calls on the same room load identical state, both pass through the lock, both reserve for the same run — load happens before lock | MEDIUM | technical-spec.md:515-517 | RE-2, 4/5 consensus |
| 13 | H-13 | A wrapped AgentTools (audit, rate limit, signing) cannot mediate the per-turn full-context REST call because the state source goes through `tools.rest` directly | MEDIUM | technical-spec.md:264-267 | SA-1 |
| 14 | H-14 | Internal SDK code that imports `_ensure_nest_asyncio`, `_run_async`, or other private names from `adapters/crewai.py` breaks silently after Phase 0 extraction | MEDIUM | technical-spec.md:431 | AR-5 |
| 15 | H-15 | `buffered_syntheses[*].content` storage of LLM-produced text contradicts the spec's "no peer reply content in metadata" privacy assertion | MEDIUM | technical-spec.md:843-844 | SA-3 |
| 16 | H-16 | Per-turn `Flow.__init__` registers a fresh trace listener on `crewai_event_bus` — listener count grows unbounded over a long-running process | MEDIUM | technical-spec.md:570 | PE-2 (merged with AR-4) |
| 17 | H-17 | Converter scans the full task-event log on every turn — for K total events the per-turn cost is O(K×D) where D is delegation list length | MEDIUM | technical-spec.md:218-225 | PE-3 |
| 18 | H-18 | The CrewAIFlowAdapter has no validated user — the only known real-world router demand for this routing shape switched to a different adapter | MEDIUM | technical-spec.md:39-86, :977-985 | DA-1 |
| 19 | H-19 | A 700-line refactor in Phase 0 introduces flake-level regressions in nest_asyncio + ContextVar timing that pass tests but fail in production | MEDIUM | technical-spec.md:402-479 | DA-4 |
| 20 | H-20 | The 32-bit correlation token is computed from public input — peers can compute and spoof it, but reply_ambiguous fail-closed limits state-mutation impact | LOW | technical-spec.md:382-385 | SA-2, downgraded by DA |
| 21 | H-21 | `max_delegation_rounds` semantics are unspecified — implementer must guess whether N delegations in one Flow output count as 1 round or N | LOW | technical-spec.md:294 | RE-6 |
| 22 | H-22 | Thenvoi task events may have a TTL the spec does not document — old runs with pruned events become "new input" on late peer reply | LOW | technical-spec.md:843 | RE-7 |
| 23 | H-23 | Importing CrewAIFlowAdapter and never calling create_crewai_tools should not invoke the irreversible nest_asyncio.apply patch | LOW | technical-spec.md:431 | PE-4 |
