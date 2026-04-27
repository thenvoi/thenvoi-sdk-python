# Findings — CrewAI Flow Adapter Spec

Ranked by composite priority (severity × confidence × consensus).

---

## Finding 1: Reservation-without-send-confirmation stalls a run permanently

**Severity:** CRITICAL
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:359-366`
**Consensus:** 5/5

### Evidence

The reserve-send-confirm protocol writes a reservation event (step 3), sends the visible message (step 4), then writes a sent-confirmation event (step 5). The spec says: "If the sent-confirm task event fails after the visible message succeeds, record an error event if possible and mark the in-memory result `indeterminate` for this process. A future retry sees only the reservation and must fail closed."

There is no retry policy on the confirmation event itself. A single transient `send_event` failure — separate from the visible send, often caused by a network blip or a brief platform hiccup — wedges the run forever. The downstream "operator reconciliation path" is named as future work, not v1.

### Why it matters

`send_event` failure is not exotic. In a busy room with hundreds of turns per day, the probability of step 5 failing at least once over a run's lifetime is non-trivial. Combined with Finding 4 (write amplification: 2N+2 events per run), every additional delegation increases the surface area for this failure. The adapter advertises "deterministic single-worker behavior" but a network blip during a confirmation write turns "deterministic" into "permanently stuck."

### Recommendation

In v1, retry the confirmation `send_event` with bounded backoff (e.g., 3 attempts over ~1 second) before marking indeterminate. Most step-5 failures are transient. Reserve `indeterminate` for genuinely unrecoverable cases (e.g., the visible send itself raised but partially succeeded). Add an acceptance criterion: "After a single transient send_event failure on the confirm step, the next retry attempt within the same on_message succeeds and the run records `delegated_pending` correctly. After three consecutive failures, the run records `indeterminate`."

---

## Finding 2: Sub-Crew side-effect wiring is unspecified

**Severity:** HIGH
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:546-548`, intersection with Phase 0 `:402-479`
**Consensus:** 5/5

### Evidence

Phase 3 says `runtime.create_crewai_tools()` returns `BaseTool` instances whose writes route through the adapter's `SideEffectExecutor`, generating side_effect_keys of the form `{run_id}:subcrew:{auto_inc}`. But the shared `build_thenvoi_crewai_tools` from Phase 0 is framework-agnostic — its tools accept a `get_context` callable and a `CrewAIToolReporter`, neither of which knows about `run_id` or carries an auto-increment counter. Nothing in the Phase 0 contract gives a sub-Crew tool the wiring to look up the active Flow run, derive a side_effect_key, write a reservation, or call the adapter-private `SideEffectExecutor`.

### Why it matters

A Monday-morning implementer reading Phase 3 cannot wire this up without inventing structure the spec did not agree to. The escape-hatch promise (Flow authors can spawn sub-Crews) does not connect to the safety promise (every visible side effect goes through reserve-send-confirm).

### Recommendation

Specify the wiring concretely. One viable shape: extend `CrewAIToolReporter` (or add a parallel injection point) so when `CrewAIFlowAdapter` calls `build_thenvoi_crewai_tools`, it passes a reporter whose `report_call` looks up the active Flow runtime via the adapter's ContextVar, derives the side_effect_key, and routes the visible send through the adapter-private executor. Document this as a separate subsection of Phase 3, with an explicit acceptance criterion: "A sub-Crew tool call writes a reservation event with the correct `subcrew:{n}` side_effect_key, and the visible send is suppressed if the same key has a sent record."

---

## Finding 3: Per-turn full-room-context fetch is O(N×M) over a room's lifetime

**Severity:** HIGH
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:241-249`, `:803-805`
**Consensus:** 5/5

### Evidence

`RestCrewAIFlowStateSource.load_task_events` calls `agent_api_context.get_agent_chat_context(chat_id=room_id)` per turn. That endpoint returns the full room context (every message and every event in the room). For a room with M total messages over N turns, total cost is O(N×M). The spec's resolved-questions table acknowledges this and defers mitigation to "a future task-event-only endpoint," but ships v1 without any cap on room size or any caching layer.

### Why it matters

A room that runs for a week with steady traffic accumulates thousands of events. A peer reply in turn 500 refetches the entire log just to find the orchestration state. Combined with Finding 4 (2N+2 task events per run), the read amplification compounds.

### Recommendation

Add a v1 mitigation: an in-memory cache keyed by `(room_id, last_seen_event_id)` so the source only refetches when the cache is stale or the platform indicates new events. Document an explicit v1 SLO ceiling for room size — e.g., "characterized for rooms up to 1,000 messages; degrades linearly beyond." If the Thenvoi REST API supports paginated fetch with a `since` cursor, use it. The spec must own a number, not defer it.

---

## Finding 4: 2N+2 task events per run — write and read amplification

**Severity:** HIGH
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:359-366`, `:218-225`
**Consensus:** 4/5

### Evidence

Reserve-send-confirm writes 2 task events per visible side effect. A parallel fan-out with N peer delegations plus one final synthesis writes 2N+2 task events per run. Combined with the per-turn full-context fetch (Finding 3), every turn pays the read fanout multiplied by the cumulative write count from all prior runs. For a room with 10 active runs of 5 delegations each, a single peer reply turn reads ~120 task events.

### Why it matters

This is the production cliff: the design works for short demos but degrades superlinearly under sustained load. The reservation-then-confirm pattern only adds value if "visible send succeeded but confirm failed" is a real failure mode worth detecting. If most failures happen at step 4 (visible send) or step 1 (state load), the reservation event adds cost without value.

### Recommendation

Justify the cost or simplify. A single task event written *after* the visible send (with the platform message_id present) covers the same retry-detection logic with half the writes: on retry, if the state shows a `delegation_message_id` for the same `delegation_id`, suppress the duplicate. The reservation-first pattern only matters when the Thenvoi platform returns a message_id from a partially failed send — which is not a documented platform behavior in the spec. Until that failure mode is empirically real, the simpler design is correct.

---

## Finding 5: `inserted_at` ordering has no tiebreaker

**Severity:** HIGH
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:223`, `:250`
**Consensus:** 4/5

### Evidence

The merge-semantics section says the converter "scans them in `inserted_at` order and applies merge rules per `run_id`." Two task events written within the same turn — a reservation and the immediately following sent-confirmation — frequently have identical millisecond-resolution timestamps. Python's sort is stable only with respect to insertion order, which is the REST response order, which the platform does not guarantee.

### Why it matters

Last-write-wins on scalar fields means the order of equal-key events decides which value sticks. If "sent" applies before "reservation," the merged state shows `status=delegated_pending` without a reservation marker — which then makes the indeterminate detection in Finding 1 fire spuriously. The bug surfaces as random false positives in production with no obvious cause.

### Recommendation

Sort by `(inserted_at, message_id)` where `message_id` is the platform's monotonic event id. Require the state source to read the platform's event id from the response and use it as a tiebreaker. Add a fixture: two events with identical `inserted_at` apply in event-id order regardless of REST response order.

---

## Finding 6: Two sources of truth — `flow.state` and task events

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:308-316`
**Consensus:** 4/5

### Evidence

Spec says: "CrewAI copies top-level input keys into `flow.state`, so every top-level value must be JSON-serializable." But state is also reconstructed from task events. If a `@listen` method mutates `flow.state` mid-run, those mutations are lost — only the terminal decision dict reaches the adapter. The spec does not document this lifetime contract or warn Flow authors. A naive implementer assumes `flow.state` survives between `kickoff_async` calls and writes mid-Flow state there.

### Why it matters

This is an implementer footgun, not a design failure. Real bugs will look like "my Flow remembers state inside one turn but loses it on the next" with no obvious cause. The fix is documentation, not redesign.

### Recommendation

Add an explicit "Flow state lifetime" subsection: `flow.state` is fresh per `kickoff_async` invocation, all durable orchestration state lives in task events, in-Flow mutations are scratch only. Document that no `persistence` argument should be passed to `Flow.__init__`. Add an example that shows the right pattern (read state from input, return decision in dict, never write to `self.state`).

---

## Finding 7: Default `metadata_namespace="crewai_flow"` collides if two Flow adapters share a room

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:295`, `:245`
**Consensus:** 5/5

### Evidence

The default namespace is `"crewai_flow"` for everyone. Two CrewAIFlowAdapters running in the same room (e.g. one routing agent and one specialist that also uses Flow) would both write task events under the same namespace and reconstruct each other's state on every turn — leading to nondeterministic behavior on join and finalize.

### Why it matters

This breaks once the SDK has more than one Flow-based agent in production. The trigger condition is reasonable: "I have a router agent and a specialist agent, both built on CrewAIFlowAdapter, both invited to the same room."

### Recommendation

Either prefix the default namespace with `agent_id` at construction (`crewai_flow:{agent_id}`), or document that any deployment with multiple `CrewAIFlowAdapter` agents in the same room MUST set distinct `metadata_namespace` values. Add a runtime warning if two namespaces collide based on task-event sniffing.

---

## Finding 8: Late peer reply after finalization is misclassified as a new user input

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:380`
**Consensus:** 5/5

### Evidence

Reply-matching rule 8: "If no candidate matches, treat the message as a new input, not a reply." Combined with the absorbing terminal state on `finalized` — a peer's slow reply that arrives after the run finalized has no pending delegation to match against, so it falls through to "new input." The adapter then treats the peer's late reply as if the peer is the user asking a new question. Sender role is not filtered.

### Why it matters

Real failure: peer takes longer than expected, run times out and finalizes with what was available, peer's late reply arrives, adapter starts a new run treating the peer as the user. The router answers the peer's reply as if it were a fresh question. User confusion at minimum, runaway agent loop in the worst case.

### Recommendation

Refine rule 8: only inbound messages from `User`-type senders can start a new run. `Agent`-type messages with no matching pending delegation are discarded with a debug-level log entry, never treated as new input. Add a fixture covering this exact scenario.

---

## Finding 9: `flow_factory` exception handling is unspecified

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:570`
**Consensus:** 5/5

### Evidence

If `flow_factory()` raises (Flow author does eager DB connection or other init in `__init__`), the spec does not say whether the adapter catches and records `failed`, or propagates. `SimpleAdapter.on_event` will see the exception and the turn fails opaquely — no task event written, no error event for the user.

### Recommendation

Wrap `flow_factory()` in try/except. On exception, record `failed` with code `flow_factory_error`, emit a normal error event, and return cleanly. Add an acceptance test where `flow_factory` raises and the run records failed state.

---

## Finding 10: `HistoryCrewAIFlowStateSource` silent fallback risks total state loss in production

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:266-267`
**Consensus:** 5/5

### Evidence

The opt-in `HistoryCrewAIFlowStateSource` is documented as "for tests and bootstrap-only deployments." But a user who picks it for tests and copy-pastes the test config into production gets silent state loss on every non-bootstrap turn — exactly the failure the REST source exists to prevent. There is no warning at construction or first non-bootstrap use.

### Recommendation

Emit a one-time WARNING-level log on first non-bootstrap use: "HistoryCrewAIFlowStateSource is not safe for non-bootstrap turns. If you see this in production, switch to RestCrewAIFlowStateSource." Better: gate the constructor behind an explicit confirmation kwarg like `acknowledge_test_only=True`.

---

## Finding 11: No end-to-end acceptance criterion for the v1 user promise

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:881-975` (entire acceptance criteria section)
**Consensus:** 5/5

### Evidence

The TL;DR promises "supports parallel fan-out with join, sequential composition, tagged-peer enforcement, and explicit waiting turns." 68 acceptance criteria test each in isolation. None runs a full multi-turn ordered trace covering: parent request → two delegations → first reply on turn N+1 → wait → second reply on turn N+5 → exactly one synthesis → duplicate parent request on turn N+10 suppressed → no second final.

### Why it matters

Unit criteria can all pass while the composed behavior breaks. This is the acceptance gap that lets a v1 ship green and fail in real use.

### Recommendation

Add one acceptance criterion in Phase 5: "End-to-end ordered-trace fixture covering parent request → two delegations → first reply (turn 2) → wait → second reply (turn 6) → final synthesis (turn 6) → duplicate parent request (turn 10) suppressed → no second final. Test asserts the full sequence of visible messages and task events matches a fixture exactly."

---

## Finding 12: Race between concurrent `on_message` calls — load happens before lock

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:515-517`
**Consensus:** 4/5

### Evidence

"Per-room async locks guard the reserve-send-confirm sequence... `SimpleAdapter.on_event` already serializes turns per `ExecutionContext`, so the locks are defensive." But state loading happens *before* lock acquisition. Two near-simultaneous turns load identical state, both queue at the lock, both reserve for the same run. If `SimpleAdapter.on_event` truly serializes per-ExecutionContext this never fires — but calling the locks "defensive" implies the author isn't sure.

### Recommendation

Either remove the locks (and document that the platform runtime is the only serializer) or move state loading inside the lock so the lock guards the entire load-decide-write critical section. The current "lock guards only writes" is double-checked-locking without the double-check.

---

## Finding 13: `tools.rest` punches through `AgentToolsProtocol`

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:264-267`
**Consensus:** 3/5

### Evidence

The state source uses `getattr(tools, 'rest', None)` to reach the concrete `AgentTools.rest` attribute. Any deployment that wraps `AgentTools` for audit, rate limiting, signing, or PII redaction loses the ability to mediate the per-turn full-context REST call.

### Why it matters

This is mostly architectural. No SDK caller currently wraps `AgentTools` for auditing. But the design encodes a pattern the protocol was meant to prevent.

### Recommendation

Add a `fetch_room_context(room_id)` method to `AgentToolsProtocol` (or a separate `StateSourceProtocol`) and have the state source consume it. Wrappers can then intercept. Defer if needed, but document the reason and the eventual seam.

---

## Finding 14: Phase 0 imports of private names not audited

**Severity:** MEDIUM
**Confidence:** MEDIUM
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:431`, acceptance `:893-894`
**Consensus:** 4/5

### Evidence

Phase 0 deletes module-level helpers (`_ensure_nest_asyncio`, `_run_async`, etc.) and rebuilds them in `integrations/crewai/`. Any test or downstream code that does `from thenvoi.adapters.crewai import _ensure_nest_asyncio` or patches that name in monkeypatch fixtures breaks silently. Acceptance criterion 7 says existing `test_crewai_adapter.py` passes unchanged but does not require an SDK-wide audit of internal references.

### Recommendation

Add a Phase 0 step: run `rg "from thenvoi.adapters.crewai import _" --type py` and migrate any internal references. Add an acceptance criterion: "No file outside `src/thenvoi/adapters/crewai.py` imports the now-extracted private names."

---

## Finding 15: `buffered_syntheses[*].content` storage contradicts the privacy assertion

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:843-844`
**Consensus:** 4/5

### Evidence

Privacy section asserts: "It must not store user secrets, API keys, full prompts, or full peer replies. Peer reply content stays in chat history." But `buffered_syntheses[*].content` stores partial synthesis text the Flow itself produced. Flow code is user-written. Nothing prevents a Flow from synthesizing content that includes peer-reply substrings, secrets the Flow saw, or PII.

### Recommendation

Either weaken the assertion ("Storage of partial synthesis text is at the Flow author's discretion; review your Flow's content pipeline before deploying") or add a size cap and a redaction hook (e.g., a `features.privacy_filter` callable applied to all task-event content before write).

---

## Finding 16: Flow-factory + event-bus listener accumulation per turn

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:570`
**Consensus:** 4/5

### Evidence

`Flow.__init__` runs `trace_listener.setup_listeners(crewai_event_bus)` and emits `FlowCreatedEvent` on the module-level `crewai_event_bus`. Per-turn instantiation = per-turn listener registration. For long-running processes with hot rooms, listener accumulation is a known leak pattern unless CrewAI deduplicates — which the spec does not verify.

### Recommendation

Add a Phase 3 acceptance: a memory test that runs 100 sequential `kickoff_async` cycles and asserts `crewai_event_bus` listener count is bounded. If it grows unbounded, the adapter must clean up bus listeners in the per-turn `finally` block, or cache Flow instances per room with explicit state reset.

---

## Finding 17: Converter merge cost on every turn

**Severity:** MEDIUM
**Confidence:** MEDIUM
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:218-225`
**Consensus:** 3/5

### Evidence

For a room with K total task events spread across R runs, every turn re-runs O(K) over the converter. Each event triggers list-merge-by-key on `delegations`, `sequential_chains`, `buffered_syntheses`. Per-turn cost is O(K×D) where D is delegation list length.

### Recommendation

In Phase 1, document the rule: terminal events for a `run_id` allow that run to be excluded from merge work on subsequent turns (only scan-not-merge to verify absorption). Add a benchmark fixture with a 1,000-event log and assert per-turn `convert` time stays under an explicit budget.

---

## Finding 18: No validated user segment exists for `CrewAIFlowAdapter`

**Severity:** MEDIUM
**Confidence:** MEDIUM
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:39-86`, Appendix A `:977-985`
**Consensus:** 3/5

### Evidence

The Problem section names "Maya" but Maya is invented. The spec's own Appendix A admits the only known real-world demand for this routing shape switched to a different adapter. The Phase 1-5 orchestration logic exists to serve a hypothetical user segment.

### Recommendation

Either land a single concrete user (a feature request, an issue, a public CrewAI project that hit this wall) or scope down. Phase 0 is independently valuable regardless of the Flow adapter. If the Flow adapter has no user, ship Phase 0 alone and defer the rest until demand surfaces.

---

## Finding 19: Phase 0 disruption is underestimated — 700-line refactor of working code

**Severity:** MEDIUM
**Confidence:** MEDIUM
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:402-479`, acceptance `:893`
**Consensus:** 4/5

### Evidence

Phase 0 deletes ~700 lines from the working `CrewAIAdapter` and rebuilds them through a new function with new injection points. The acceptance check is "existing tests pass unchanged." Existing tests cover behavior, not all the timing and error-path quirks of `nest_asyncio` + `ContextVar` + sync-to-async bridging. Subtle changes in patch timing or closure capture could cause flake-level regressions that pass tests but fail in production.

### Recommendation

Add to Phase 0: a soak test running the existing CrewAI agent path through 100 message turns with concurrent rooms, asserting no exceptions and no event-loop policy mutations. Without it, "tests pass unchanged" is a weak guarantee for a refactor of this size.

---

## Finding 20: 32-bit correlation token derived from public input

**Severity:** LOW
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:382-385`
**Consensus:** 2/5

### Evidence

Token is `sha256(side_effect_key)[:8]` (32 bits). Both inputs (`run_id` = public message_id, `normalized_target_key` = room participant) are observable to all room participants. A peer can compute the same token. A malicious peer could embed a different delegation's token in their reply to spoof matching or force `reply_ambiguous`.

### Recommendation

Document the token as a cooperative disambiguator, not a security control. The `reply_ambiguous` fail-closed already neutralizes most spoofing paths. If the threat model expects untrusted peers, increase to 128 bits and derive from a per-adapter HMAC secret. For v1 with cooperative peers, document and move on.

---

## Finding 21: `max_delegation_rounds` semantics undefined

**Severity:** LOW
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:294`, `:292`
**Consensus:** 5/5

### Evidence

Spec defines a numeric limit but never says what counts as a round. Per inbound message? Per run lifetime? Per `delegation_id`? A Flow that returns `delegations: [a, b, c, d, e]` in one turn — does that consume 5 rounds or 1?

### Recommendation

Define: "A round is one `Flow.kickoff_async` call that returns a `delegate` decision. The counter increments once per round, regardless of how many delegations are in the round. Cap is per `run_id`." Test the boundary.

---

## Finding 22: Task event durability assumed not verified

**Severity:** LOW
**Confidence:** MEDIUM
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:843`, `:803`
**Consensus:** 4/5

### Evidence

The spec relies entirely on Thenvoi task events being durable. It never checks: do task events have a TTL? Are they pruned? If a peer's reply arrives 30 days after the parent and the platform has pruned older task events, the orchestration state is gone and the reply becomes a new input.

### Recommendation

Add a Resolved Question: "What is the task event retention guarantee on Thenvoi?" If unbounded, document. If bounded, document the bound and add a configuration knob `max_run_age` that proactively closes (records `failed` with code `run_aged_out`) any run older than retention minus a safety margin.

---

## Finding 23: `nest_asyncio` patch surface remains accessible to Flow-only users

**Severity:** LOW
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:431`
**Consensus:** 3/5

### Evidence

Phase 0 moves the `nest_asyncio` patch to `integrations/crewai/runtime.py`. Flow-only users (who never call `runtime.create_crewai_tools()`) load the runtime module but don't fire the lazy patch. Verify the patch is gated only inside `run_async`, not at module import time.

### Recommendation

Add an acceptance criterion: "Importing `CrewAIFlowAdapter` and running one `direct_response` turn does not invoke `nest_asyncio.apply`. Verified by patching `nest_asyncio.apply` and asserting it is never called."
