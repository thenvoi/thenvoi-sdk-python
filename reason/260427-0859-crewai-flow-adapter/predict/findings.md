# Findings - CrewAI Flow Adapter Spec

## Finding 1: Non-bootstrap peer replies will not see prior task events

**Severity:** CRITICAL
**Confidence:** HIGH
**Location:** `src/thenvoi/preprocessing/default.py:78`
**Consensus:** 5/5 personas

**Evidence:**

`DefaultPreprocessor.process` sets `is_bootstrap = not ctx.is_llm_initialized`, initializes `raw_history = []`, and only calls `_load_history(ctx, msg)` when `is_bootstrap` is true at `src/thenvoi/preprocessing/default.py:78-85`. It then passes `HistoryProvider(raw=raw_history)` into the adapter at `src/thenvoi/preprocessing/default.py:96-103`. `SimpleAdapter.on_event` converts only that supplied history before calling `on_message` at `src/thenvoi/core/simple_adapter.py:137-155`.

The spec depends on the opposite behavior. It says later peer replies trigger another `on_message`, reconstruct state from task events, and run the Flow again with matched reply state at `reason/260427-0859-crewai-flow-adapter/technical-spec.md:87-116`. It also says the converter reconstructs the latest run state by scanning task events in history at `technical-spec.md:207`.

**Why this fails the Genpact goal:**

Scenario 3 and Scenario 4 rely on later peer replies seeing the prior delegation task events. Today, those later turns will not receive prior task events unless the adapter keeps local memory or the runtime hydrates state again. That means restart reconstruction, cross-process recovery, and even normal non-bootstrap reply matching can fail.

**Recommendation:**

Add an explicit state hydration contract before implementing `CrewAIFlowAdapter`. Options: add a runtime `TaskEventStateProvider` the adapter can query every turn; add an adapter feature declaring it needs task-event replay on every message; or change preprocessing to hydrate task events for stateful adapters. The spec should not claim durable reply matching until a non-bootstrap peer reply can reconstruct prior delegations from persisted events.

**Persona Votes:**

| Persona | Vote | Note |
|---------|------|------|
| Architecture Reviewer | confirm | This breaks the core state-source assumption. |
| Reliability Engineer | confirm | Restart and later reply behavior fail without hydration. |
| SDK DX Reviewer | confirm | Users will not understand why durable state is absent on later turns. |
| Test Strategy Reviewer | confirm | Tests must simulate non-bootstrap peer replies with empty history. |
| Devil's Advocate | confirm | This is the strongest reason the spec is currently bad. |

## Finding 2: Visible sends are not idempotent or atomic with state events

**Severity:** CRITICAL
**Confidence:** HIGH
**Location:** `src/thenvoi/runtime/tools.py:1194`
**Consensus:** 5/5 personas

**Evidence:**

`AgentTools.send_message` accepts only `content` and `mentions` at `src/thenvoi/runtime/tools.py:1194-1196` and sends `ChatMessageRequest(content=content, mentions=mention_items)` at `src/thenvoi/runtime/tools.py:1249-1252`. `send_event` is the method that accepts `metadata` at `src/thenvoi/runtime/tools.py:1258-1263`.

The spec orders side effects as: check reconstructed state, send visible message, then record a task event with the returned message id. It explicitly says a crash between visible send and task-event write may resend a duplicate because the platform has no idempotency key at `reason/260427-0859-crewai-flow-adapter/technical-spec.md:280-286`.

**Why this fails the Genpact goal:**

Duplicate delegation or duplicate final synthesis is one of the safety-net failures. Treating it as a documented v1 crash window may be acceptable for an experimental adapter, but it is not enough to remove demo safety code.

**Recommendation:**

Change the spec to use a reservation/fence before visible sends: record `delegation_reserved` or `finalization_reserved` with a deterministic operation key, send the visible message, then record `*_sent` with the platform message id. On retry, an unresolved reservation should fail closed or emit an indeterminate-state diagnostic, not resend blindly. Better: add platform idempotency keys or an atomic visible-message-plus-metadata operation.

**Persona Votes:**

| Persona | Vote | Note |
|---------|------|------|
| Architecture Reviewer | confirm | Task events alone cannot prove visible side-effect state. |
| Reliability Engineer | confirm | This is a hard blocker for retry/restart safety. |
| SDK DX Reviewer | confirm | The public guarantee would be misleading. |
| Test Strategy Reviewer | confirm | Crash-window tests need failure-injecting fake tools. |
| Devil's Advocate | confirm | The spec admits the failure mode it claims to remove. |

## Finding 3: Raw runtime tools can bypass adapter safety policies

**Severity:** CRITICAL
**Confidence:** MEDIUM
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:388`
**Consensus:** 4/5 personas

**Evidence:**

The spec adds a `CrewAIFlowRuntime` object and a `get_current_flow_runtime()` helper so Flow methods can access runtime objects without storing them in `flow.state` at `technical-spec.md:387-390`. Existing CrewAI tools can call `tools.send_message` directly through a context-bound tool bridge, as shown by the current adapter's tool pattern around `src/thenvoi/adapters/crewai.py:300-338` and the `thenvoi_send_message` path later in that file.

If a Flow method can call raw `runtime.tools.send_message(...)`, the adapter cannot enforce `require_delegation_before_final`, join checks, duplicate suppression, or finalization state. A Flow could emit a visible partial answer and still return `waiting` or `delegate`.

**Recommendation:**

Expose a policy-safe runtime facade, not raw `AgentTools`. For v1, visible writes should only happen through adapter decisions. Runtime helpers may allow read-only participant/contact/context operations. Any write helper should reserve state and pass through the same side-effect executor used for `delegate` and `synthesize` decisions.

**Persona Votes:**

| Persona | Vote | Note |
|---------|------|------|
| Architecture Reviewer | confirm | Two side-effect paths break the abstraction. |
| Reliability Engineer | confirm | Bypasses idempotency and join policy. |
| SDK DX Reviewer | partial | Runtime tools are useful, but the write boundary must be explicit. |
| Test Strategy Reviewer | confirm | Tests must prove Flow code cannot send visible messages directly. |
| Devil's Advocate | confirm | If direct tools are allowed, this is just another safety net. |

## Finding 4: Ambiguous reply matching should fail closed

**Severity:** HIGH
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:273`
**Consensus:** 5/5 personas

**Evidence:**

The matching rules say if the sender matches more than one pending run, the adapter chooses the run with the newest parent message timestamp and records `reply_ambiguous` at `technical-spec.md:273-277`. Since visible chat messages cannot carry run metadata through `send_message`, the sender identity is the primary matching signal.

**Why this fails the Genpact goal:**

The same peer can have multiple pending runs in a busy room. Choosing the newest run can attach a stale reply to the wrong parent request, causing a plausible but incorrect final synthesis.

**Recommendation:**

Do not mutate state on ambiguous sender matches. Record `reply_ambiguous`, keep candidate delegations pending, and require deterministic correlation. If the platform cannot provide reply/thread metadata, include a short opaque operation token in delegation content and require it for matching when a peer has more than one pending delegation.

**Persona Votes:** all five confirmed.

## Finding 5: Task-event persistence conflicts with optional `AdapterFeatures.emit`

**Severity:** HIGH
**Confidence:** HIGH
**Location:** `src/thenvoi/core/types.py:29`
**Consensus:** 4/5 personas

**Evidence:**

`Emit.TASK_EVENTS` is an optional emit feature at `src/thenvoi/core/types.py:29-35`, and `AdapterFeatures.emit` defaults to an empty frozen set at `src/thenvoi/core/types.py:48-49`. `SimpleAdapter.on_started` warns about unsupported emit values at `src/thenvoi/core/simple_adapter.py:115-129`.

The spec makes task events mandatory: every state transition writes a task event, and durable state lives in task-event metadata. But the public constructor still accepts normal `features` at `technical-spec.md:213-225` without saying whether task-state events are internal and mandatory or governed by `Emit.TASK_EVENTS`.

**Recommendation:**

Separate internal orchestration persistence from optional user-facing event emission. Either emit state events regardless of `features.emit`, with explicit documentation, or force-enable/require `Emit.TASK_EVENTS` and reject incompatible feature settings. The adapter should declare `SUPPORTED_EMIT` accordingly.

**Persona Votes:** Architecture, Reliability, SDK DX, Devil's Advocate confirmed; Test Strategy marked as probable but requiring implementation details.

## Finding 6: Terminal Flow output shape is still too implicit

**Severity:** HIGH
**Confidence:** MEDIUM
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:255`
**Consensus:** 3/5 personas

**Evidence:**

The spec now says each Flow must return one terminal decision dictionary at `technical-spec.md:255`, and the decision table lists `direct_response`, `delegate`, `waiting`, `synthesize`, and `failed` at `technical-spec.md:259-265`. It still does not show exact JSON with a discriminator field such as `decision`, `type`, or `kind`.

**Recommendation:**

Define exact Pydantic output models. Example: `CrewAIFlowDirectResponse(decision="direct_response", content: str, mentions: list[str])`. Require all examples, tests, and malformed-output handling to use the exact discriminator field.

## Finding 7: Participant identity normalization needs structured snapshots, not strings

**Severity:** HIGH
**Confidence:** MEDIUM
**Location:** `src/thenvoi/runtime/formatters.py:95`
**Consensus:** 4/5 personas

**Evidence:**

`build_participants_message` formats a string for the LLM and instructs use of exact handles at `src/thenvoi/runtime/formatters.py:95-107`. The spec relies on participant identity normalization for UUIDs, namespaced handles, bare handles, and display names at `technical-spec.md:473-477`.

A string participant message is not enough to implement reliable normalization. The adapter needs structured participant data with id, handle, display name, sender type, and normalized key.

**Recommendation:**

Add `ParticipantSnapshot` to the Flow input and task metadata. Refresh or receive structured participants before matching replies and enforcing tagged-peer policies. Do not base normalization on `participants_msg`.

## Finding 8: Tests are not strong enough to prove safety-net removal

**Severity:** HIGH
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:579`
**Consensus:** 5/5 personas

**Evidence:**

The spec says unit tests use mocked CrewAI modules and fake `AgentTools` at `technical-spec.md:579-583`. It excludes real API E2E tests at `technical-spec.md:654`. Current `FakeAgentTools` always succeeds for `send_message` and `send_event` at `src/thenvoi/testing/fake_tools.py:58-82`.

The safety-net failures are about ordered visible messages, failed state writes, duplicate retries, non-bootstrap replay, ambiguous replies, and concurrent processing. Current fakes and compile/lint example checks do not exercise those failure modes.

**Recommendation:**

Add a purpose-built `FlowFakeAgentTools` with injectable send failures, stable ids, event-to-history replay, and ordered trace assertions. Add tests for non-bootstrap peer reply with empty history, two adapter instances racing on the same run, crash after visible send, out-of-order task events, and exact ordered Genpact Scenario 3/4 traces.

## Finding 9: The spec may be putting a framework-neutral router inside a CrewAI adapter

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `reason/260427-0859-crewai-flow-adapter/technical-spec.md:143`
**Consensus:** 4/5 personas

**Evidence:**

The safety requirements are generic router invariants: no answer before named delegates run, no synthesis before required peers reply, do not lose peer replies, normalize identities, and avoid silent turns at `technical-spec.md:143`. The approaches table considers a framework-neutral orchestrator and rejects it as a larger product decision at `technical-spec.md:153`.

**Recommendation:**

Either narrow the spec to a CrewAI-specific proof-of-concept, or extract the room-orchestration state machine into a framework-neutral layer with CrewAI Flow as one decision engine. If the immediate need is Genpact, keep the CrewAI plan but stop describing generic router safety as CrewAI-specific.

## Finding 10: Example plan conflicts with current numbering and could oversell readiness

**Severity:** MEDIUM
**Confidence:** HIGH
**Location:** `examples/crewai/README.md:18`
**Consensus:** 3/5 personas

**Evidence:**

The current README already lists `06_jerry_agent.py` and `07_contact_and_memory_agent.py` at `examples/crewai/README.md:18-26`. The spec proposes `examples/crewai/06_flow_router.py` at `technical-spec.md:541-543` and acceptance criteria repeat that filename at `technical-spec.md:729-734`. The example is only required to compile/lint, not prove safety behavior.

**Recommendation:**

Use `08_flow_router.py` or `flow_router.py`. Make it a deterministic offline simulation or label it explicitly as illustrative. Add README language that `CrewAIAdapter` remains right for simple role/backstory agents, while `CrewAIFlowAdapter` is only for room routers whose correctness depends on pending peer sets and joins.
