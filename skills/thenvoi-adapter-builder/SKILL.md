---
name: thenvoi-adapter-builder
description: Research a vendor SDK or product, choose the right Thenvoi adapter shape, implement the integration, wire repo parity, and review it against the rest of this SDK.
---

# Thenvoi Adapter Builder

Use this skill when the task is to add or review a framework integration in `thenvoi-sdk-python`.

The job is:
- read the target vendor's official docs and SDK
- work out how that product should fit the Thenvoi platform
- choose the closest adapter archetype in this repo
- build the adapter, converter, wiring, tests, and examples
- validate that it reaches parity with the rest of the SDK

Typical prompts:
- "Add a Thenvoi adapter for CrewAI."
- "Take this SDK and build the Thenvoi integration."
- "Review whether this adapter matches the rest of the SDK."
- "Research this product page and implement the right Thenvoi adapter."

## What a finished integration usually includes

- `src/thenvoi/adapters/<framework>.py`
- `src/thenvoi/converters/<framework>.py`, or a deliberate non-standard converter treatment
- exports in `src/thenvoi/adapters/__init__.py` and `src/thenvoi/converters/__init__.py`
- optional dependency wiring in `pyproject.toml`
- conformance registration in `tests/framework_configs/`
- framework-specific tests in `tests/adapters/` and `tests/converters/`
- examples under `examples/<framework>/`
- docker or helper files when that framework needs them

Read `AGENTS.md` first, then use the references bundled with this skill.

## Workflow

### 1. Research the vendor from primary sources

Use the official product docs, SDK reference, and API docs first.

Capture these facts before you code:
- conversation model
- tool-calling model
- async or sync behavior
- streaming support
- session or state model
- retry and failure surfaces
- whether the SDK executes tools itself or lets Thenvoi stay in control

If the target is not a normal chat SDK, write that down early. Some integrations are protocol-, session-, or metadata-shaped.

### 2. Choose the closest Thenvoi archetype

Read `references/repo_adapter_matrix.md` and `references/design_decision_playbook.md` and pick the nearest sibling before you design anything.

Common matches:
- `anthropic` or `gemini` for direct SDK plus manual tool loop
- `pydantic_ai` for framework-native tool registration
- `crewai` or `parlant` for framework-owned agent/session models
- `claude_sdk`, `letta`, or `codex` for persistent or server-managed state
- `a2a` or `a2a_gateway` for protocol-shaped integrations

Do not invent a new adapter pattern if a sibling already solves the same shape.

While choosing the shape, decide these two things explicitly:
- adapter archetype: direct SDK, framework-native tools, orchestration framework, persistent session, or protocol bridge
- converter archetype: standard history, sender-preserving history, or metadata/session-state only

If the vendor has an official SDK and this repo already replaced a simulated version of a similar product, start from the official SDK path unless the docs clearly show it is unusable.

If the vendor offers automatic tool execution, disable it when you can and keep Thenvoi as the executor unless the product contract is inherently server-side, as with Letta MCP or Codex app-server flows.

### 3. Check repo history and review context

For real adapter work, inspect:
- `git log -- <relevant files>`
- related PRs and review comments
- `cass search "<framework> adapter thenvoi" --robot --limit 5 --fields minimal`
- Linear issues when they exist

Use that pass to answer:
- what the first version got right
- what review pushed back on
- what parity fixes landed later
- whether the repo later replaced an earlier design with a more native one

Use these references for the history pass:
- `references/adapter_history.md`
- `references/review_patterns.md`
- `references/review_failure_catalog.md`

If the adapter is session-managed, server-managed, or approval-driven, read the
closest entries in `references/review_failure_catalog.md` before you edit
anything. At minimum, check the sections for `codex`, `letta`, and `opencode`
shapes when they are relevant.

### 4. Write a short implementation plan

Before editing, write down:
- target archetype
- converter shape
- tool model
- state model
- required repo wiring
- validation sequence

If the work is large or the user asked for it, share the short plan before the main implementation.

### 5. Build the integration end to end

Follow the repo's TDD flow from `AGENTS.md`:
- scaffold source files first
- wire conformance second
- add or restore tests third
- implement converter before adapter when that makes sense

For shared files such as `pyproject.toml`, `__init__.py`, and `tests/framework_configs/*.py`, patch only the framework-related hunks.

For framework-specific files that do not exist yet, create the full set the integration needs:
- source files
- tests
- examples
- helper or docker files when the integration depends on them

### 6. Validate in layers

Bootstrap the repo copy you are working in before running repo-local checks:

```bash
uv sync --extra dev
```

Then run focused checks:

```bash
uv run pytest tests/framework_conformance/test_config_drift.py -v
uv run pytest tests/framework_conformance/test_adapter_conformance.py -v -k "<framework>"
uv run pytest tests/framework_conformance/test_converter_conformance.py -v -k "<framework>"
uv run pytest tests/adapters/test_<framework>_adapter.py tests/converters/test_<framework>.py -v
```

Then run broader validation as needed:

```bash
uv run ruff check .
uv run ruff format . --check
uv run pyrefly check
uv run pytest tests/ --ignore=tests/integration/ -v
```

If a framework is intentionally outside standard converter conformance, validate the exclusion and run the right framework-specific tests instead.

For session-managed or server-managed adapters, conformance is only the floor.
Add focused regression tests for the real control flow:
- manual approval and question follow-up messages can run while a turn is active
- prompt submission failures do not leave the room stuck in a busy state
- session loss or remote 404 recovery preserves prior room context instead of
  silently dropping it
- cleanup is safe when a turn task, timeout task, or stream task is still live
- turn-complete delivery does not race the caller returning early; if a watcher
  task sends fallback text or final error events, test that `on_message()`
  does not return before those side effects are visible

### 7. Review parity before finishing

Use this checklist after the code already works:
- compare constructor shape, logging style, and tool handling with 2 sibling adapters
- check exports, optional deps, and conformance builders
- check examples against repo conventions, including PEP 723 and `load_agent_config(...)`
- check cleanup, restart, and multi-room behavior
- check tool-call IDs and best-effort reporting
- check shared helpers before keeping bespoke logic
- check that the integration includes the same platform parity expected elsewhere in the repo
- check whether the chosen shape still matches the vendor docs after implementation, or whether the code drifted back toward a fake prompt shim
- check room execution semantics against Thenvoi's runtime model, not only the
  vendor SDK model

### 7.5. Example conventions review

Before finishing any adapter PR, inspect the examples as if they were part of
the API surface.

Check these specifically:
- examples should prefer `load_agent_config(...)` for Thenvoi credentials
- do not add `python-dotenv` just to load Thenvoi URLs in examples
- do not require `THENVOI_WS_URL` or `THENVOI_REST_URL` in examples unless the
  example is explicitly about overriding platform endpoints
- if `Agent.create(...)` already has sane hosted defaults, let the example use
  them instead of threading extra env vars through every sample
- keep PEP 723 dependencies minimal; do not add optional packages that the
  example does not actually need
- if a docker or helper example needs URL overrides, prefer documented defaults
  over fail-fast env checks unless local-only behavior really requires them
- run the relevant formatter or pre-commit hooks against new helper files, not
  just the adapter code

### 8. Session-managed adapter guardrails

Use this section for adapters shaped like `codex`, `letta`, `claude_sdk`, or
`opencode`, or any adapter that keeps remote sessions, asks follow-up
questions, or waits for permissions.

Before shipping, answer these explicitly:
- Does `on_message()` hold the room hostage while waiting for a remote approval,
  question, or long-lived session event?
- Can a queued follow-up user message reach the control handler while the
  previous turn is still open?
- If prompt submission fails after per-room turn state is allocated, does the
  adapter clear that state right away?
- If the remote session disappears, does the next turn rebuild enough context to
  keep the conversation coherent?
- Are timeout tasks, background turn tasks, and cleanup paths all idempotent?

If any answer is unclear, write the regression test before you trust the code.

For these adapters, add tests that prove:
- the first turn returns after a manual approval or question prompt is emitted
- a second message in the same room can resolve that prompt
- the final assistant reply still arrives after the follow-up control message
- a failed `prompt_async` or equivalent call does not trigger a permanent
  "still processing" state
- the final assistant reply or fallback error is observable before
  `on_message()` returns when the turn already reached completion
- session rehydration on remote loss includes replayed history or an equivalent
  recovery path

### 9. Real validation beats shim validation

If the framework is installed locally or can run on a local server, exercise the
real path at least once before calling the adapter done.

Prefer this order:
1. raw SDK or HTTP smoke test against the real vendor runtime
2. adapter smoke test in this repo against the same runtime
3. focused unit and conformance coverage
4. repo review pass with `thenvoi-review-pr` or the closest repo review skill

Do not stop at mocked tests if the repo and machine already have the real tool.
If CI runs `pre-commit`, run `uv run pre-commit run --all-files` locally before
you push. Repo-local `ruff check` alone is not enough.

## Repo rules that keep coming up in review

- use the official SDK when possible
- keep Thenvoi in control of tool execution unless the framework architecture makes that impossible or clearly wrong
- catch `pydantic.ValidationError` separately
- use lazy imports for optional deps
- keep cleanup idempotent
- avoid global locks that serialize unrelated rooms
- do not let reporting failures break the main turn
- do not stop after source files compile; repo wiring matters

## Known traps in this repo

- conformance is necessary but not sufficient
- standard conformance is the default, but `codex`, `letta`, `a2a`, and `a2a_gateway` show when explicit exclusions are the correct move
- metadata-only converters should not be forced into standard converter conformance
- examples fail review on small convention misses
- missing tool parity is a common late review failure
- shared helpers exist for tool parsing, naming, and schema generation
- over-broad locks, global flags, and room-id-in-prompt hacks usually fail review
- if the vendor framework already stores state or sessions, do not rebuild that state machine in Thenvoi unless the repo history shows a concrete reason
- server-managed adapters often fail after the first green test pass because of
  turn lifecycle races, especially when event-loop watcher tasks deliver the
  final room output
- examples are part of review scope; extra dotenv wiring and unnecessary
  platform URL plumbing create avoidable follow-up churn

## How to research a new vendor in depth

Before writing code, spend real time understanding the vendor:

1. Read the official SDK docs end to end, not just the quickstart
2. Find the tool/function calling docs specifically — this drives the adapter shape
3. Check if the SDK supports async, streaming, and session persistence
4. Look for SDK-level automatic tool execution and whether it can be disabled
5. Find the role/turn formatting rules — some providers reject consecutive same-role messages
6. Search for known limitations, rate limits, and retry semantics
7. If an official SDK exists, prefer it over HTTP wrappers or community packages
8. If the product manages remote state (sessions, threads, agents), that changes the adapter shape fundamentally

After understanding the vendor, read the 2-3 closest sibling adapters in this repo.
Then read the review feedback those adapters received (see `references/review_failure_catalog.md`).

## Files in this skill

- `references/build_checklist.md`: repo-parity checklist
- `references/design_decision_playbook.md`: how to choose the right adapter and converter shape
- `references/adapter_history.md`: why the current adapter families look the way they do
- `references/repo_adapter_matrix.md`: map of adapter families in this repo
- `references/review_patterns.md`: recurring review findings and high-signal PRs
- `references/review_failure_catalog.md`: verbatim reviewer comments from every adapter PR
- `references/adapter_matrix.md`: adapter lineage and closest analogs
