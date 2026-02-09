# Framework verification: tests to run

> **TL;DR** `uv run pytest tests/ --framework <name> -v`

Use this list to verify that your **adapter** and **converter** implement the Thenvoi agent framework contract correctly.

---

## Mandatory tests to add a new framework (checklist)

To add a new framework, it **must** pass all applicable tests below. There are two layers:

1. **Parameterized conformance tests** – All mandatory tests that can be expressed generically live in [`tests/framework_conformance/`](framework_conformance/). Same test code runs for every framework; behavior differences (e.g. empty sender, own-message filtering) are expressed via config. You must register your framework in [`tests/framework_configs/`](framework_configs/) with the correct behavior and **pass** these tests (or be explicitly skipped only where the contract allows).
2. **Framework-specific coverage** – The remaining areas that require per-framework mocks or APIs (bootstrap, invoke/response, error handling, etc.) **must** be implemented in [`tests/adapters/test_<framework>_adapter.py`](adapters/) and [`tests/converters/test_<framework>.py`](converters/). The assertion logic can differ per framework; the coverage is mandatory.

### Adapter: mandatory parameterized tests (must pass)

Run: `uv run pytest tests/framework_conformance/test_adapter_conformance.py -k "<your_framework_id>" -v`

| # | Test (class::method)                                               | Contract                                                               |
| - | ------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| 1 | `TestAdapterInitialization::test_default_initialization`         | Default attribute values match config.                                 |
| 2 | `TestAdapterInitialization::test_custom_initialization`          | Custom kwargs are accepted and stored (skipped if no custom kwargs).   |
| 3 | `TestAdapterInitialization::test_defaults_to_empty_custom_tools` | Custom tools attribute is `[]` by default (skipped if no such attr). |
| 4 | `TestAdapterInitialization::test_has_history_converter`          | Adapter has non-null `history_converter` (skipped if not exposed).   |
| 5 | `TestAdapterOnStarted::test_after_on_started_sets_agent_name_and_description` | After `on_started(agent_name, agent_description)`, adapter has them set (skipped if live client required, e.g. PydanticAI). |
| 6 | `TestAdapterCleanup::test_cleanup_nonexistent_room_is_safe`      | `on_cleanup("nonexistent-room")` does not raise.                     |
| 7 | `TestAdapterCleanup::test_cleanup_all_safe_when_supported`      | If adapter has `cleanup_all()`, calling it does not raise (skipped if no such method). |

### Adapter: mandatory framework-specific coverage (must implement)

These cannot be expressed as generic parameterized tests (they need per-framework mocks or APIs). Each new adapter **must** have tests (in `tests/adapters/test_<framework>_adapter.py`) that cover:

| #  | Area                                                      | What must be tested                                                                                                 |
| -- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1  | **OnMessage – bootstrap**                          | First message in a room initializes history and triggers model invoke.                                              |
| 2  | **OnMessage – existing history**                   | When history is provided (e.g. on bootstrap), it is loaded and used for the next invoke.                            |
| 3  | **OnMessage – invoke and response**                | Adapter invokes the underlying model and produces a response (flow can be mocked).                                  |
| 4  | **OnCleanup (used room)**                          | Cleaning up a room that was used removes or resets state; no crash.                                                 |
| 5  | **Error handling**                                  | When the model or tools raise, the adapter handles it (no unhandled crash; clear tools or reset state if required). |
| 6  | **Custom tools** (if supported)                     | Platform tools and custom tools are merged/registered correctly.                                                    |
| 7  | **Tool execution** (if adapter uses platform tools) | `execute_tool_call` is invoked correctly and result is fed back into the flow.                                    |
| 8  | **Streaming** (if adapter streams)                  | Stream events are produced and handled (e.g. stream start/end, chunks).                                             |
| 9  | **Execution reporting** (if supported)              | Enable/disable and reporting behavior are tested.                                                                   |

### Converter: mandatory parameterized tests (must pass)

Run: `uv run pytest tests/framework_conformance/test_converter_conformance.py -k "<your_framework_id>" -v`

| #  | Test (class::method)                                               | Contract                                                                                                |
| -- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| 1  | `TestUserTextMessages::test_converts_user_text_with_sender_name` | User text appears as `[Sender]: content`.                                                             |
| 2  | `TestUserTextMessages::test_handles_empty_sender_name`           | Empty `sender_name` handled per config (`content_as_is` / `brackets_empty` / `unknown_prefix`). |
| 3  | `TestUserTextMessages::test_handles_missing_sender_name`         | Missing `sender_name` handled per config (skipped if `has_missing_sender_name_test=False`).         |
| 4  | `TestEmptyHistory::test_empty_history`                           | `convert([])` returns config `empty_result`.                                                        |
| 5  | `TestMessageTypeDefaults::test_defaults_to_text_message_type`    | No `message_type` → treated as text.                                                                 |
| 6  | `TestThoughtMessageSkipping::test_skips_thought_messages`        | `message_type: "thought"` → not in output.                                                           |
| 7  | `TestOwnMessageFiltering::test_own_message_handling`             | Own assistant messages filtered or included per config.                                                 |
| 8  | `TestOwnMessageFiltering::test_includes_other_agents_messages`   | Other agents’ messages always included.                                                                |
| 9  | `TestOwnMessageFiltering::test_skips_only_own_keeps_others`      | Only own messages filtered; others kept.                                                                |
| 10 | `TestOwnMessageFiltering::test_set_agent_name_updates_filtering` | `set_agent_name` changes which messages are filtered.                                                 |
| 11 | `TestOwnMessageFiltering::test_includes_all_when_no_agent_name`  | No agent name set → all assistant messages included.                                                   |
| 12 | `TestEdgeCases::test_handles_empty_content`                      | Empty `content` skipped or kept per config.                                                           |
| 13 | `TestEdgeCases::test_defaults_to_user_role`                      | No `role` → treated as user (skipped if `has_role_concept=False`).                                 |
| 14 | `TestToolEventHandling::test_tool_events_skipped_for_simple_converters` | If `skips_tool_events=True`, tool_call/tool_result are omitted (skipped otherwise). |
| 15 | `TestToolEventConversion::test_converts_tool_call_to_framework_format`   | Tool_call (and paired tool_result) are converted and appear in output (skipped if `skips_tool_events`). |
| 16 | `TestToolEventConversion::test_converts_tool_result_paired_with_call`    | Tool_result is converted and paired with tool_call (skipped if `skips_tool_events`). |
| 17 | `TestToolEventConversion::test_mixed_history_includes_user_assistant_tool_messages` | User + assistant + tool_call + tool_result converted in order (skipped if `skips_tool_events`). |

### Converter: mandatory framework-specific coverage (must implement)

Tool call/result and mixed history are covered by the parameterized tests above when `skips_tool_events=False`. Each new converter **must** still have tests (in `tests/converters/test_<framework>.py`) that cover:

| # | Area                                    | When                                                          | What must be tested                                                                                                                   |
| - | --------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | **Tool batching** (if applicable) | Converter processes tool events and batches multiple tool calls | Multiple consecutive `tool_call` messages are batched according to framework rules.                                                 |
| 2 | **Framework-specific behavior**   | Any framework-specific rule                                   | At least one test that asserts framework-specific behavior (e.g. tool call pairing, error handling, special message types, workflow). |

Converters that **skip** tool events (`skips_tool_events=True`, e.g. CrewAI, Parlant) do not need tool-event tests; they must still pass the parameterized tests and any framework-specific behavior tests.

---

## Quick reference

**Run only your framework’s tests (recommended):**

```bash
uv run pytest tests/ --framework <framework_name> -v
```

Valid `framework_name`: `anthropic`, `langgraph`, `crewai`, `claude_sdk`, `pydantic_ai`, `parlant`.

This runs **only** that framework’s conformance tests (adapter + converter) and its framework-specific adapter and converter test files. **What `--framework` runs (all required for framework verification):**

| What runs | Description |
|-----------|-------------|
| Adapter conformance | All parametrized tests in `test_adapter_conformance.py` for your `[adapter_id]` (7 tests). |
| Converter conformance | All parametrized tests in `test_converter_conformance.py` for your `[converter_id]` (17 tests). |
| Framework-specific adapter | Entire file `tests/adapters/test_<framework>_adapter.py`. |
| Framework-specific converter | Entire file `tests/converters/test_<framework>.py`. |

Not included (and not required for framework verification): other frameworks' slices, shared utility tests (`test_a2a_gateway.py`, `test_tool_parsing.py`), smoke/session/platform/runtime tests, integration tests. You do **not** need to run the full suite when adding or changing a framework.

---

## 1. Run conformance tests for your framework (without --framework)

Conformance tests are parameterized over all registered frameworks. Run only your framework’s slice:

```bash
# Adapter conformance (use your adapter framework_id)
uv run pytest tests/framework_conformance/test_adapter_conformance.py -k "<framework_id>" -v

# Converter conformance (use your converter framework_id)
uv run pytest tests/framework_conformance/test_converter_conformance.py -k "<framework_id>" -v
```

**Adapter framework_ids:** `anthropic`, `langgraph`, `crewai`, `claude_sdk`, `pydantic_ai`, `parlant`
**Converter framework_ids:** `anthropic`, `langchain`, `crewai`, `claude_sdk`, `pydantic_ai`, `parlant`
*(Note: LangGraph adapter uses converter `langchain`. The canonical mapping lives in `CONVERTER_ID_FOR_ADAPTER` in [`tests/framework_configs/__init__.py`](framework_configs/__init__.py).)*

---

## 2. Run framework-specific tests

Run the dedicated test files for your adapter and converter:

| Framework  | Adapter tests                                  | Converter tests                          |
| ---------- | ---------------------------------------------- | ---------------------------------------- |
| Anthropic  | `tests/adapters/test_anthropic_adapter.py`   | `tests/converters/test_anthropic.py`   |
| LangGraph  | `tests/adapters/test_langgraph_adapter.py`   | `tests/converters/test_langchain.py`   |
| CrewAI     | `tests/adapters/test_crewai_adapter.py`      | `tests/converters/test_crewai.py`      |
| Claude SDK | `tests/adapters/test_claude_sdk_adapter.py`  | `tests/converters/test_claude_sdk.py`  |
| PydanticAI | `tests/adapters/test_pydantic_ai_adapter.py` | `tests/converters/test_pydantic_ai.py` |
| Parlant    | `tests/adapters/test_parlant_adapter.py`     | `tests/converters/test_parlant.py`     |

```bash
# Example: verify Anthropic
uv run pytest tests/adapters/test_anthropic_adapter.py tests/converters/test_anthropic.py -v
```

---

## 3. One-shot: “Is my framework implemented correctly?”

**Preferred:** use `--framework` so only that framework's tests run:

```bash
uv run pytest tests/ --framework anthropic -v
uv run pytest tests/ --framework langgraph -v
uv run pytest tests/ --framework crewai -v
# etc.
```

**Alternative** (no `--framework`): run conformance + framework-specific files and filter by id:

```bash
# Anthropic
uv run pytest \
  tests/framework_conformance/ \
  tests/adapters/test_anthropic_adapter.py \
  tests/converters/test_anthropic.py \
  -k "anthropic" -v

# LangGraph (adapter id langgraph, converter id langchain)
uv run pytest \
  tests/framework_conformance/test_adapter_conformance.py -k "langgraph" \
  tests/framework_conformance/test_converter_conformance.py -k "langchain" \
  tests/adapters/test_langgraph_adapter.py \
  tests/converters/test_langchain.py \
  -v
```

---

## 4. Adding a new framework

1. Implement adapter and converter (and register in `src/thenvoi/`).
2. Add `AdapterConfig` to [`tests/framework_configs/adapters.py`](framework_configs/adapters.py) and `ConverterConfig` to [`tests/framework_configs/converters.py`](framework_configs/converters.py) (including an `OutputAdapter` if the converter returns a custom shape). If the adapter's converter has a different framework_id, add an entry to `CONVERTER_ID_FOR_ADAPTER` in [`tests/framework_configs/__init__.py`](framework_configs/__init__.py).
3. The `--framework <name>` run map is derived automatically from the config registries in `tests/conftest.py`. No manual entry needed — just ensure your `AdapterConfig.framework_id` matches the name you want to use.
4. Run conformance tests for your new `framework_id`; fix behavior or config until they pass.
5. Add `tests/adapters/test_<your>_adapter.py` and `tests/converters/test_<your>.py` for framework-specific behavior.
6. Run: `uv run pytest tests/ --framework <name> -v`.

---

## 5. Run all framework-related tests (all frameworks)

```bash
uv run pytest tests/framework_conformance/ tests/adapters/ tests/converters/ -v --ignore=tests/integration/
```

This excludes integration tests and shared helpers like `test_a2a_gateway.py`, `test_tool_parsing.py`; include them if you care about those contracts.

The authoritative list of all conformance tests is in the tables in section "Mandatory tests to add a new framework" above.
