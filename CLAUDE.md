# Thenvoi Python SDK

This is a Python SDK that connects AI agents to the Thenvoi collaborative platform.

## Core Features

1. Multi-framework support (LangGraph, Anthropic, CrewAI)
2. Platform tools: thenvoi_send_message, thenvoi_send_event, thenvoi_add_participant, thenvoi_remove_participant, thenvoi_get_participants, thenvoi_lookup_peers, thenvoi_create_chatroom
3. WebSocket + REST transport: Real-time messaging with REST API fallback

## Code Structure

```
src/thenvoi/
├── adapters/       # Framework adapters (langgraph, anthropic, crewai)
├── converters/     # History converters per framework
├── core/           # Protocols, types, base classes
├── runtime/        # Execution context, tools, formatters
├── platform/       # WebSocket/REST transport, events
├── preprocessing/  # Event filtering before adapter
├── client/         # Low-level API clients
└── agent.py        # Main entry point
```

## Testing Structure

```
tests/
├── adapters/       # Unit tests per adapter (mocked)
├── converters/     # Unit tests per converter
├── core/           # Core logic tests
├── runtime/        # Runtime tests
├── integration/    # Real API tests (skipped in CI)
└── conftest.py     # Shared fixtures
```

## Commands

```bash
# Install dependencies
uv sync --extra dev

# Run unit tests
uv run pytest tests/ --ignore=tests/integration/ -v

# Run single test
uv run pytest tests/ -k "test_name"

# Run with coverage
uv run pytest tests/ --ignore=tests/integration/ --cov=src/thenvoi

# Run integration tests (requires API key)
uv run pytest tests/integration/ -v -s --no-cov

# Linting and formatting
uv run ruff check .
uv run ruff format .
uv run pyrefly check
```

## Environment Variables

- `THENVOI_REST_URL`: REST API URL (default: https://app.thenvoi.com)
- `THENVOI_WS_URL`: WebSocket URL (default: wss://app.thenvoi.com/api/v1/socket/websocket)
- `OPENAI_API_KEY`: OpenAI API key (for LangGraph examples)
- `ANTHROPIC_API_KEY`: Anthropic API key (for Anthropic/Claude SDK examples)

## Adding a New Framework Integration

When adding a new framework adapter and converter, follow this TDD workflow. Use the lowercase module name (e.g. `openai`, `gemini`) and derive the PascalCase class prefix (e.g. `OpenAI`, `Gemini`).

### Phase 1: Scaffold Source Files

1. Create converter at `src/thenvoi/converters/<framework>.py` — class `{Framework}HistoryConverter` with stub `convert()`, `set_agent_name()`, `__init__(*, agent_name=None)`. Use `from thenvoi.converters._tool_parsing import parse_tool_call, parse_tool_result`.
2. Create adapter at `src/thenvoi/adapters/<framework>.py` — class `{Framework}Adapter` extending `SimpleAdapter[T]` with `__init__` params: `model`, `custom_section`, `enable_execution_reporting`, `history_converter`. Stub `on_message`, `on_started`, `on_cleanup`.
3. If the framework needs an external SDK, add an optional dependency group in `pyproject.toml`.

### Phase 2: Register with Conformance Infrastructure

1. Add an output adapter in `tests/framework_configs/output_adapters.py` — choose base class matching output format (`BaseDictListOutputAdapter`, `StringOutputAdapter`, `SenderDictListAdapter`, or `LangChainOutputAdapter`).
2. Register converter config in `tests/framework_configs/converters.py` — factory function, builder function returning `ConverterConfig` with behavioral flags, append to `_CONVERTER_CONFIG_BUILDERS`.
3. Register adapter config in `tests/framework_configs/adapters.py` — factory function with mocked constructor args, builder function returning `AdapterConfig`, append to `_ADAPTER_CONFIG_BUILDERS`.

### Phase 3: Run Conformance Tests (Expect Failures)

```bash
uv run pytest tests/framework_conformance/test_config_drift.py -v
uv run pytest tests/framework_conformance/test_adapter_conformance.py -v -k "<framework>"
uv run pytest tests/framework_conformance/test_converter_conformance.py -v -k "<framework>"
```

### Phase 4: Implement the Converter

In `src/thenvoi/converters/<framework>.py`, implement `convert()`: text messages as `[sender_name]: content`, own agent filtering, other agent remapping, tool events via `parse_tool_call`/`parse_tool_result`, skip thought messages, default role to `"user"`.

### Phase 5: Implement the Adapter

In `src/thenvoi/adapters/<framework>.py`: `on_started` sets agent name/description and creates client, `on_message` converts history and invokes LLM, `on_cleanup` cleans per-room state safely.

### Phase 6: Write Framework-Specific Tests

- Adapter tests in `tests/adapters/test_<framework>_adapter.py` — LLM invocation, tool execution, error handling, custom tools.
- Converter tests in `tests/converters/test_<framework>.py` — tool event format, batching, malformed input.

### Phase 7: Final Validation

```bash
uv run pytest tests/framework_conformance/ tests/framework_configs/ -v
uv run pytest tests/adapters/test_<framework>_adapter.py tests/converters/test_<framework>.py -v
uv run pytest tests/ --ignore=tests/integration/ -v
uv run ruff check . && uv run ruff format .
```

### Key Files Reference

| Purpose | Path |
|---|---|
| Adapter source | `src/thenvoi/adapters/<framework>.py` |
| Converter source | `src/thenvoi/converters/<framework>.py` |
| Adapter config registry | `tests/framework_configs/adapters.py` |
| Converter config registry | `tests/framework_configs/converters.py` |
| Output adapters | `tests/framework_configs/output_adapters.py` |
| Adapter conformance tests | `tests/framework_conformance/test_adapter_conformance.py` |
| Converter conformance tests | `tests/framework_conformance/test_converter_conformance.py` |
| Config drift detection | `tests/framework_conformance/test_config_drift.py` |

## Example Files (examples/ directory)

### PEP 723 Script Metadata (Required for `uv run` support)

Every example file must include PEP 723 inline script metadata at the top for standalone execution with `uv run`:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[<extra>]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Brief description of what this example does.

Run with:
    uv run examples/<framework>/<example_file>.py
"""
```

Replace `<extra>` with the appropriate framework extra (e.g., `langgraph`, `anthropic`, `crewai`, `claude-sdk`, `pydantic-ai`, `parlant`).

### Other Requirements

- Use `load_agent_config("agent_name")` for credentials, NOT direct `os.environ.get()`
- Always load and validate `THENVOI_WS_URL` and `THENVOI_REST_URL` with `ValueError`
- Use `raise ValueError(...)` for missing required config, NOT `logger.error()` + `sys.exit()`
- Use single sys.path line: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))`
- Never hardcode UUIDs in docstrings - reference `agent_config.yaml` instead
- All `async def main()` functions must have `-> None` return type hint
- Always include `from __future__ import annotations` as first import
