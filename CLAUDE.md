# Thenvoi Python SDK

This is a Python SDK that connects AI agents to the Thenvoi collaborative platform.

## Core Features

1. Multi-framework support (LangGraph, Anthropic, CrewAI)
2. Platform tools: send_message, add_participant, remove_participant, get_participants, lookup_peers
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

- `THENVOI_REST_URL`: REST API URL (default: https://api.thenvoi.com)
- `THENVOI_WS_URL`: WebSocket URL (default: wss://api.thenvoi.com/ws)
- `OPENAI_API_KEY`: OpenAI API key (for LangGraph examples)
- `ANTHROPIC_API_KEY`: Anthropic API key (for Anthropic/Claude SDK examples)

## Example Files (examples/ directory)

- Use `load_agent_config("agent_name")` for credentials, NOT direct `os.environ.get()`
- Always load and validate `THENVOI_WS_URL` and `THENVOI_REST_URL` with `ValueError`
- Use `raise ValueError(...)` for missing required config, NOT `logger.error()` + `sys.exit()`
- Use single sys.path line: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))`
- Never hardcode UUIDs in docstrings - reference `agent_config.yaml` instead
- All `async def main()` functions must have `-> None` return type hint
- Always include `from __future__ import annotations` as first import
