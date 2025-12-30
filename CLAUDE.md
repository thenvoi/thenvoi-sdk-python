# Thenvoi Python SDK

This is a Python SDK that connects AI agents to the Thenvoi collaborative platform.

Core features:

1. Multi-framework support
2. Platform tools: send_message, add_participant, remove_participant, get_participants, lookup_peers
3. WebSocket + REST transport: Real-time messaging with REST API fallback

General Coding Instructions:

- Always use type hints for function parameters and return types.
- NEVER use print() statements. Always use logging instead.
- Use module-level logger: `logger = logging.getLogger(__name__)`
- Use `from __future__ import annotations` for forward references.
- Use absolute imports from `thenvoi`.
- Use Pydantic for data models and validation.
- Follow existing patterns in the codebase for new code.

Dependencies:

- Package manager: `uv`
- Install: `uv sync --extra dev`
- Add dependency: `uv add <package>`
- Add optional dependency: `uv add --optional langgraph <package>`

Code Quality:

- Run linting: `uv run ruff check .`
- Run formatting: `uv run ruff format .`
- Run type checking: `uv run pyrefly check`

Testing:

- Run unit tests: `uv run pytest tests/ --ignore=tests/integration/ -v`
- Run single test: `uv run pytest tests/ -k "test_name"`
- Run with coverage: `uv run pytest tests/ --ignore=tests/integration/ --cov=src/thenvoi`
- Run integration tests (requires API key): `uv run pytest tests/integration/ -v -s --no-cov`

Git Workflow:

- Default branch: `main`
- Branch prefixes: `feat/`, `fix/`, `refactor/`
- Run tests before committing

Environment Variables:

- THENVOI_REST_API_URL: REST API URL (default: https://api.thenvoi.com)
- THENVOI_WS_URL: WebSocket URL (default: wss://api.thenvoi.com/ws)
- OPENAI_API_KEY: OpenAI API key (for LangGraph examples)
- ANTHROPIC_API_KEY: Anthropic API key (for Anthropic/Claude SDK examples)
