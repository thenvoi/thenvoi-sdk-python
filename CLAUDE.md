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
- Use Context7 MCP to fetch up-to-date documentation

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
- PR titles MUST use conventional commits format: `feat:`, `fix:`, or `docs:` prefix
  - Example: `feat: Add custom tools support to all adapters`
  - Example: `fix: Handle validation errors in execute_tool_call`
  - Example: `docs: Update README with new adapter examples`
- Run tests before committing

Environment Variables:

- THENVOI_REST_URL: REST API URL (default: https://api.thenvoi.com)
- THENVOI_WS_URL: WebSocket URL (default: wss://api.thenvoi.com/ws)
- OPENAI_API_KEY: OpenAI API key (for LangGraph examples)
- ANTHROPIC_API_KEY: Anthropic API key (for Anthropic/Claude SDK examples)

Example Files (examples/ directory):

- Use `load_agent_config("agent_name")` for credentials, NOT direct os.environ.get()
- Always load and validate THENVOI_WS_URL and THENVOI_REST_URL with ValueError
- Use `raise ValueError(...)` for missing required config, NOT logger.error()+sys.exit()
- Use single sys.path line: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))`
- Never hardcode UUIDs in docstrings - reference agent_config.yaml instead
- All `async def main()` functions must have `-> None` return type hint
- Always include `from __future__ import annotations` as first import
