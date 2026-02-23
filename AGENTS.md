<!-- This file mirrors CLAUDE.md and provides equivalent project context for OpenAI Codex and other agents. Keep in sync with CLAUDE.md when making changes. -->

# Thenvoi Python SDK

This is a Python SDK that connects AI agents to the Thenvoi collaborative platform.

## Core Features

1. Multi-framework support (LangGraph, Anthropic, CrewAI, Claude SDK, Pydantic AI, Parlant, Letta)
2. Platform tools for chat, contacts, and memory management
3. WebSocket + REST transport: Real-time messaging with REST API fallback

## Platform Tools

### Chat Tools
- `thenvoi_send_message`: Send message to chat room (requires mentions)
- `thenvoi_send_event`: Send non-message event (thought, error, task)
- `thenvoi_add_participant`: Add agent/user to room
- `thenvoi_remove_participant`: Remove participant from room
- `thenvoi_get_participants`: List room participants
- `thenvoi_lookup_peers`: Find available agents/users
- `thenvoi_create_chatroom`: Create new chat room

### Contact Tools
- `thenvoi_list_contacts`: List agent's contacts with pagination
- `thenvoi_add_contact`: Send contact request to add someone
- `thenvoi_remove_contact`: Remove existing contact
- `thenvoi_list_contact_requests`: List received and sent requests
- `thenvoi_respond_contact_request`: Approve, reject, or cancel requests

### Memory Tools
- `thenvoi_list_memories`: List memories with filters (scope, system, type)
- `thenvoi_store_memory`: Store new memory with content, system, type, segment
- `thenvoi_get_memory`: Retrieve a specific memory by ID
- `thenvoi_supersede_memory`: Mark memory as superseded (soft delete)
- `thenvoi_archive_memory`: Archive memory (hide but preserve)

## REST Client API Pattern

The SDK uses Fern-generated REST client with property-based namespace API:

```python
# Pattern: agent_api_<resource>.method()
await link.rest.agent_api_chats.create_agent_chat(...)
await link.rest.agent_api_messages.create_agent_chat_message(...)
await link.rest.agent_api_participants.list_agent_chat_participants(...)
```

**Sub-clients**: `identity`, `peers`, `contacts`, `chats`, `messages`, `events`, `participants`, `context`, `memories`, `profile`, `agents`

## WebSocket Channels & Events

### Channels (Phoenix Channels Protocol V2)

| Channel | Topic Format | Events |
|---------|--------------|--------|
| Agent Rooms | `agent_rooms:{agent_id}` | `room_added`, `room_removed` |
| Chat Room | `chat_room:{chat_room_id}` | `message_created` |
| User Rooms | `user_rooms:{user_id}` | `room_added`, `room_removed` |
| Room Participants | `room_participants:{chat_room_id}` | `participant_added`, `participant_removed` |
| Tasks | `tasks:{user_id}` | `task_created`, `task_updated` |

### Payload Models (Pydantic)

```python
MessageCreatedPayload:
  id, content, message_type, sender_type, sender_id, sender_name?,
  metadata?, inserted_at, updated_at

RoomAddedPayload:
  id, title, task_id?, inserted_at, updated_at

RoomRemovedPayload:
  id, status, type, title, removed_at

ParticipantAddedPayload:
  id, name, type

ParticipantRemovedPayload:
  id

Mention:
  id, handle, name
```

### PlatformEvent Union (Tagged Union Pattern)

```python
PlatformEvent = (
    MessageEvent | RoomAddedEvent | RoomRemovedEvent
    | ParticipantAddedEvent | ParticipantRemovedEvent
)
```

Each event has: `type` (literal), `room_id`, `payload`, `raw`

### Contact Events (via `agent_contacts:{agent_id}` channel)

| Event | Payload Fields |
|-------|----------------|
| `contact_request_received` | `id`, `from_handle`, `from_name`, `message?`, `status`, `inserted_at` |
| `contact_request_updated` | `id`, `status` |
| `contact_added` | `id`, `handle`, `name`, `type` |
| `contact_removed` | `id` |

## Contact Event Handling

The SDK supports three strategies for handling contact WebSocket events via `ContactEventConfig`:

### Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `DISABLED` (default) | Ignores contact events | Agents that don't manage contacts |
| `CALLBACK` | Calls programmatic callback | Auto-approve bots, custom logic |
| `HUB_ROOM` | Routes to dedicated chat room | LLM-based contact management |

### Configuration

```python
from thenvoi.runtime.types import ContactEventConfig, ContactEventStrategy

# CALLBACK strategy - programmatic handling
async def auto_approve(event, tools):
    if isinstance(event, ContactRequestReceivedEvent):
        await tools.respond_contact_request("approve", request_id=event.payload.id)

agent = Agent.create(
    adapter=adapter,
    contact_config=ContactEventConfig(
        strategy=ContactEventStrategy.CALLBACK,
        on_event=auto_approve,
    ),
)

# HUB_ROOM strategy - LLM handles contacts in dedicated room
agent = Agent.create(
    adapter=adapter,
    contact_config=ContactEventConfig(
        strategy=ContactEventStrategy.HUB_ROOM,
        hub_task_id="optional-task-id",  # Links hub room to a task
    ),
)

# Broadcast contact changes to all rooms (composable with any strategy)
agent = Agent.create(
    adapter=adapter,
    contact_config=ContactEventConfig(
        strategy=ContactEventStrategy.DISABLED,
        broadcast_changes=True,  # Inject "[Contacts]: X is now a contact" messages
    ),
)
```

### HUB_ROOM Details

- Creates dedicated chat room at agent startup
- Injects system prompt with contact management instructions
- Converts contact events to synthetic `MessageEvent` for LLM processing
- Posts task events to room for persistence/visibility
- Enriches `ContactRequestUpdatedEvent` with sender info via cache + API fallback

### REST Client OMIT vs Null

When calling REST endpoints with optional parameters, **never pass `None`** - the Fern client sends `null` which fails backend validation. Instead, use kwargs:

```python
# WRONG - sends {"action": "approve", "handle": null, "request_id": "..."}
await client.respond_to_agent_contact_request(action="approve", handle=None, request_id="...")

# CORRECT - sends {"action": "approve", "request_id": "..."}
kwargs = {"action": "approve", "request_id": "..."}
await client.respond_to_agent_contact_request(**kwargs)
```

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

## Coding Standards

- Always use type hints for function parameters and return types
- Use `from __future__ import annotations` as the first import in every file
- Never use `print()` — use `logging` with module-level `logger = logging.getLogger(__name__)`
- Use `%s` placeholders in log messages for lazy evaluation
- Use Pydantic v2 for data models; use `model_dump()` not `dict()`
- Target Python 3.10+; use `list[str]` not `List[str]`, `str | None` not `Optional[str]`
- Use async/await everywhere in async codebases; use `AsyncMock` for testing async methods
- Catch `pydantic.ValidationError` separately from generic `Exception`
- Use `raise ValueError(...)` for missing required config, not `logger.error()` + `sys.exit()`

## Pre-Commit Checklist

```bash
uv run ruff check .
uv run ruff format .
uv run pyrefly check
uv run pytest tests/ --ignore=tests/integration/ -v
```
