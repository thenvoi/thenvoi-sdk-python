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

- `THENVOI_REST_URL`: REST API URL (default: https://api.thenvoi.com)
- `THENVOI_WS_URL`: WebSocket URL (default: wss://api.thenvoi.com/ws)
- `OPENAI_API_KEY`: OpenAI API key (for LangGraph examples)
- `ANTHROPIC_API_KEY`: Anthropic API key (for Anthropic/Claude SDK examples)

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
