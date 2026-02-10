"""Pytest fixtures for thenvoi SDK tests.

Most fixtures are provided by thenvoi-testing-python (auto-registered plugin).

Available from thenvoi_testing:
- factory: MockDataFactory for creating test data
- mock_agent_api, mock_human_api, mock_api_client: API client mocks
- mock_websocket: WebSocket client mock
- fake_agent_tools: FakeAgentTools for adapter testing
- sample_room_message, sample_agent_message: Message payloads

This file contains SDK-specific fixtures and event helpers that must
return SDK-native types for pattern matching compatibility.

Run only one framework's tests:
  uv run pytest tests/ --framework anthropic -v
  uv run pytest tests/ --framework langgraph -v
"""

from __future__ import annotations

import functools
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from thenvoi.client.streaming import (
    MessageCreatedPayload,
    MessageMetadata,
    RoomAddedPayload,
    RoomRemovedPayload,
    RoomOwner,
    ParticipantAddedPayload,
    ParticipantRemovedPayload,
)
from thenvoi.platform.event import (
    MessageEvent,
    RoomAddedEvent,
    RoomRemovedEvent,
    ParticipantAddedEvent,
    ParticipantRemovedEvent,
)
from thenvoi.runtime.types import PlatformMessage

# Imported after SDK imports to satisfy E402 (module-level import not at top).
# thenvoi_testing is an auto-registered pytest plugin; its markers module must
# be imported after the SDK types above so that both packages are on sys.path.
from thenvoi_testing.markers import pytest_ignore_collect_in_ci as _ignore_collect_in_ci


# Framework name -> (adapter_id, converter_id, adapter_file, converter_file)
# Use this name with: pytest tests/ --framework <name>
# Derived from the config registries; add new frameworks there first.
# Built lazily (via @functools.cache) so that running unrelated tests
# (e.g. tests/runtime/) does not force-import all framework configs and
# their transitive dependencies.
#
# CREWAI MOCK ISOLATION — READ BEFORE CHANGING TEST COLLECTION ORDER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The CrewAI conformance config (tests/framework_configs/adapters.py)
# imports CrewAIAdapter via _get_crewai_adapter_cls(), which injects
# MagicMock crewai/nest_asyncio modules at import time.  The returned
# class is @functools.cache'd and retains references to those mocks in
# its module globals.  It is ONLY safe for primitive attribute inspection.
#
# The framework-specific CrewAI tests (tests/adapters/test_crewai_adapter.py)
# use independent monkeypatch-based mocking and import their own class.
#
# These two paths are isolated because _get_crewai_adapter_cls() removes
# the adapter module from sys.modules after import and verifies no mock
# modules leaked.  However, if you reorder collection so that the
# framework-specific tests import the adapter module BEFORE the
# conformance factory runs, the @functools.cache'd class may see a
# different (real or differently-mocked) module.  If you change test
# collection order, always verify both paths:
#   uv run pytest tests/ --framework crewai -v
# See _get_crewai_adapter_cls() docstring in
# tests/framework_configs/adapters.py for the full explanation.


@functools.cache
def _get_framework_run_map() -> dict[str, tuple[str, str, str, str]]:
    from tests.framework_configs import CONVERTER_ID_FOR_ADAPTER
    from tests.framework_configs.adapters import ADAPTER_CONFIGS
    from tests.framework_configs.converters import CONVERTER_CONFIGS

    converter_by_id = {c.framework_id: c for c in CONVERTER_CONFIGS}
    run_map: dict[str, tuple[str, str, str, str]] = {}

    for ac in ADAPTER_CONFIGS:
        fid = ac.framework_id
        cid = CONVERTER_ID_FOR_ADAPTER.get(fid, fid)
        cc = converter_by_id[cid]
        adapter_file = f"test_{fid}_adapter.py"
        converter_file = f"test_{cc.framework_id}.py"
        run_map[fid] = (fid, cid, adapter_file, converter_file)

    return run_map


def pytest_configure(config):
    """Register warning filters for known mock + async introspection quirks."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore:coroutine 'GracefulShutdown._shutdown' was never awaited:RuntimeWarning",
    )


def pytest_addoption(parser):
    """Add --framework option to run only one framework's tests."""
    parser.addoption(
        "--framework",
        action="store",
        default=None,
        help="Run ONLY that framework's conformance + framework-specific adapter and converter tests. "
        "All other tests (core, runtime, shared utilities, etc.) are deselected. "
        "Valid: anthropic, langgraph, crewai, claude_sdk, pydantic_ai, parlant.",
    )


def pytest_ignore_collect(collection_path):
    """Skip integration tests in CI environment."""
    return _ignore_collect_in_ci(str(collection_path), "integration")


def pytest_collection_modifyitems(config, items):
    """When --framework is set, keep only that framework's conformance + framework-specific tests.

    NOTE: This deselects ALL non-matching tests (core, runtime, shared utilities,
    other frameworks, etc.).  Use ``--framework`` only for framework verification,
    not as a substitute for the full test suite.
    """
    framework_name = config.getoption("--framework", default=None)
    if not framework_name:
        return
    framework_name = framework_name.strip().lower()
    run_map = _get_framework_run_map()
    if framework_name not in run_map:
        raise pytest.UsageError(
            f"Unknown --framework={framework_name!r}. "
            f"Valid: {', '.join(sorted(run_map))}"
        )
    adapter_id, converter_id, adapter_file, converter_file = run_map[framework_name]

    def keep(item):
        nodeid = item.nodeid
        # Extract file path (before "::") so we match exact filenames, not
        # substrings (e.g. test_adapter_conformance_extra.py would not match).
        file_part = nodeid.split("::", 1)[0].replace("\\", "/")
        # Conformance: parametrized ids always appear as the final [...] suffix.
        if file_part.endswith("framework_conformance/test_adapter_conformance.py"):
            return nodeid.endswith(f"[{adapter_id}]")
        if file_part.endswith("framework_conformance/test_converter_conformance.py"):
            return nodeid.endswith(f"[{converter_id}]")
        # Framework-specific adapter/converter test files
        if file_part.endswith(f"adapters/{adapter_file}"):
            return True
        if file_part.endswith(f"converters/{converter_file}"):
            return True
        return False

    selected = []
    deselected = []
    for item in items:
        if keep(item):
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


# =============================================================================
# Event Factory Helpers (must return SDK-native types for pattern matching)
# =============================================================================


def make_message_event(
    room_id: str = "room-123",
    msg_id: str = "msg-123",
    content: str = "Test message",
    sender_id: str = "user-456",
    sender_type: str = "User",
    **kwargs,
) -> MessageEvent:
    """Create a MessageEvent using SDK-native types."""
    payload = MessageCreatedPayload(
        id=msg_id,
        content=content,
        message_type=kwargs.get("message_type", "text"),
        sender_id=sender_id,
        sender_type=sender_type,
        chat_room_id=room_id,
        inserted_at=kwargs.get("inserted_at", "2024-01-01T00:00:00Z"),
        updated_at=kwargs.get("updated_at", "2024-01-01T00:00:00Z"),
        metadata=kwargs.get("metadata", MessageMetadata(mentions=[], status="sent")),
    )
    return MessageEvent(room_id=room_id, payload=payload)


def make_room_added_event(
    room_id: str = "room-123", title: str = "Test Room", **kwargs
) -> RoomAddedEvent:
    """Create a RoomAddedEvent using SDK-native types."""
    payload = RoomAddedPayload(
        id=room_id,
        title=title,
        owner=kwargs.get(
            "owner", RoomOwner(id="user-1", name="Test User", type="User")
        ),
        status=kwargs.get("status", "active"),
        type=kwargs.get("type", "direct"),
        created_at=kwargs.get("created_at", "2024-01-01T00:00:00Z"),
        participant_role=kwargs.get("participant_role", "member"),
    )
    return RoomAddedEvent(room_id=room_id, payload=payload)


def make_room_removed_event(
    room_id: str = "room-123", title: str = "Test Room", **kwargs
) -> RoomRemovedEvent:
    """Create a RoomRemovedEvent using SDK-native types."""
    payload = RoomRemovedPayload(
        id=room_id,
        status=kwargs.get("status", "removed"),
        type=kwargs.get("type", "direct"),
        title=title,
        removed_at=kwargs.get("removed_at", "2024-01-01T00:00:00Z"),
    )
    return RoomRemovedEvent(room_id=room_id, payload=payload)


def make_participant_added_event(
    room_id: str = "room-123",
    participant_id: str = "user-456",
    name: str = "Test User",
    type: str = "User",
) -> ParticipantAddedEvent:
    """Create a ParticipantAddedEvent using SDK-native types."""
    payload = ParticipantAddedPayload(id=participant_id, name=name, type=type)
    return ParticipantAddedEvent(room_id=room_id, payload=payload)


def make_participant_removed_event(
    room_id: str = "room-123",
    participant_id: str = "user-456",
) -> ParticipantRemovedEvent:
    """Create a ParticipantRemovedEvent using SDK-native types."""
    payload = ParticipantRemovedPayload(id=participant_id)
    return ParticipantRemovedEvent(room_id=room_id, payload=payload)


# =============================================================================
# SDK-Specific Fixtures
# =============================================================================


@pytest.fixture
def dummy_message_handler():
    """Dummy message handler for tests that don't need handler logic."""

    async def handler(msg: MessageCreatedPayload) -> None:
        pass

    return handler


@pytest.fixture
def mock_thenvoi_agent(mock_api_client, mock_websocket):
    """Mock ThenvoiAgent coordinator for session/adapter tests."""
    agent = AsyncMock()
    agent.agent_id = "agent-123"
    agent.agent_name = "TestBot"
    agent._api_client = mock_api_client
    agent._ws_client = mock_websocket
    agent.active_sessions = {}

    agent._send_message_internal = AsyncMock(
        return_value={"id": "msg-123", "status": "sent"}
    )
    agent._send_event_internal = AsyncMock(
        return_value={"id": "evt-123", "status": "sent"}
    )
    agent._add_participant_internal = AsyncMock(
        return_value={"id": "user-456", "name": "Test User", "role": "member"}
    )
    agent._remove_participant_internal = AsyncMock(
        return_value={"id": "user-456", "name": "Test User", "status": "removed"}
    )
    agent._lookup_peers_internal = AsyncMock(
        return_value={
            "peers": [{"id": "peer-1", "name": "Peer One", "type": "Agent"}],
            "metadata": {
                "page": 1,
                "page_size": 50,
                "total_count": 1,
                "total_pages": 1,
            },
        }
    )
    agent._get_participants_internal = AsyncMock(
        return_value=[{"id": "agent-123", "name": "TestBot", "type": "Agent"}]
    )
    agent._create_chatroom_internal = AsyncMock(return_value="new-room-123")
    agent.get_context = AsyncMock()

    return agent


@pytest.fixture
def mock_agent_session():
    """Mock AgentSession for isolated tests."""
    session = AsyncMock()
    session.room_id = "room-123"
    session.is_llm_initialized = False
    session.participants = []
    session._last_participants_hash = None
    return session


@pytest.fixture
def sample_platform_message():
    """PlatformMessage fixture for new architecture."""
    return PlatformMessage(
        id="msg-123",
        room_id="room-123",
        content="@TestBot hello",
        sender_id="user-456",
        sender_type="User",
        sender_name="Test User",
        message_type="text",
        metadata={"mentions": [{"id": "agent-123", "name": "TestBot"}]},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_agent_platform_message():
    """PlatformMessage from the agent itself (for filtering tests)."""
    return PlatformMessage(
        id="msg-456",
        room_id="room-123",
        content="Hello there!",
        sender_id="agent-123",
        sender_type="Agent",
        sender_name="TestBot",
        message_type="text",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )
