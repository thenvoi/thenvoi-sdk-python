"""Fixtures for integration tests against a real API server.

These tests require a running API server and valid credentials.
Credentials are loaded from .env.test file automatically.

Run integration tests:
    uv run pytest tests/integration/ -v

Skip integration tests (run only unit tests):
    uv run pytest tests/ --ignore=tests/integration/

To override .env.test values, set environment variables:
    THENVOI_API_KEY="your-key" uv run pytest tests/integration/ -v

Room reuse strategy:
    Tests share 2 session-scoped rooms to stay within the platform's
    10-room-per-agent limit.  Rooms persist across runs — the fixtures
    reuse existing rooms when possible.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from thenvoi_rest import AsyncRestClient, ChatRoomRequest
from thenvoi_rest.types import ParticipantRequest
from thenvoi_testing.markers import skip_without_env, skip_without_envs
from thenvoi_testing.settings import ThenvoiTestSettings

if TYPE_CHECKING:
    from _pytest.config.argparsing import Parser

# =============================================================================
# Pytest Plugin Hooks
# =============================================================================

logger = logging.getLogger(__name__)


def pytest_addoption(parser: Parser) -> None:
    """Add --no-clean option to pytest."""
    parser.addoption(
        "--no-clean",
        action="store_true",
        default=False,
        help="Skip cleanup of test-created agents and chats (useful for limit testing)",
    )


# =============================================================================
# No-Clean Mode Helper
# =============================================================================


def is_no_clean_mode(request: pytest.FixtureRequest | None = None) -> bool:
    """Check if no-clean mode is enabled.

    No-clean mode can be enabled via:
    - --no-clean pytest option
    - THENVOI_TEST_NO_CLEAN environment variable

    Usage in fixtures:
        @pytest.fixture
        async def my_agent(user_api_client, request):
            agent = await create_agent(...)
            yield agent
            if not is_no_clean_mode(request):
                await user_api_client.human_api_agents.delete_my_agent(id=agent.id, force=True)
    """
    # Check environment variable first
    if os.environ.get("THENVOI_TEST_NO_CLEAN", "").lower() in ("1", "true", "yes"):
        return True

    # Check pytest option if request is available
    if request is not None:
        try:
            return bool(request.config.getoption("--no-clean", default=False))
        except ValueError:
            # Option not registered (e.g., running without the plugin)
            pass

    return False


# =============================================================================
# Test Settings
# =============================================================================


class TestSettings(ThenvoiTestSettings):
    """Settings for integration tests, loaded from .env.test."""

    _env_file_path = Path(__file__).parent.parent / ".env.test"


# Load settings from .env.test
test_settings = TestSettings()


def get_api_key() -> str | None:
    return test_settings.thenvoi_api_key or None


def get_api_key_2() -> str | None:
    return test_settings.thenvoi_api_key_2 or None


def get_user_api_key() -> str | None:
    return test_settings.thenvoi_api_key_user or None


def get_base_url() -> str:
    return test_settings.thenvoi_base_url


def get_ws_url() -> str:
    return test_settings.thenvoi_ws_url


def get_test_agent_id() -> str | None:
    return test_settings.test_agent_id or None


def get_test_agent_id_2() -> str | None:
    return test_settings.test_agent_id_2 or None


# Skip markers using thenvoi_testing shared markers
requires_api = skip_without_env("THENVOI_API_KEY")

requires_multi_agent = skip_without_envs(
    ["THENVOI_API_KEY", "THENVOI_API_KEY_2"],
    reason="Both THENVOI_API_KEY and THENVOI_API_KEY_2 required for multi-agent tests",
)

requires_user_api = skip_without_env("THENVOI_API_KEY_USER")


# =============================================================================
# Agent / Peer Info Data Classes
# =============================================================================


@dataclass(frozen=True)
class AgentInfo:
    """Immutable snapshot of an agent's identity."""

    id: str
    name: str
    handle: str | None


@dataclass(frozen=True)
class PeerInfo:
    """Immutable snapshot of a peer."""

    id: str
    name: str
    type: str


# =============================================================================
# Session-Scoped Event Loop  (required for session-scoped async fixtures)
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Session-Scoped API Clients
# =============================================================================


@pytest.fixture(scope="session")
def session_api_client() -> AsyncRestClient | None:
    """Session-scoped REST client for the primary agent."""
    api_key = get_api_key()
    if not api_key:
        return None
    return AsyncRestClient(api_key=api_key, base_url=get_base_url())


@pytest.fixture(scope="session")
def session_api_client_2() -> AsyncRestClient | None:
    """Session-scoped REST client for the secondary agent."""
    api_key = get_api_key_2()
    if not api_key:
        return None
    return AsyncRestClient(api_key=api_key, base_url=get_base_url())


# =============================================================================
# Function-Scoped API Client Fixtures  (kept for backward-compat)
# =============================================================================


@pytest.fixture
def api_client() -> AsyncRestClient | None:
    """Create a real async API client for integration tests (primary agent).

    Returns None if THENVOI_API_KEY is not set.
    Uses function scope to avoid event loop issues with async tests.
    """
    api_key = get_api_key()
    if not api_key:
        return None

    return AsyncRestClient(
        api_key=api_key,
        base_url=get_base_url(),
    )


@pytest.fixture
def api_client_2() -> AsyncRestClient | None:
    """Create a real async API client for the secondary agent.

    Returns None if THENVOI_API_KEY_2 is not set.
    """
    api_key = get_api_key_2()
    if not api_key:
        return None

    return AsyncRestClient(
        api_key=api_key,
        base_url=get_base_url(),
    )


@pytest.fixture
def user_api_client() -> AsyncRestClient | None:
    """Create a real async API client with user API key.

    This client uses the User API (human_api) for operations like:
    - Registering new agents
    - Listing owned agents
    - Deleting agents

    Returns None if THENVOI_API_KEY_USER is not set.
    """
    api_key = get_user_api_key()
    if not api_key:
        return None

    return AsyncRestClient(
        api_key=api_key,
        base_url=get_base_url(),
    )


@pytest.fixture
def integration_settings() -> TestSettings:
    """Provide test settings to integration tests."""
    return test_settings


# =============================================================================
# Session-Scoped Identity / Peer Helpers
# =============================================================================


@pytest_asyncio.fixture(scope="session")
async def shared_agent1_info(
    session_api_client: AsyncRestClient | None,
) -> AgentInfo | None:
    """Session-scoped identity info for Agent 1."""
    if session_api_client is None:
        return None
    response = await session_api_client.agent_api_identity.get_agent_me()
    data = response.data
    return AgentInfo(id=data.id, name=data.name, handle=getattr(data, "handle", None))


@pytest_asyncio.fixture(scope="session")
async def shared_agent2_info(
    session_api_client_2: AsyncRestClient | None,
) -> AgentInfo | None:
    """Session-scoped identity info for Agent 2."""
    if session_api_client_2 is None:
        return None
    response = await session_api_client_2.agent_api_identity.get_agent_me()
    data = response.data
    return AgentInfo(id=data.id, name=data.name, handle=getattr(data, "handle", None))


@pytest_asyncio.fixture(scope="session")
async def shared_user_peer(
    session_api_client: AsyncRestClient | None,
) -> PeerInfo | None:
    """Session-scoped User peer (the agent owner)."""
    if session_api_client is None:
        return None
    response = await session_api_client.agent_api_peers.list_agent_peers()
    user_peer = next((p for p in (response.data or []) if p.type == "User"), None)
    if user_peer is None:
        return None
    return PeerInfo(id=user_peer.id, name=user_peer.name, type=user_peer.type)


# =============================================================================
# Session-Scoped Shared Rooms
# =============================================================================


async def _ensure_participant(
    api_client: AsyncRestClient,
    chat_id: str,
    participant_id: str,
    role: str = "member",
) -> None:
    """Add a participant to a room if not already present."""
    response = await api_client.agent_api_participants.list_agent_chat_participants(
        chat_id
    )
    existing_ids = {p.id for p in (response.data or [])}
    if participant_id not in existing_ids:
        await api_client.agent_api_participants.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=participant_id, role=role),
        )
        logger.info("Added participant %s to room %s", participant_id, chat_id)


@pytest_asyncio.fixture(scope="session")
async def shared_room(
    session_api_client: AsyncRestClient | None,
    shared_user_peer: PeerInfo | None,
) -> str | None:
    """Session-scoped chat room reused across tests and runs.

    Reuses an existing room if available, otherwise creates a new one.
    A User peer is ensured as participant so messages can be sent with mentions.
    No cleanup -- the room persists for future runs.
    """
    if session_api_client is None:
        return None

    # Try to reuse an existing room
    response = await session_api_client.agent_api_chats.list_agent_chats()
    existing_rooms = response.data or []
    if existing_rooms:
        chat_id = existing_rooms[0].id
        logger.info("Reusing existing room for shared_room: %s", chat_id)
    else:
        # Create a new room
        create_response = await session_api_client.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest()
        )
        chat_id = create_response.data.id
        logger.info("Created new shared_room: %s", chat_id)

    # Ensure User peer is a participant
    if shared_user_peer is not None:
        await _ensure_participant(session_api_client, chat_id, shared_user_peer.id)

    return chat_id


@pytest_asyncio.fixture(scope="session")
async def shared_multi_agent_room(
    session_api_client: AsyncRestClient | None,
    session_api_client_2: AsyncRestClient | None,
    shared_agent2_info: AgentInfo | None,
    shared_user_peer: PeerInfo | None,
) -> str | None:
    """Session-scoped chat room with both agents, reused across tests and runs.

    Finds an existing room that has Agent 2 as participant, or creates a new one.
    Ensures Agent 2 and User peer are participants.
    No cleanup -- the room persists for future runs.
    """
    if session_api_client is None or session_api_client_2 is None:
        return None
    if shared_agent2_info is None:
        return None

    agent2_id = shared_agent2_info.id

    # Look for an existing room that already has Agent 2
    response = await session_api_client.agent_api_chats.list_agent_chats()
    existing_rooms = response.data or []

    chat_id: str | None = None
    for room in existing_rooms:
        participants_response = await session_api_client.agent_api_participants.list_agent_chat_participants(
            room.id
        )
        participant_ids = {p.id for p in (participants_response.data or [])}
        if agent2_id in participant_ids:
            chat_id = room.id
            logger.info("Reusing existing multi-agent room: %s", chat_id)
            break

    if chat_id is None:
        # Create a new room and add Agent 2
        create_response = await session_api_client.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest()
        )
        chat_id = create_response.data.id
        logger.info("Created new shared_multi_agent_room: %s", chat_id)
        await _ensure_participant(session_api_client, chat_id, agent2_id)

    # Ensure User peer is present
    if shared_user_peer is not None:
        await _ensure_participant(session_api_client, chat_id, shared_user_peer.id)

    return chat_id


# =============================================================================
# Function-Scoped Test Resource Fixtures
# =============================================================================


@pytest.fixture
async def test_chat(shared_room: str | None) -> str | None:
    """Provide the shared room as the test chat.

    This replaces the old fixture that created a new room per test.
    """
    if shared_room is None:
        pytest.skip("THENVOI_API_KEY not set")

    return shared_room


@pytest.fixture
async def test_peer_id(api_client: AsyncRestClient | None) -> str | None:
    """Get a peer ID that can be used for testing participant operations.

    Returns the first available peer that is NOT the agent's owner.
    This avoids P4 protection rule (agent cannot remove its own owner).
    """
    if api_client is None:
        pytest.skip("THENVOI_API_KEY not set")

    # Get agent's owner_uuid to exclude from peer selection
    agent_me = await api_client.agent_api_identity.get_agent_me()
    agent_owner_uuid = (
        str(agent_me.data.owner_uuid) if agent_me.data.owner_uuid else None
    )

    response = await api_client.agent_api_peers.list_agent_peers()
    if response.data:
        # Prefer a peer that is NOT the agent's owner (to avoid P4 protection rule)
        for peer in response.data:
            if peer.id != agent_owner_uuid:
                return peer.id
        # Fallback to first peer if all peers are the owner (unlikely)
        return response.data[0].id
    return None
