"""Fixtures for integration tests against a real API server.

These tests require a running API server and valid credentials.
Credentials are loaded from .env.test file automatically.

Run integration tests:
    uv run pytest tests/integration/ -v

Skip integration tests (run only unit tests):
    uv run pytest tests/ --ignore=tests/integration/

To override .env.test values, set environment variables:
    THENVOI_API_KEY="your-key" uv run pytest tests/integration/ -v

Cleanup behavior:
    By default, tests clean up any agents/chats they create.
    Use --no-clean to skip cleanup and accumulate test data:

    uv run pytest tests/integration/ -v --no-clean

    Or set the environment variable:
    THENVOI_TEST_NO_CLEAN=1 uv run pytest tests/integration/ -v

    No-clean mode is useful for:
    - Testing user limits (max agents, max chats)
    - Finding edge case bugs with accumulated data
    - Debugging test failures by inspecting created resources
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from thenvoi_rest import AsyncRestClient, ChatMessageRequest, ChatRoomRequest
from thenvoi_testing.markers import skip_without_env, skip_without_envs
from thenvoi_testing.settings import ThenvoiTestSettings

if TYPE_CHECKING:
    from _pytest.config.argparsing import Parser

# =============================================================================
# Pytest Plugin Hooks
# =============================================================================


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
# API Client Fixtures
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
# Test Resource Fixtures
# =============================================================================


@pytest.fixture
async def test_chat(api_client: AsyncRestClient | None):
    """Create a temporary chat for testing and clean up after.

    Yields the chat ID for use in tests.
    Note: Cleanup may not be possible if delete is not supported.
    """
    if api_client is None:
        pytest.skip("THENVOI_API_KEY not set")

    from thenvoi_rest.types import (
        ChatMessageRequestMentionsItem as Mention,
        ParticipantRequest,
    )

    # Create a test chat
    response = await api_client.agent_api_chats.create_agent_chat(
        chat=ChatRoomRequest()
    )
    chat_id = response.data.id

    # Get a peer to add to the room so we can send a descriptive message
    peers_response = await api_client.agent_api_peers.list_agent_peers()
    if peers_response.data:
        peer = peers_response.data[0]
        await api_client.agent_api_participants.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=peer.id, role="member"),
        )

        # Add descriptive message (triggers auto-title)
        await api_client.agent_api_messages.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"Integration test fixture: @{peer.name} temporary chat for testing participant operations",
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )

    yield chat_id

    # Cleanup: We can't delete chats via agent API, so just leave it
    # The chat will remain but won't affect other tests


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
