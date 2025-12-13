"""Fixtures for integration tests against a real API server.

These tests require a running API server and valid credentials.
Credentials are loaded from .env.test file automatically.

Run integration tests:
    uv run pytest tests/integration/ -v

Skip integration tests (run only unit tests):
    uv run pytest tests/ --ignore=tests/integration/

To override .env.test values, set environment variables:
    THENVOI_API_KEY="your-key" uv run pytest tests/integration/ -v
"""

from pathlib import Path

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict
from thenvoi_rest import AsyncRestClient, ChatRoomRequest


class TestSettings(BaseSettings):
    """Settings for integration tests, loaded from .env.test."""

    # Primary agent (AR-2 Darter)
    thenvoi_api_key: str = ""
    test_agent_id: str = ""

    # Secondary agent (MQ 12) - for multi-agent tests
    thenvoi_api_key_2: str = ""
    test_agent_id_2: str = ""

    # Server URLs
    thenvoi_base_url: str = "http://localhost:4000"
    thenvoi_ws_url: str = "ws://localhost:4000/api/v1/socket/websocket"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env.test",
        case_sensitive=False,
        extra="ignore",
    )


# Load settings from .env.test
test_settings = TestSettings()


def get_api_key() -> str | None:
    return test_settings.thenvoi_api_key or None


def get_api_key_2() -> str | None:
    return test_settings.thenvoi_api_key_2 or None


def get_base_url() -> str:
    return test_settings.thenvoi_base_url


def get_ws_url() -> str:
    return test_settings.thenvoi_ws_url


def get_test_agent_id() -> str | None:
    return test_settings.test_agent_id or None


def get_test_agent_id_2() -> str | None:
    return test_settings.test_agent_id_2 or None


# Skip marker for integration tests
requires_api = pytest.mark.skipif(
    not get_api_key(), reason="THENVOI_API_KEY environment variable not set"
)

# Skip marker for multi-agent tests (requires both agents)
requires_multi_agent = pytest.mark.skipif(
    not get_api_key() or not get_api_key_2(),
    reason="Both THENVOI_API_KEY and THENVOI_API_KEY_2 required for multi-agent tests",
)


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
    """Create a real async API client for the secondary agent (MQ 12).

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
def integration_settings() -> TestSettings:
    """Provide test settings to integration tests."""
    return test_settings


@pytest.fixture
async def test_chat(api_client: AsyncRestClient | None):
    """Create a temporary chat for testing and clean up after.

    Yields the chat ID for use in tests.
    Note: Cleanup may not be possible if delete is not supported.
    """
    if api_client is None:
        pytest.skip("THENVOI_API_KEY not set")

    # Create a test chat
    response = await api_client.agent_api.create_agent_chat(
        chat=ChatRoomRequest(title="Integration Test Chat")
    )
    chat_id = response.data.id

    yield chat_id

    # Cleanup: We can't delete chats via agent API, so just leave it
    # The chat will remain but won't affect other tests


@pytest.fixture
async def test_peer_id(api_client: AsyncRestClient | None) -> str | None:
    """Get a peer ID that can be used for testing participant operations.

    Returns the first available peer (agent or user) that can be added to chats.
    """
    if api_client is None:
        pytest.skip("THENVOI_API_KEY not set")

    response = await api_client.agent_api.list_agent_peers()
    if response.data:
        return response.data[0].id
    return None
