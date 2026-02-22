"""E2E test configuration and fixtures.

E2E tests run adapters against a real Thenvoi platform with real (cheap) LLMs.
They verify platform functionality and integration correctness, not LLM output quality.

Run manually only, never in CI/CD:
    E2E_TESTS_ENABLED=true uv run pytest tests/e2e/ -v -s --no-cov

Configuration is loaded from .env.test with E2E-specific overrides from env vars.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from thenvoi_rest import AsyncRestClient, ChatRoomRequest
from thenvoi_rest.types import ParticipantRequest
from thenvoi_testing.settings import ThenvoiTestSettings

from thenvoi.client.streaming import WebSocketClient

logger = logging.getLogger(__name__)


# =============================================================================
# E2E Settings
# =============================================================================


class E2ESettings(ThenvoiTestSettings):
    """Settings for E2E tests, extending the standard test settings.

    Loads from .env.test and allows E2E-specific overrides via env vars.
    Pydantic BaseSettings automatically maps environment variables to fields
    (e.g. E2E_LLM_MODEL -> e2e_llm_model) with case-insensitive matching.
    """

    _env_file_path = Path(__file__).parent.parent.parent / ".env.test"

    # E2E-specific settings (override via environment variables)
    e2e_llm_model: str = "gpt-4o-mini"
    e2e_anthropic_model: str = "claude-3-haiku-20240307"
    e2e_timeout: int = 30
    e2e_tests_enabled: bool = False


# Singleton settings instance
e2e_settings = E2ESettings()


# =============================================================================
# Skip Markers
# =============================================================================

requires_e2e = pytest.mark.skipif(
    not e2e_settings.e2e_tests_enabled or not e2e_settings.thenvoi_api_key,
    reason="E2E_TESTS_ENABLED=false or THENVOI_API_KEY not set",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def e2e_config() -> E2ESettings:
    """Provide E2E settings to tests."""
    return e2e_settings


@pytest.fixture
def api_client(e2e_config: E2ESettings) -> AsyncRestClient:
    """Create a REST API client for the primary test agent."""
    if not e2e_config.thenvoi_api_key:
        pytest.skip("THENVOI_API_KEY not set")

    return AsyncRestClient(
        api_key=e2e_config.thenvoi_api_key,
        base_url=e2e_config.thenvoi_base_url,
    )


@pytest.fixture
async def e2e_chat_room(
    api_client: AsyncRestClient,
) -> AsyncGenerator[str, None]:
    """Create a unique chat room per test, yield room_id.

    Cleans up by leaving the room (rooms can't be deleted via agent API).
    """
    response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
    chat_id = response.data.id
    logger.info("Created E2E test chat room: %s", chat_id)

    yield chat_id

    # Cleanup: rooms can't be deleted via agent API, so just log
    logger.info("E2E test chat room %s will persist (no delete API)", chat_id)


@pytest.fixture
async def e2e_chat_room_with_user(
    api_client: AsyncRestClient,
) -> AsyncGenerator[tuple[str, str, str], None]:
    """Create a chat room with a User peer added.

    Yields (chat_id, user_id, user_name) tuple.
    The user peer is needed so agents can send @mentioned messages.
    """
    # Create chat room
    response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
    chat_id = response.data.id

    # Find a User peer to add
    peers_response = await api_client.agent_api.list_agent_peers()
    user_peer = next((p for p in peers_response.data if p.type == "User"), None)
    if user_peer is None:
        pytest.skip("No User peer available for E2E tests")

    # Add user to the room
    await api_client.agent_api.add_agent_chat_participant(
        chat_id,
        participant=ParticipantRequest(participant_id=user_peer.id, role="member"),
    )

    logger.info(
        "Created E2E chat room %s with user %s (%s)",
        chat_id,
        user_peer.name,
        user_peer.id,
    )

    yield chat_id, user_peer.id, user_peer.name

    logger.info("E2E test chat room %s will persist (no delete API)", chat_id)


@pytest.fixture
async def ws_client(e2e_config: E2ESettings) -> AsyncGenerator[WebSocketClient, None]:
    """WebSocket client for observing agent responses.

    Note: This creates a second WebSocket connection alongside the Agent's own
    connection for the same agent_id. The platform broadcasts messages to all
    connections for an agent, so both the agent and this test observer receive
    messages. This is intentional for test observability.
    """
    if not e2e_config.thenvoi_api_key:
        pytest.skip("THENVOI_API_KEY not set")

    ws = WebSocketClient(
        ws_url=e2e_config.thenvoi_ws_url,
        api_key=e2e_config.thenvoi_api_key,
        agent_id=e2e_config.test_agent_id or None,
    )
    async with ws:
        yield ws
