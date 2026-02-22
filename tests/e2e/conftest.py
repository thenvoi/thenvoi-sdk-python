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
from thenvoi_rest import AsyncRestClient
from thenvoi_testing.settings import ThenvoiTestSettings

from thenvoi.client.streaming import WebSocketClient

from tests.e2e.helpers import TrackingWebSocketClient, create_room_with_user

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


# Lazy singleton — avoids blowing up test collection if .env.test is
# missing or malformed (only E2E tests need these settings).
_e2e_settings: E2ESettings | None = None


def _get_e2e_settings() -> E2ESettings:
    global _e2e_settings  # noqa: PLW0603
    if _e2e_settings is None:
        _e2e_settings = E2ESettings()
    return _e2e_settings


# =============================================================================
# Skip Markers
# =============================================================================


_e2e_skip_reason: str = ""


def _e2e_disabled() -> bool:
    """Check if E2E tests should be skipped.

    Evaluated once at module import time (when the ``requires_e2e`` marker
    is created). The result is then cached in the ``skipif`` condition.
    Sets ``_e2e_skip_reason`` so the skip message is actionable.
    """
    global _e2e_skip_reason  # noqa: PLW0603
    try:
        settings = _get_e2e_settings()
        if not settings.e2e_tests_enabled:
            _e2e_skip_reason = "E2E_TESTS_ENABLED is not set to true"
            return True
        if not settings.thenvoi_api_key:
            _e2e_skip_reason = "THENVOI_API_KEY is not set"
            return True
        return False
    except Exception as exc:
        _e2e_skip_reason = f"E2E settings could not be loaded: {exc}"
        logger.warning(
            "E2E settings could not be loaded (missing .env.test?), skipping E2E tests",
            exc_info=True,
        )
        return True


_e2e_is_disabled = _e2e_disabled()

requires_e2e = pytest.mark.skipif(
    _e2e_is_disabled,
    reason=_e2e_skip_reason or "E2E tests disabled",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def e2e_config() -> E2ESettings:
    """Provide E2E settings to tests."""
    return _get_e2e_settings()


@pytest.fixture
async def api_client(
    e2e_config: E2ESettings,
) -> AsyncGenerator[AsyncRestClient, None]:
    """Create a REST API client for the primary test agent."""
    if not e2e_config.thenvoi_api_key:
        pytest.skip("THENVOI_API_KEY not set")

    client = AsyncRestClient(
        api_key=e2e_config.thenvoi_api_key,
        base_url=e2e_config.thenvoi_base_url,
    )
    yield client
    await client.close()


@pytest.fixture
async def e2e_chat_room_with_user(
    api_client: AsyncRestClient,
) -> tuple[str, str, str]:
    """Create a chat room with a User peer added.

    Returns (chat_id, user_id, user_name) tuple.
    The user peer is needed so agents can send @mentioned messages.

    Note: rooms persist — there is no delete API for agents.
    """
    return await create_room_with_user(api_client)


@pytest.fixture(scope="module")
async def e2e_agent_id() -> str:
    """Get the agent ID for the test agent (cached per module).

    Uses ``_get_e2e_settings()`` directly instead of the function-scoped
    ``e2e_config`` fixture, since module-scoped fixtures cannot depend on
    function-scoped ones.
    """
    settings = _get_e2e_settings()
    if not settings.thenvoi_api_key:
        pytest.skip("THENVOI_API_KEY not set")

    client = AsyncRestClient(
        api_key=settings.thenvoi_api_key,
        base_url=settings.thenvoi_base_url,
    )
    try:
        agent_me = await client.agent_api.get_agent_me()
        return agent_me.data.id
    finally:
        await client.close()


@pytest.fixture
async def ws_client(
    e2e_config: E2ESettings,
) -> AsyncGenerator[TrackingWebSocketClient, None]:
    """WebSocket client for observing agent responses.

    Note: This creates a second WebSocket connection alongside the Agent's own
    connection for the same agent_id. The platform broadcasts messages to all
    connections for an agent, so both the agent and this test observer receive
    messages. This is intentional for test observability.

    Wraps the raw WebSocketClient in a TrackingWebSocketClient that tracks
    joined channels and explicitly leaves them on teardown.
    """
    if not e2e_config.thenvoi_api_key:
        pytest.skip("THENVOI_API_KEY not set")

    ws = WebSocketClient(
        ws_url=e2e_config.thenvoi_ws_url,
        api_key=e2e_config.thenvoi_api_key,
        agent_id=e2e_config.test_agent_id,
    )

    async with ws:
        tracking_ws = TrackingWebSocketClient(ws)
        yield tracking_ws
        await tracking_ws.cleanup_channels()
