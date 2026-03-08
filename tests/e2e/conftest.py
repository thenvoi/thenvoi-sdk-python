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
from typing import TYPE_CHECKING

import pytest
from thenvoi_rest import AsyncRestClient
from thenvoi_testing.settings import ThenvoiTestSettings

from thenvoi.client.streaming import WebSocketClient

from tests.e2e.helpers import (
    TrackingWebSocketClient,
    create_room_with_user,
    created_room_ids,
)

if TYPE_CHECKING:
    from tests.e2e.adapters.conftest import AdapterFactory

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

    # Standard ThenvoiTestSettings convention for locating the env file.
    _env_file_path = Path(__file__).parent.parent.parent / ".env.test"

    # E2E-specific settings (override via environment variables)
    e2e_llm_model: str = "gpt-4o-mini"
    e2e_anthropic_model: str = "claude-3-haiku-20240307"
    e2e_timeout: int = 30
    e2e_tests_enabled: bool = False


# Singleton instance — initialised lazily on first access via _get_e2e_settings().
# Note: the first call happens at module import time (via _check_e2e_status below),
# but is wrapped in a try/except so a missing or malformed .env.test won't blow up
# test collection.
_e2e_settings: E2ESettings | None = None


def _get_e2e_settings() -> E2ESettings:
    global _e2e_settings  # noqa: PLW0603
    if _e2e_settings is None:
        _e2e_settings = E2ESettings()
    return _e2e_settings


# =============================================================================
# Skip Markers
# =============================================================================


def _check_e2e_status() -> tuple[bool, str]:
    """Check if E2E tests should be skipped.

    Evaluated once at module import time (when the ``requires_e2e`` marker
    is created). Returns ``(is_disabled, reason)`` so the skip message is
    actionable.
    """
    try:
        settings = _get_e2e_settings()
        if not settings.e2e_tests_enabled:
            return True, "E2E_TESTS_ENABLED is not set to true"
        if not settings.thenvoi_api_key:
            return True, "THENVOI_API_KEY is not set"
        return False, ""
    except Exception as exc:
        logger.warning(
            "E2E settings could not be loaded (missing .env.test?), skipping E2E tests",
            exc_info=True,
        )
        return True, f"E2E settings could not be loaded: {exc}"


_e2e_is_disabled, _e2e_skip_reason = _check_e2e_status()

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


@pytest.fixture(scope="session", autouse=True)
def e2e_room_summary() -> None:
    """Log a summary of rooms created during the E2E test session.

    Rooms persist on the platform (no delete API for agents), so this
    summary helps operators track accumulation across runs.
    """
    yield  # type: ignore[misc]
    if created_room_ids:
        logger.info(
            "E2E session created %d room(s) that will persist: %s",
            len(created_room_ids),
            ", ".join(created_room_ids),
        )


@pytest.fixture
async def api_client(
    e2e_config: E2ESettings,
) -> AsyncRestClient:
    """Create a REST API client for the primary test agent.

    Note: AsyncRestClient has no close() method — the underlying httpx
    client is managed internally and cleaned up on garbage collection.
    """
    if not e2e_config.thenvoi_api_key:
        pytest.skip("THENVOI_API_KEY not set")

    return AsyncRestClient(
        api_key=e2e_config.thenvoi_api_key,
        base_url=e2e_config.thenvoi_base_url,
    )


@pytest.fixture
async def e2e_chat_room_with_user(
    api_client: AsyncRestClient,
) -> tuple[str, str, str]:
    """Create a chat room with a User peer added.

    Returns (chat_id, user_id, user_name) tuple.
    The user peer is needed so agents can send @mentioned messages.

    Intentionally function-scoped: adapter tests are parametrized across
    multiple adapters, and each adapter sends different messages into the
    room. Sharing a room would leak context between adapters. This does
    mean one orphaned room per test — see note below.

    Note: rooms persist — there is no delete API for agents.
    """
    return await create_room_with_user(api_client)


@pytest.fixture(scope="session")
async def e2e_agent_id() -> str:
    """Get the agent ID for the test agent (cached for the entire session).

    Uses ``_get_e2e_settings()`` directly instead of the function-scoped
    ``e2e_config`` fixture, since session-scoped fixtures cannot depend on
    function-scoped ones.

    Note: Session-scoped because the agent ID is stable for a given API key
    and never changes mid-run. If the underlying agent is recreated between
    tests, this cached value would be stale — but that scenario doesn't
    apply to E2E runs against a persistent platform.
    """
    settings = _get_e2e_settings()
    if not settings.thenvoi_api_key:
        pytest.skip("THENVOI_API_KEY not set")

    # Short-lived client — AsyncRestClient has no close() method, so the
    # underlying httpx client is cleaned up on garbage collection when
    # this local variable falls out of scope after the fixture returns.
    client = AsyncRestClient(
        api_key=settings.thenvoi_api_key,
        base_url=settings.thenvoi_base_url,
    )
    agent_me = await client.agent_api_identity.get_agent_me()
    return agent_me.data.id


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


@pytest.fixture(
    params=[
        "langgraph",
        "anthropic",
        "pydantic_ai",
        "claude_sdk",
        "crewai",
    ]
)
def adapter_entry(
    request: pytest.FixtureRequest,
) -> tuple[str, AdapterFactory]:
    """Parametrized fixture yielding (name, factory) for each adapter.

    Defined here (e2e/conftest.py) so both adapters/ and scenarios/ tests
    share a single definition. The ADAPTER_FACTORIES import is deferred to
    avoid a circular dependency (adapters/conftest.py imports E2ESettings
    from this module). The ``AdapterFactory`` type is imported under
    ``TYPE_CHECKING`` for the same reason.
    """
    from tests.e2e.adapters.conftest import ADAPTER_FACTORIES

    name: str = request.param
    return name, ADAPTER_FACTORIES[name]
