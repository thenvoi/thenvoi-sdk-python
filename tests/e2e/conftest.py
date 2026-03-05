"""E2E test configuration and fixtures.

E2E tests run adapters against a real Thenvoi platform with real (cheap) LLMs.
They verify platform functionality and integration correctness, not LLM output quality.

Run manually only, never in CI/CD:
    E2E_TESTS_ENABLED=true uv run pytest tests/e2e/ -v -s --no-cov

Configuration is loaded from .env.test with E2E-specific overrides from env vars.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from dotenv import load_dotenv
from pydantic import ValidationError
from thenvoi_rest import AsyncRestClient
from thenvoi_testing.settings import ThenvoiTestSettings

# Load .env.test into os.environ so LLM libraries (langchain, anthropic, etc.)
# can pick up OPENAI_API_KEY, ANTHROPIC_API_KEY, and other keys.
_ENV_TEST_PATH = Path(__file__).parent.parent.parent / ".env.test"
load_dotenv(_ENV_TEST_PATH, override=False)

from thenvoi.client.streaming import WebSocketClient  # noqa: E402

from tests.e2e.helpers import (  # noqa: E402
    TrackingWebSocketClient,
    create_room_with_user,
)

if TYPE_CHECKING:
    from tests.e2e.adapters.conftest import AdapterFactory

logger = logging.getLogger(__name__)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Ensure all E2E async tests use the session-scoped event loop.

    Fixtures already default to the session loop via
    ``asyncio_default_fixture_loop_scope = "session"`` in pyproject.toml,
    but test functions default to function-scoped loops. This mismatch
    causes "Future attached to a different loop" errors when tests call
    into session-scoped WS/REST clients. Applying ``loop_scope="session"``
    to every E2E test aligns them with the fixture loop.
    """
    session_marker = pytest.mark.asyncio(loop_scope="session")
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(session_marker)


# Platform limits agents to 10 active chat rooms; cap room searches accordingly.
_MAX_ROOMS_TO_SEARCH = 10


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
        settings = E2ESettings()
        if not settings.e2e_tests_enabled:
            return True, "E2E_TESTS_ENABLED is not set to true"
        if not settings.thenvoi_api_key:
            return True, "THENVOI_API_KEY is not set"
        return False, "E2E tests enabled"
    except (ValidationError, ValueError, OSError) as exc:
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


@pytest.fixture(scope="session")
def e2e_config() -> E2ESettings:
    """Provide E2E settings to tests (session-scoped singleton)."""
    return E2ESettings()


@pytest.fixture(scope="session")
def e2e_created_room_ids() -> list[str]:
    """Session-scoped list tracking room IDs created during the E2E run.

    Passed to ``create_room_with_user()`` so the helper can record new rooms
    without relying on module-level mutable state.
    """
    return []


@pytest.fixture(scope="session", autouse=True)
def e2e_room_summary(e2e_created_room_ids: list[str]) -> Generator[None, None, None]:
    """Log a summary of rooms created during the E2E test session.

    Rooms persist on the platform (no delete API for agents), so this
    summary helps operators track accumulation across runs.
    """
    yield
    if e2e_created_room_ids:
        logger.info(
            "E2E session created %d room(s) that will persist: %s",
            len(e2e_created_room_ids),
            ", ".join(e2e_created_room_ids),
        )


@pytest.fixture(scope="session")
def e2e_session_client(
    e2e_config: E2ESettings,
) -> AsyncRestClient:
    """Session-scoped REST client shared across all E2E fixtures.

    Avoids creating multiple short-lived AsyncRestClient instances in each
    session-scoped fixture. AsyncRestClient has no close() method — the
    underlying httpx client is managed internally.
    """
    if not e2e_config.thenvoi_api_key:
        pytest.skip("THENVOI_API_KEY not set")

    return AsyncRestClient(
        api_key=e2e_config.thenvoi_api_key,
        base_url=e2e_config.thenvoi_base_url,
    )


@pytest.fixture
def api_client(
    e2e_session_client: AsyncRestClient,
) -> AsyncRestClient:
    """Function-scoped alias for the session REST client.

    Provides backward-compatible fixture name for tests that inject ``api_client``.
    """
    return e2e_session_client


async def _find_room_with_participant(
    client: AsyncRestClient,
    participant_id: str,
    exclude_ids: set[str] | None = None,
) -> str | None:
    """Find an existing room that contains the given participant.

    Searches at most ``_MAX_ROOMS_TO_SEARCH`` rooms to avoid excessive API
    calls for agents with many rooms.  Returns the room ID or ``None`` if
    no match.
    """
    chats_response = await client.agent_api_chats.list_agent_chats()
    existing_rooms = chats_response.data or []

    checked = 0
    for room in existing_rooms:
        if exclude_ids and room.id in exclude_ids:
            continue
        if checked >= _MAX_ROOMS_TO_SEARCH:
            break
        checked += 1
        participants_response = (
            await client.agent_api_participants.list_agent_chat_participants(room.id)
        )
        participant_ids = [p.id for p in (participants_response.data or [])]
        if participant_id in participant_ids:
            return room.id

    return None


@pytest.fixture(scope="session")
async def e2e_shared_room(
    e2e_session_client: AsyncRestClient,
    e2e_created_room_ids: list[str],
) -> tuple[str, str, str]:
    """Session-scoped shared chat room with a User peer.

    Returns (chat_id, user_id, user_name) tuple. Reuses an existing room
    from prior runs when possible to avoid room accumulation (the platform
    has no delete API for agents).

    Tests that check WS responses (smoke, tool execution) can safely share
    this room because each agent only processes messages arriving while it's
    connected. Context hydration may include prior messages, but assertions
    target the specific WS response, not room history.
    """
    client = e2e_session_client

    # Find a User peer
    peers_response = await client.agent_api_peers.list_agent_peers()
    user_peer = next((p for p in peers_response.data if p.type == "User"), None)
    if user_peer is None:
        pytest.skip("No User peer available for E2E tests")

    # Try to reuse an existing room that has this User peer
    room_id = await _find_room_with_participant(client, user_peer.id)
    if room_id is not None:
        logger.info(
            "E2E: Reusing existing room %s with user %s", room_id, user_peer.name
        )
        return room_id, user_peer.id, user_peer.name

    # No suitable room found — create one
    return await create_room_with_user(client, room_tracker=e2e_created_room_ids)


@pytest.fixture(scope="session")
async def e2e_isolation_room_pair(
    e2e_session_client: AsyncRestClient,
    e2e_shared_room: tuple[str, str, str],
    e2e_created_room_ids: list[str],
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Session-scoped pair of rooms for isolation tests.

    Returns ((room_a_id, user_id, user_name), (room_b_id, user_id, user_name)).
    Room A reuses e2e_shared_room. Room B reuses a second existing room or
    creates one (once per session). This limits room accumulation to at most
    1 new room instead of 2 per adapter × 5 adapters = 10.
    """
    room_a_id, user_id, user_name = e2e_shared_room
    client = e2e_session_client

    # Try to find a second existing room (different from room A) with User peer
    room_b_id = await _find_room_with_participant(
        client, user_id, exclude_ids={room_a_id}
    )
    if room_b_id is not None:
        logger.info("E2E: Reusing existing room %s as isolation room B", room_b_id)
        return (room_a_id, user_id, user_name), (room_b_id, user_id, user_name)

    # No suitable second room found — create one
    room_b = await create_room_with_user(client, room_tracker=e2e_created_room_ids)
    return (room_a_id, user_id, user_name), room_b


@pytest.fixture
def e2e_chat_room_with_user(
    e2e_shared_room: tuple[str, str, str],
) -> tuple[str, str, str]:
    """Provide a chat room with a User peer (delegates to session-scoped shared room).

    Returns (chat_id, user_id, user_name) tuple. Uses the session-scoped
    shared room to avoid creating rooms per test. Tests check WS responses
    which are unaffected by prior room history.
    """
    return e2e_shared_room


@pytest.fixture(scope="session")
async def e2e_agent_id(e2e_session_client: AsyncRestClient) -> str:
    """Get the agent ID for the test agent (cached for the entire session).

    Note: Session-scoped because the agent ID is stable for a given API key
    and never changes mid-run. If the underlying agent is recreated between
    tests, this cached value would be stale — but that scenario doesn't
    apply to E2E runs against a persistent platform.
    """
    agent_me = await e2e_session_client.agent_api_identity.get_agent_me()
    return agent_me.data.id


@pytest.fixture
async def ws_client(
    e2e_config: E2ESettings,
) -> AsyncGenerator[TrackingWebSocketClient, None]:
    """WebSocket client for observing agent responses.

    Connects as the **User** (via ``thenvoi_api_key_user``) rather than
    the agent. The platform enforces one WS connection per agent, so a
    second agent connection would kill the Agent's own connection. The
    User is a room participant and receives the same ``message_created``
    events, making it a safe observer that coexists with the Agent.

    Wraps the raw WebSocketClient in a TrackingWebSocketClient that tracks
    joined channels and explicitly leaves them on teardown.
    """
    if not e2e_config.thenvoi_api_key_user:
        pytest.skip("THENVOI_API_KEY_USER not set (needed for WS observer)")

    ws = WebSocketClient(
        ws_url=e2e_config.thenvoi_ws_url,
        api_key=e2e_config.thenvoi_api_key_user,
        agent_id=None,  # User connection, not agent
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
