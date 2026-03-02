"""Integration fixture definitions with minimal cross-module coupling."""

from __future__ import annotations

from collections.abc import AsyncIterator
import logging
import os
import uuid

import pytest
from thenvoi_rest import AsyncRestClient, ChatMessageRequest, ChatRoomRequest

from tests.support.integration.contracts.cleanup import is_no_clean_mode
from tests.support.integration.contracts.markers import (
    enforce_live_fixture_environment,
)
from tests.support.integration.contracts.selection import _select_preferred_peer
from tests.support.integration.contracts.settings import TestSettings, get_test_settings

logger = logging.getLogger(__name__)


@pytest.fixture
def api_client(request: pytest.FixtureRequest) -> AsyncRestClient:
    """Create a real async API client for integration tests (primary agent)."""
    enforce_live_fixture_environment(request, "api_client")

    settings = get_test_settings()
    api_key = settings.thenvoi_api_key
    if not api_key:
        pytest.fail("Expected THENVOI_API_KEY after integration fixture admission")

    return AsyncRestClient(
        api_key=api_key,
        base_url=settings.thenvoi_base_url,
    )


@pytest.fixture
def api_client_2(request: pytest.FixtureRequest) -> AsyncRestClient:
    """Create a real async API client for the secondary agent."""
    enforce_live_fixture_environment(request, "api_client_2")

    settings = get_test_settings()
    api_key = settings.thenvoi_api_key_2
    if not api_key:
        pytest.fail("Expected THENVOI_API_KEY_2 after integration fixture admission")

    return AsyncRestClient(
        api_key=api_key,
        base_url=settings.thenvoi_base_url,
    )


@pytest.fixture
def user_api_client(request: pytest.FixtureRequest) -> AsyncRestClient:
    """Create a real async API client with user API key."""
    enforce_live_fixture_environment(request, "user_api_client")

    settings = get_test_settings()
    api_key = settings.thenvoi_api_key_user
    if not api_key:
        pytest.fail("Expected THENVOI_API_KEY_USER after integration fixture admission")

    return AsyncRestClient(
        api_key=api_key,
        base_url=settings.thenvoi_base_url,
    )


@pytest.fixture
def integration_settings(request: pytest.FixtureRequest) -> TestSettings:
    """Provide test settings to integration tests."""
    enforce_live_fixture_environment(request, "integration_settings")
    return get_test_settings()


@pytest.fixture
def integration_run_id() -> str:
    """Stable run identifier used to namespace integration fixture artifacts."""
    return os.environ.get("THENVOI_TEST_RUN_ID") or f"run-{uuid.uuid4().hex[:10]}"


@pytest.fixture
async def test_chat(
    api_client: AsyncRestClient,
    request: pytest.FixtureRequest,
    integration_run_id: str,
) -> AsyncIterator[str]:
    """Create a temporary chat fixture and clean up after."""
    from thenvoi_rest.types import (
        ChatMessageRequestMentionsItem as Mention,
        ParticipantRequest,
    )

    response = await api_client.agent_api_chats.create_agent_chat(
        chat=ChatRoomRequest()
    )
    chat_id = response.data.id
    fixture_tag = request.node.nodeid.replace("/", "-").replace("::", "-")

    peers_response = await api_client.agent_api_peers.list_agent_peers()
    if peers_response.data:
        peer = _select_preferred_peer(list(peers_response.data))
        if peer is None:
            pytest.fail("No peer available for deterministic test_chat fixture setup")
        await api_client.agent_api_participants.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=peer.id, role="member"),
        )

        await api_client.agent_api_messages.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=(
                    f"[{integration_run_id}:{fixture_tag}] Integration test fixture: "
                    f"@{peer.name} temporary chat for testing participant operations"
                ),
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )

    yield chat_id

    if is_no_clean_mode(request):
        logger.info(
            "No-clean mode active; preserving integration fixture chat %s",
            chat_id,
        )
        return

    delete_chat = getattr(api_client.agent_api_chats, "delete_agent_chat", None)
    if callable(delete_chat):
        try:
            await delete_chat(chat_id=chat_id)
            logger.debug("Deleted integration fixture chat %s", chat_id)
        except Exception as exc:
            logger.warning(
                "Failed to delete integration fixture chat %s: %s", chat_id, exc
            )
    else:
        logger.debug(
            "agent_api_chats.delete_agent_chat unavailable; leaving fixture chat %s",
            chat_id,
        )


@pytest.fixture
async def test_peer_id(api_client: AsyncRestClient) -> str | None:
    """Get a peer ID for participant operations, excluding the owner peer."""
    agent_me = await api_client.agent_api_identity.get_agent_me()
    agent_owner_uuid = (
        str(agent_me.data.owner_uuid) if agent_me.data.owner_uuid else None
    )

    response = await api_client.agent_api_peers.list_agent_peers()
    if response.data:
        selected = _select_preferred_peer(
            list(response.data),
            exclude_peer_id=agent_owner_uuid,
        )
        if selected is not None:
            return selected.id
    return None


__all__ = [
    "api_client",
    "api_client_2",
    "integration_run_id",
    "integration_settings",
    "test_chat",
    "test_peer_id",
    "user_api_client",
]
