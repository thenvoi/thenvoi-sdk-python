"""Shared history-converter integration lifecycle helpers."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging

import pytest
from thenvoi_rest import AsyncRestClient, ChatRoomRequest

logger = logging.getLogger(__name__)


@pytest.fixture
def no_clean(request: pytest.FixtureRequest) -> bool:
    """Return whether integration cleanup is disabled for the current run."""
    return request.config.getoption("--no-clean", default=False)


@asynccontextmanager
async def create_test_chat(
    api_client: AsyncRestClient,
    skip_cleanup: bool = False,
):
    """Create a temporary chat and leave it on teardown unless no-clean is enabled."""
    response = await api_client.agent_api_chats.create_agent_chat(chat=ChatRoomRequest())
    chat_id = response.data.id
    logger.info("Created test chat: %s", chat_id)

    agent_me = await api_client.agent_api_identity.get_agent_me()
    agent_id = agent_me.data.id

    try:
        yield chat_id, agent_me.data
    finally:
        if skip_cleanup:
            logger.info("Skipping cleanup for chat %s (--no-clean)", chat_id)
            return

        try:
            await api_client.agent_api_participants.remove_agent_chat_participant(
                chat_id,
                agent_id,
            )
            logger.info("Cleanup: left chat %s", chat_id)
        except Exception as error:
            logger.warning("Cleanup failed for chat %s: %s", chat_id, error)


__all__ = ["create_test_chat", "logger", "no_clean"]
