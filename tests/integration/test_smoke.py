"""Smoke tests: verify SDK can connect to local platform.

Run with: uv run pytest tests/integration/test_smoke.py -v -s
"""

import logging

from tests.integration.conftest import requires_api

logger = logging.getLogger(__name__)


@requires_api
class TestSmokeIntegration:
    """Basic connectivity tests."""

    async def test_can_fetch_agent_me(self, api_client):
        """Verify agent identity can be fetched from real API."""
        response = await api_client.agent_api_identity.get_agent_me()
        assert response.data is not None
        assert response.data.name is not None
        assert response.data.id is not None
        logger.info(
            "Connected as agent: %s (ID: %s)", response.data.name, response.data.id
        )

    async def test_can_list_chats(self, api_client):
        """Verify agent can list its chats."""
        response = await api_client.agent_api_chats.list_agent_chats()
        assert response.data is not None
        logger.info("Agent has %s chats", len(response.data))

    async def test_can_list_peers(self, api_client):
        """Verify agent can list available peers."""
        response = await api_client.agent_api_peers.list_agent_peers()
        assert response.data is not None
        logger.info("Agent can see %s peers", len(response.data))
