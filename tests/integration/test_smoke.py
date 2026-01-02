"""Smoke tests: verify SDK can connect to local platform.

Run with: uv run pytest tests/integration/test_smoke.py -v -s
"""

from tests.integration.conftest import requires_api


@requires_api
class TestSmokeIntegration:
    """Basic connectivity tests."""

    async def test_can_fetch_agent_me(self, api_client):
        """Verify agent identity can be fetched from real API."""
        response = await api_client.agent_api.get_agent_me()
        assert response.data is not None
        assert response.data.name is not None
        assert response.data.id is not None
        print(f"Connected as agent: {response.data.name} (ID: {response.data.id})")

    async def test_can_list_chats(self, api_client):
        """Verify agent can list its chats."""
        response = await api_client.agent_api.list_agent_chats()
        assert response.data is not None
        print(f"Agent has {len(response.data)} chats")

    async def test_can_list_peers(self, api_client):
        """Verify agent can list available peers."""
        response = await api_client.agent_api.list_agent_peers()
        assert response.data is not None
        print(f"Agent can see {len(response.data)} peers")
