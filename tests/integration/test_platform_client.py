"""Integration tests for ThenvoiPlatformClient.

Run with: uv run pytest tests/integration/test_platform_client.py -v -s
"""

from tests.integration.conftest import get_test_agent_id, requires_api
from thenvoi.agent.core import ThenvoiPlatformClient


@requires_api
class TestPlatformClientIntegration:
    """Tests for ThenvoiPlatformClient against real API."""

    async def test_platform_client_fetches_metadata(self, integration_settings):
        """Test that ThenvoiPlatformClient can fetch agent metadata."""
        client = ThenvoiPlatformClient(
            agent_id=integration_settings.test_agent_id,
            api_key=integration_settings.thenvoi_api_key,
            ws_url=integration_settings.thenvoi_ws_url,
            thenvoi_restapi_url=integration_settings.thenvoi_base_url,
        )

        await client.fetch_agent_metadata()

        assert client.name is not None
        assert client.description is not None
        print(f"Platform client connected as: {client.name}")
        print(f"Description: {client.description}")

        # Verify agent ID matches what we configured
        expected_agent_id = get_test_agent_id()
        if expected_agent_id:
            assert client.agent_id == expected_agent_id

    async def test_platform_client_has_api_client(self, integration_settings):
        """Test that ThenvoiPlatformClient provides access to API client."""
        client = ThenvoiPlatformClient(
            agent_id=integration_settings.test_agent_id,
            api_key=integration_settings.thenvoi_api_key,
            ws_url=integration_settings.thenvoi_ws_url,
            thenvoi_restapi_url=integration_settings.thenvoi_base_url,
        )

        # API client should be available before fetching metadata
        assert client.api_client is not None

        # Can use it directly
        response = await client.api_client.agent_api.get_agent_me()
        assert response.data is not None
