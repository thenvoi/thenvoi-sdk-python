"""
Tests for ThenvoiPlatformClient.

These test the contract between SDK and platform:
- Agent must exist on platform
- Agent metadata is available after fetching
"""

import pytest
from unittest.mock import AsyncMock
from thenvoi.agent.core.platform_client import ThenvoiPlatformClient
from thenvoi.client.rest import UnauthorizedError


async def test_fetches_metadata_for_existing_agent():
    """After fetching, agent name and description should be available."""

    # Setup: create a fake agent object
    class FakeAgent:
        id = "agent-123"
        name = "TestBot"
        description = "A test bot"

    # Mock the API to return this agent
    mock_api = AsyncMock()
    mock_api.agent_api.get_agent_me.return_value = AsyncMock(data=FakeAgent())

    # Create client
    client = ThenvoiPlatformClient(
        agent_id="agent-123",
        api_key="test-key",
        ws_url="ws://localhost",
        thenvoi_restapi_url="http://localhost",
    )

    # Replace the real API client with our mock
    client.api_client = mock_api

    # Act: fetch metadata
    await client.fetch_agent_metadata()

    # Assert: name and description are now available (BEHAVIOR users depend on)
    assert client.name == "TestBot"
    assert client.description == "A test bot"


async def test_raises_when_api_key_invalid():
    """
    Should raise UnauthorizedError when API key is invalid.

    With the new agent_api.get_agent_me(), authentication is via API key,
    so invalid keys raise UnauthorizedError.
    """
    # Mock API to raise UnauthorizedError (matches real API behavior)
    mock_api = AsyncMock()
    mock_api.agent_api.get_agent_me.side_effect = UnauthorizedError("Invalid API key")

    client = ThenvoiPlatformClient(
        agent_id="agent-123",
        api_key="invalid-key",
        ws_url="ws://localhost",
        thenvoi_restapi_url="http://localhost",
    )
    client.api_client = mock_api

    # Should raise UnauthorizedError (real API behavior)
    with pytest.raises(UnauthorizedError):
        await client.fetch_agent_metadata()
