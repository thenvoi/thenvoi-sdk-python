"""
Tests for ThenvoiPlatformClient.

These test the contract between SDK and platform:
- Agent must exist on platform
- Agent metadata is available after fetching
"""

import pytest
from unittest.mock import AsyncMock
from thenvoi.agent.core.platform_client import ThenvoiPlatformClient
from thenvoi.client.rest import NotFoundError


async def test_fetches_metadata_for_existing_agent():
    """After fetching, agent name and description should be available."""

    # Setup: create a fake agent object
    class FakeAgent:
        id = "agent-123"
        name = "TestBot"
        description = "A test bot"

    # Mock the API to return this agent
    mock_api = AsyncMock()
    mock_api.agents.get_agent.return_value = AsyncMock(data=FakeAgent())

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


async def test_raises_when_agent_not_found():
    """
    Should raise NotFoundError when agent doesn't exist on platform.

    Verified behavior: Real API raises NotFoundError (not returns data=None).
    """
    # Mock API to raise NotFoundError (matches real API behavior)
    mock_api = AsyncMock()
    mock_api.agents.get_agent.side_effect = NotFoundError("Agent not found")

    client = ThenvoiPlatformClient(
        agent_id="nonexistent-agent",
        api_key="test-key",
        ws_url="ws://localhost",
        thenvoi_restapi_url="http://localhost",
    )
    client.api_client = mock_api

    # Should raise NotFoundError (real API behavior, verified with integration test)
    with pytest.raises(NotFoundError):
        await client.fetch_agent_metadata()
