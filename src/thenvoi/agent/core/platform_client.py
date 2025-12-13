"""
Minimal platform client for Thenvoi integrations.

Provides ONLY the truly common infrastructure:
- Agent validation (agents must be created on platform first)
- WebSocket connection
- API client access

Each framework decides its own message handling, routing, and formatting.
"""

import logging

from thenvoi.client.rest import AsyncRestClient, AgentMe
from thenvoi.client.streaming import WebSocketClient

logger = logging.getLogger(__name__)


class OpenAPISpecIncompleteError(Exception):
    """
    Raised when OpenAPI spec doesn't match actual API behavior.

    This is a workaround for cases where the generated SDK doesn't enforce
    non-null constraints that the real API does. Backend team needs to update
    the OpenAPI spec to properly define these constraints.
    """

    pass


class ThenvoiPlatformClient:
    """
    Minimal client for Thenvoi platform integration.

    Provides only the essential platform infrastructure:
    - Agent validation (agents must be created externally on platform)
    - WebSocket connection management
    - API client access

    This class makes NO assumptions about:
    - How messages are handled
    - How rooms are managed
    - What message format is used
    - How state is maintained

    Each framework builds its own logic on top of this foundation.
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        ws_url: str,
        thenvoi_restapi_url: str,
    ):
        """
        Initialize platform client.

        Args:
            agent_id: Agent ID from platform (agent must be created externally)
            api_key: Agent-specific API key from platform
            ws_url: WebSocket URL
            thenvoi_restapi_url: Base URL for REST API
        """
        self.agent_id = agent_id
        self.api_key = api_key
        self.ws_url = ws_url
        self.thenvoi_restapi_url = thenvoi_restapi_url

        # Create API client
        self.api_client = AsyncRestClient(
            api_key=self.api_key, base_url=self.thenvoi_restapi_url
        )

        # These must be populated by fetch_agent_metadata() before use
        # If accessed before fetch_agent_metadata() is called, will raise AttributeError
        self.name: str
        self.description: str
        self.ws_client: WebSocketClient
        self.platform_agent: AgentMe

    async def fetch_agent_metadata(self) -> AgentMe:
        """
        Fetch agent metadata from platform.

        The agent must be created externally on the platform before calling this method.
        This method fetches and stores agent metadata (name, description) locally.

        Returns:
            Agent object from platform

        Raises:
            UnauthorizedError: If API key is invalid (raised by API)
            ValueError: If agent metadata is missing required fields
            OpenAPISpecIncompleteError: If API returns None (should never happen)
        """
        logger.debug(f"Fetching metadata for agent ID: {self.agent_id}")

        # API raises UnauthorizedError if bad API key
        response = await self.api_client.agent_api.get_agent_me()

        # Type safety check: OpenAPI spec allows response.data to be None but API never returns None
        # (it raises NotFoundError instead). This check satisfies type checker.
        if not response.data:
            raise OpenAPISpecIncompleteError(
                "API returned None for response.data - this should never happen. "
                "API should raise NotFoundError instead. OpenAPI spec needs updating."
            )

        agent = response.data

        if not agent.description:
            raise ValueError(f"Agent {self.agent_id} has no description on platform")

        self.name = agent.name
        self.description = agent.description
        self.platform_agent = agent

        logger.debug(f"Fetched metadata for agent: {self.name} (ID: {self.agent_id})")
        return agent

    async def connect_websocket(self) -> WebSocketClient:
        """
        Create WebSocket connection to platform.

        Returns:
            WebSocket client (use as async context manager)
        """
        logger.debug(f"Creating WebSocket connection to: {self.ws_url}")
        self.ws_client = WebSocketClient(self.ws_url, self.api_key, self.agent_id)
        return self.ws_client
