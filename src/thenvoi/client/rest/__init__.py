"""
Re-export wrapper for Thenvoi REST API client.

Usage:
    from thenvoi.client.rest import AsyncRestClient
    async_client = AsyncRestClient(api_key="your-api-key")
"""

from thenvoi_rest import (
    RestClient,
    AsyncRestClient,
    AgentMe,
    ChatMessageRequest,
    ChatEventRequest,
    ParticipantRequest,
    NotFoundError,
    UnauthorizedError,
)

__all__ = [
    "RestClient",
    "AsyncRestClient",
    "AgentMe",
    "ChatMessageRequest",
    "ChatEventRequest",
    "ParticipantRequest",
    "NotFoundError",
    "UnauthorizedError",
]
