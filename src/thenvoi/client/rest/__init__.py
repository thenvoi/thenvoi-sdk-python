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
    ChatRoomRequest,
    ParticipantRequest,
    NotFoundError,
    UnauthorizedError,
)
from thenvoi_rest.types import ChatMessageRequestMentionsItem

__all__ = [
    "RestClient",
    "AsyncRestClient",
    "AgentMe",
    "ChatMessageRequest",
    "ChatMessageRequestMentionsItem",
    "ChatEventRequest",
    "ChatRoomRequest",
    "ParticipantRequest",
    "NotFoundError",
    "UnauthorizedError",
]
