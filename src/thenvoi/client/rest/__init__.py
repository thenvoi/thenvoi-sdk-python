"""
Re-export wrapper for Thenvoi REST API client.

This module re-exports the Fern-generated REST API client under the thenvoi.client.rest namespace.

Usage:
    from thenvoi.client.rest import RestClient, AsyncRestClient

    # Sync client
    client = RestClient(api_key="your-api-key")

    # Async client
    async_client = AsyncRestClient(api_key="your-api-key")
"""

from thenvoi_rest import (
    RestClient,
    AsyncRestClient,
    RestClientEnvironment,
    Agent,
    ChatMessageRequest,
    AddChatParticipantRequestParticipant,
    NotFoundError,
)

__all__ = [
    # Clients
    "RestClient",
    "AsyncRestClient",
    "RestClientEnvironment",
    # Types
    "Agent",
    "ChatMessageRequest",
    "AddChatParticipantRequestParticipant",
    # Errors
    "NotFoundError",
]
