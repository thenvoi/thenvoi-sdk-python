"""
ContactTools - Standalone contact tools for CALLBACK strategy.

Unlike AgentTools which is room-bound, ContactTools is agent-level
and used for programmatic contact handling via callbacks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .operations import ContactOperationsMixin
from .service import ContactService

if TYPE_CHECKING:
    from thenvoi.client.rest import AsyncRestClient

logger = logging.getLogger(__name__)


class ContactTools(ContactOperationsMixin):
    """
    Agent-level contact tools for programmatic contact handling.

    Used by ContactEventHandler to execute contact operations in CALLBACK
    strategy. Operates at agent level (not room-bound).

    Note: ContactTools vs AgentTools
        - ContactTools: Agent-level. Used for programmatic contact handling
          in CALLBACK strategy. Contains only contact management methods.
        - AgentTools: Room-bound. Used by LLM agents in chat rooms.
          Has full tool suite including contacts, but tied to a room.

    Example:
        async def auto_approve(event: ContactEvent, tools: ContactTools) -> None:
            if isinstance(event, ContactRequestReceivedEvent):
                await tools.respond_contact_request("approve", request_id=event.payload.id)

        agent = Agent.create(
            adapter=...,
            contact_config=ContactEventConfig(
                strategy=ContactEventStrategy.CALLBACK,
                on_event=auto_approve,
            ),
        )
    """

    def __init__(self, rest: "AsyncRestClient"):
        """
        Initialize ContactTools.

        Args:
            rest: AsyncRestClient for API calls
        """
        self._rest = rest
        self._contact_service = ContactService(rest)
