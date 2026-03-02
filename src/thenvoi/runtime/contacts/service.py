"""Shared contact-domain service operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thenvoi.client.rest import AsyncRestClient

logger = logging.getLogger(__name__)


class ContactService:
    """Shared contact-domain operations used by AgentTools and ContactTools."""

    def __init__(self, rest: "AsyncRestClient"):
        self._rest = rest

    async def list_contacts(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        """List agent contacts with pagination metadata."""
        logger.debug("Listing contacts: page=%s, page_size=%s", page, page_size)
        response = await self._rest.agent_api_contacts.list_agent_contacts(
            page=page, page_size=page_size
        )

        contacts = []
        if response.data:
            contacts = [
                {
                    "id": contact.id,
                    "handle": contact.handle,
                    "name": contact.name,
                    "type": contact.type,
                }
                for contact in response.data
            ]

        metadata = {
            "page": response.metadata.page if response.metadata else page,
            "page_size": response.metadata.page_size
            if response.metadata
            else page_size,
            "total_count": response.metadata.total_count
            if response.metadata
            else len(contacts),
            "total_pages": response.metadata.total_pages if response.metadata else 1,
        }
        return {"contacts": contacts, "metadata": metadata}

    async def add_contact(
        self, handle: str, message: str | None = None
    ) -> dict[str, Any]:
        """Send a contact request or auto-approve inverse pending request."""
        logger.debug("Adding contact: handle=%s", handle)
        response = await self._rest.agent_api_contacts.add_agent_contact(
            handle=handle, message=message
        )
        if not response.data:
            raise RuntimeError("Failed to add contact - no response data")
        return {"id": response.data.id, "status": response.data.status}

    async def remove_contact(
        self, handle: str | None = None, contact_id: str | None = None
    ) -> dict[str, Any]:
        """Remove a contact by handle or contact UUID."""
        if handle is None and contact_id is None:
            raise ValueError("Either handle or contact_id must be provided")

        logger.debug("Removing contact: handle=%s, contact_id=%s", handle, contact_id)
        kwargs: dict[str, Any] = {}
        if handle is not None:
            kwargs["handle"] = handle
        if contact_id is not None:
            kwargs["contact_id"] = contact_id

        response = await self._rest.agent_api_contacts.remove_agent_contact(**kwargs)
        if not response.data:
            raise RuntimeError("Failed to remove contact - no response data")
        return {"status": response.data.status}

    async def list_contact_requests(
        self, page: int = 1, page_size: int = 50, sent_status: str = "pending"
    ) -> dict[str, Any]:
        """List received and sent contact requests with separate metadata."""
        logger.debug(
            "Listing contact requests: page=%s, page_size=%s, sent_status=%s",
            page,
            page_size,
            sent_status,
        )
        response = await self._rest.agent_api_contacts.list_agent_contact_requests(
            page=page, page_size=page_size, sent_status=sent_status
        )

        received = []
        if response.data and response.data.received:
            received = [
                {
                    "id": req.id,
                    "from_handle": req.from_handle,
                    "from_name": req.from_name,
                    "message": req.message,
                    "status": req.status,
                    "inserted_at": str(req.inserted_at) if req.inserted_at else None,
                }
                for req in response.data.received
            ]

        sent = []
        if response.data and response.data.sent:
            sent = [
                {
                    "id": req.id,
                    "to_handle": req.to_handle,
                    "to_name": req.to_name,
                    "message": req.message,
                    "status": req.status,
                    "inserted_at": str(req.inserted_at) if req.inserted_at else None,
                }
                for req in response.data.sent
            ]

        metadata = {
            "page": response.metadata.page if response.metadata else page,
            "page_size": response.metadata.page_size
            if response.metadata
            else page_size,
            "received": {
                "total": response.metadata.received.total
                if response.metadata and response.metadata.received
                else 0,
                "total_pages": response.metadata.received.total_pages
                if response.metadata and response.metadata.received
                else 0,
            },
            "sent": {
                "total": response.metadata.sent.total
                if response.metadata and response.metadata.sent
                else 0,
                "total_pages": response.metadata.sent.total_pages
                if response.metadata and response.metadata.sent
                else 0,
            },
        }

        return {"received": received, "sent": sent, "metadata": metadata}

    async def respond_contact_request(
        self, action: str, handle: str | None = None, request_id: str | None = None
    ) -> dict[str, Any]:
        """Approve, reject, or cancel a contact request."""
        if handle is None and request_id is None:
            raise ValueError("Either handle or request_id must be provided")

        logger.debug(
            "Responding to contact request: action=%s, handle=%s, request_id=%s",
            action,
            handle,
            request_id,
        )

        kwargs: dict[str, Any] = {"action": action}
        if handle is not None:
            kwargs["handle"] = handle
        if request_id is not None:
            kwargs["request_id"] = request_id

        response = await self._rest.agent_api_contacts.respond_to_agent_contact_request(
            **kwargs
        )
        if not response.data:
            raise RuntimeError(
                "Failed to respond to contact request - no response data"
            )
        return {"id": response.data.id, "status": response.data.status}
