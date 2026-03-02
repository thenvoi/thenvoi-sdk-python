"""Shared contact operation mixin for room-bound and callback tools."""

from __future__ import annotations

from typing import Any

from thenvoi.runtime.contacts.service import ContactService


class ContactOperationsMixin:
    """Provide shared contact CRUD methods via ContactService composition."""

    _contact_service: ContactService

    async def list_contacts(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        """List agent contacts with pagination metadata."""
        return await self._contact_service.list_contacts(page=page, page_size=page_size)

    async def add_contact(
        self, handle: str, message: str | None = None
    ) -> dict[str, Any]:
        """Send a contact request to a handle."""
        return await self._contact_service.add_contact(handle=handle, message=message)

    async def remove_contact(
        self, handle: str | None = None, contact_id: str | None = None
    ) -> dict[str, Any]:
        """Remove contact by handle or contact_id."""
        return await self._contact_service.remove_contact(
            handle=handle,
            contact_id=contact_id,
        )

    async def list_contact_requests(
        self, page: int = 1, page_size: int = 50, sent_status: str = "pending"
    ) -> dict[str, Any]:
        """List received and sent contact requests."""
        return await self._contact_service.list_contact_requests(
            page=page,
            page_size=page_size,
            sent_status=sent_status,
        )

    async def respond_contact_request(
        self, action: str, handle: str | None = None, request_id: str | None = None
    ) -> dict[str, Any]:
        """Approve, reject, or cancel a contact request."""
        return await self._contact_service.respond_contact_request(
            action=action,
            handle=handle,
            request_id=request_id,
        )
