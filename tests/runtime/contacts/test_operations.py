"""Tests for shared contact operations mixin."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.runtime.contacts.operations import ContactOperationsMixin


class _ContactTools(ContactOperationsMixin):
    def __init__(self) -> None:
        self._contact_service = MagicMock()
        self._contact_service.list_contacts = AsyncMock(return_value={"contacts": []})
        self._contact_service.add_contact = AsyncMock(return_value={"status": "sent"})
        self._contact_service.remove_contact = AsyncMock(
            return_value={"status": "removed"}
        )
        self._contact_service.list_contact_requests = AsyncMock(
            return_value={"received": [], "sent": []}
        )
        self._contact_service.respond_contact_request = AsyncMock(
            return_value={"status": "approved"}
        )


@pytest.mark.asyncio
async def test_list_contacts_delegates_to_contact_service() -> None:
    tools = _ContactTools()

    result = await tools.list_contacts(page=2, page_size=25)

    assert result == {"contacts": []}
    tools._contact_service.list_contacts.assert_awaited_once_with(page=2, page_size=25)  # noqa: SLF001


@pytest.mark.asyncio
async def test_add_and_remove_contact_delegate_to_contact_service() -> None:
    tools = _ContactTools()

    add_result = await tools.add_contact(handle="@alice", message="hello")
    remove_result = await tools.remove_contact(handle="@alice")

    assert add_result == {"status": "sent"}
    assert remove_result == {"status": "removed"}
    tools._contact_service.add_contact.assert_awaited_once_with(  # noqa: SLF001
        handle="@alice",
        message="hello",
    )
    tools._contact_service.remove_contact.assert_awaited_once_with(  # noqa: SLF001
        handle="@alice",
        contact_id=None,
    )


@pytest.mark.asyncio
async def test_list_requests_and_respond_delegate_to_contact_service() -> None:
    tools = _ContactTools()

    requests = await tools.list_contact_requests(
        page=1, page_size=10, sent_status="pending"
    )
    response = await tools.respond_contact_request(
        action="approve",
        request_id="req-1",
    )

    assert requests == {"received": [], "sent": []}
    assert response == {"status": "approved"}
    tools._contact_service.list_contact_requests.assert_awaited_once_with(  # noqa: SLF001
        page=1,
        page_size=10,
        sent_status="pending",
    )
    tools._contact_service.respond_contact_request.assert_awaited_once_with(  # noqa: SLF001
        action="approve",
        handle=None,
        request_id="req-1",
    )
