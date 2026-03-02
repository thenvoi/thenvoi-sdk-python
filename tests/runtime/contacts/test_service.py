"""Tests for ContactService failure and fallback branches."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.runtime.contacts.service import ContactService


@pytest.fixture
def mock_rest_client() -> MagicMock:
    """Mock AsyncRestClient for ContactService tests."""
    client = MagicMock()
    client.agent_api_contacts = MagicMock()
    return client


@pytest.fixture
def contact_service(mock_rest_client: MagicMock) -> ContactService:
    """Create ContactService instance with mocked API client."""
    return ContactService(mock_rest_client)


class TestListContactsFallbacks:
    """Tests for list_contacts metadata fallback behavior."""

    @pytest.mark.asyncio
    async def test_list_contacts_without_metadata_uses_defaults(
        self,
        contact_service: ContactService,
        mock_rest_client: MagicMock,
    ) -> None:
        """Should derive pagination metadata when response metadata is absent."""
        contact = SimpleNamespace(
            id="contact-1",
            handle="@alice",
            name="Alice",
            type="User",
        )
        response = SimpleNamespace(data=[contact], metadata=None)
        mock_rest_client.agent_api_contacts.list_agent_contacts = AsyncMock(
            return_value=response
        )

        result = await contact_service.list_contacts(page=3, page_size=10)

        assert result["contacts"] == [
            {
                "id": "contact-1",
                "handle": "@alice",
                "name": "Alice",
                "type": "User",
            }
        ]
        assert result["metadata"] == {
            "page": 3,
            "page_size": 10,
            "total_count": 1,
            "total_pages": 1,
        }


class TestAddContactErrors:
    """Tests for add_contact error branches."""

    @pytest.mark.asyncio
    async def test_add_contact_raises_when_response_data_is_missing(
        self,
        contact_service: ContactService,
        mock_rest_client: MagicMock,
    ) -> None:
        """Should raise RuntimeError with clear message when API returns no data."""
        mock_rest_client.agent_api_contacts.add_agent_contact = AsyncMock(
            return_value=SimpleNamespace(data=None)
        )

        with pytest.raises(RuntimeError, match="Failed to add contact - no response data"):
            await contact_service.add_contact("@alice")


class TestRemoveContactErrors:
    """Tests for remove_contact error branches."""

    @pytest.mark.asyncio
    async def test_remove_contact_requires_handle_or_contact_id(
        self,
        contact_service: ContactService,
    ) -> None:
        """Should reject calls without any contact identifier."""
        with pytest.raises(ValueError, match="Either handle or contact_id"):
            await contact_service.remove_contact()

    @pytest.mark.asyncio
    async def test_remove_contact_raises_when_response_data_is_missing(
        self,
        contact_service: ContactService,
        mock_rest_client: MagicMock,
    ) -> None:
        """Should raise RuntimeError with clear message when API returns no data."""
        mock_rest_client.agent_api_contacts.remove_agent_contact = AsyncMock(
            return_value=SimpleNamespace(data=None)
        )

        with pytest.raises(
            RuntimeError,
            match="Failed to remove contact - no response data",
        ):
            await contact_service.remove_contact(handle="@alice")


class TestListContactRequestsFallbacks:
    """Tests for list_contact_requests metadata fallback behavior."""

    @pytest.mark.asyncio
    async def test_list_contact_requests_without_metadata_uses_defaults(
        self,
        contact_service: ContactService,
        mock_rest_client: MagicMock,
    ) -> None:
        """Should provide stable metadata defaults when response metadata is absent."""
        received = [
            SimpleNamespace(
                id="req-in-1",
                from_handle="@alice",
                from_name="Alice",
                message="hello",
                status="pending",
                inserted_at=None,
            )
        ]
        sent = [
            SimpleNamespace(
                id="req-out-1",
                to_handle="@bob",
                to_name="Bob",
                message="hi",
                status="pending",
                inserted_at=None,
            )
        ]
        response = SimpleNamespace(
            data=SimpleNamespace(received=received, sent=sent),
            metadata=None,
        )
        mock_rest_client.agent_api_contacts.list_agent_contact_requests = AsyncMock(
            return_value=response
        )

        result = await contact_service.list_contact_requests(page=2, page_size=25)

        assert len(result["received"]) == 1
        assert len(result["sent"]) == 1
        assert result["metadata"] == {
            "page": 2,
            "page_size": 25,
            "received": {"total": 0, "total_pages": 0},
            "sent": {"total": 0, "total_pages": 0},
        }


class TestRespondContactRequestErrors:
    """Tests for respond_contact_request error branches."""

    @pytest.mark.asyncio
    async def test_respond_contact_request_requires_handle_or_request_id(
        self,
        contact_service: ContactService,
    ) -> None:
        """Should reject calls without a request identifier."""
        with pytest.raises(ValueError, match="Either handle or request_id"):
            await contact_service.respond_contact_request(action="approve")

    @pytest.mark.asyncio
    async def test_respond_contact_request_raises_when_response_data_is_missing(
        self,
        contact_service: ContactService,
        mock_rest_client: MagicMock,
    ) -> None:
        """Should raise RuntimeError with clear message when API returns no data."""
        mock_rest_client.agent_api_contacts.respond_to_agent_contact_request = (
            AsyncMock(return_value=SimpleNamespace(data=None))
        )

        with pytest.raises(
            RuntimeError,
            match="Failed to respond to contact request - no response data",
        ):
            await contact_service.respond_contact_request(
                action="approve",
                handle="@alice",
            )
