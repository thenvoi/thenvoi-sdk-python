"""Mock data factories for unit tests.

This module provides factory functions to create mock SDK response objects
for testing without a real API server.

The mocks simulate Pydantic BaseModel behavior by implementing model_dump()
to support serialize_response() in the tools.

Default values are extracted from OpenAPI spec examples (see examples.py).
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

from tests.examples import (
    AGENTME,
    CHATMESSAGE,
    CHATEVENT,
    CHATPARTICIPANT,
    CHATROOM,
    PEER,
)


def make_uuid() -> str:
    """Generate a random UUID string."""
    return str(uuid.uuid4())


def make_timestamp() -> datetime:
    """Generate current UTC timestamp."""
    return datetime.now(timezone.utc)


def make_pydantic_mock(**attrs) -> Mock:
    """Create a Mock that behaves like a Pydantic model.

    Adds model_dump() method that returns the attributes as a dict.
    """
    mock = Mock()
    for key, value in attrs.items():
        setattr(mock, key, value)

    # Add model_dump to simulate Pydantic BaseModel behavior
    def model_dump(**kwargs):
        return {k: v for k, v in attrs.items()}

    mock.model_dump = model_dump
    return mock


class MockDataFactory:
    """Factory for creating mock SDK response objects."""

    @staticmethod
    def agent_me(
        id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        owner_uuid: Optional[str] = None,
    ) -> Mock:
        """Create a mock AgentMe object with OpenAPI example defaults."""
        return make_pydantic_mock(
            id=id or AGENTME["id"],
            name=name or AGENTME["name"],
            description=description or AGENTME["description"],
            owner_uuid=owner_uuid or AGENTME["owner_uuid"],
            inserted_at=AGENTME["inserted_at"],
            updated_at=AGENTME["updated_at"],
        )

    @staticmethod
    def peer(
        id: Optional[str] = None,
        name: Optional[str] = None,
        type: Optional[str] = None,
        description: Optional[str] = None,
        is_external: Optional[bool] = None,
        is_global: Optional[bool] = None,
    ) -> Mock:
        """Create a mock Peer object with OpenAPI example defaults."""
        return make_pydantic_mock(
            id=id or PEER["id"],
            name=name or PEER["name"],
            type=type or PEER["type"],
            description=description or PEER["description"],
            is_external=is_external if is_external is not None else PEER["is_external"],
            is_global=is_global if is_global is not None else PEER["is_global"],
        )

    @staticmethod
    def chat_room(
        id: Optional[str] = None,
        title: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Mock:
        """Create a mock ChatRoom object with OpenAPI example defaults."""
        return make_pydantic_mock(
            id=id or CHATROOM["id"],
            title=title if title is not None else CHATROOM["title"],
            task_id=task_id,
            inserted_at=CHATROOM["inserted_at"],
            updated_at=CHATROOM["updated_at"],
        )

    @staticmethod
    def chat_participant(
        id: Optional[str] = None,
        name: Optional[str] = None,
        type: Optional[str] = None,
        role: str = "member",
        status: Optional[str] = None,
        username: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> Mock:
        """Create a mock ChatParticipant object with OpenAPI example defaults.

        Note: username and display_name are explicitly set to prevent Mock's
        automatic attribute creation from interfering with participant lookups.
        """
        return make_pydantic_mock(
            id=id or CHATPARTICIPANT["id"],
            name=name or CHATPARTICIPANT["name"],
            type=type or CHATPARTICIPANT["type"],
            role=role,
            status=status or CHATPARTICIPANT["status"],
            username=username,
            display_name=display_name,
        )

    @staticmethod
    def chat_message(
        id: Optional[str] = None,
        content: Optional[str] = None,
        chat_room_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        sender_name: Optional[str] = None,
        sender_type: Optional[str] = None,
        message_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Mock:
        """Create a mock ChatMessage object with OpenAPI example defaults."""
        return make_pydantic_mock(
            id=id or CHATMESSAGE["id"],
            content=content or CHATMESSAGE["content"],
            chat_room_id=chat_room_id or CHATMESSAGE["chat_room_id"],
            sender_id=sender_id or CHATMESSAGE["sender_id"],
            sender_name=sender_name or CHATMESSAGE["sender_name"],
            sender_type=sender_type or CHATMESSAGE["sender_type"],
            message_type=message_type or CHATMESSAGE["message_type"],
            metadata=metadata if metadata is not None else CHATMESSAGE["metadata"],
            inserted_at=CHATMESSAGE["inserted_at"],
            updated_at=CHATMESSAGE["updated_at"],
        )

    @staticmethod
    def chat_event(
        id: Optional[str] = None,
        content: Optional[str] = None,
        chat_room_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        sender_name: Optional[str] = None,
        sender_type: Optional[str] = None,
        message_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Mock:
        """Create a mock ChatEvent object with OpenAPI example defaults."""
        return make_pydantic_mock(
            id=id or CHATEVENT["id"],
            content=content or CHATEVENT["content"],
            chat_room_id=chat_room_id or CHATEVENT["chat_room_id"],
            sender_id=sender_id or CHATEVENT["sender_id"],
            sender_name=sender_name or CHATEVENT["sender_name"],
            sender_type=sender_type or CHATEVENT["sender_type"],
            message_type=message_type
            or "thought",  # CHATEVENT doesn't have message_type in OpenAPI
            metadata=metadata if metadata is not None else CHATEVENT["metadata"],
            inserted_at=CHATEVENT["inserted_at"],
            updated_at=CHATEVENT["updated_at"],
        )

    @staticmethod
    def response(data: Any, meta: Optional[Dict[str, Any]] = None) -> Mock:
        """Create a mock API response wrapper.

        Creates a response that can be serialized by serialize_response().
        The data field can be a mock object with model_dump() or None.
        """

        def model_dump(**kwargs):
            if data is None:
                return {"data": None, "meta": meta}
            if hasattr(data, "model_dump"):
                return {"data": data.model_dump(**kwargs), "meta": meta}
            return {"data": data, "meta": meta}

        response = Mock()
        response.data = data
        response.meta = meta
        response.model_dump = model_dump
        return response

    @staticmethod
    def list_response(items: List[Any], meta: Optional[Dict[str, Any]] = None) -> Mock:
        """Create a mock API response for list endpoints.

        Creates a list response that can be serialized by serialize_response().
        """
        meta = meta or {"page": 1, "page_size": 50, "total": len(items)}

        def model_dump(**kwargs):
            data_items = []
            for item in items:
                if hasattr(item, "model_dump"):
                    data_items.append(item.model_dump(**kwargs))
                else:
                    data_items.append(item)
            return {"data": data_items, "meta": meta}

        response = Mock()
        response.data = items
        response.meta = meta
        response.model_dump = model_dump
        return response


# Singleton factory instance for convenience
factory = MockDataFactory()
