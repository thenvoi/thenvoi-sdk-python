"""Fake AgentTools for unit testing adapters."""

from __future__ import annotations

import uuid
from typing import Any


class FakeAgentTools:
    """
    Fake implementation of AgentToolsProtocol for testing.

    Tracks all calls and allows assertions on tool usage.
    No mocking framework needed - just use this directly.

    Example:
        async def test_adapter_sends_message():
            adapter = MyAdapter()
            tools = FakeAgentTools()

            await adapter.on_message(msg, tools, history, None,
                                     is_session_bootstrap=True, room_id="room-1")

            assert len(tools.messages_sent) == 1
            assert tools.messages_sent[0]["content"] == "Expected response"
    """

    def __init__(self):
        self.messages_sent: list[dict[str, Any]] = []
        self.events_sent: list[dict[str, Any]] = []
        self._participants: list[dict[str, Any]] = []
        self.participants_added: list[dict[str, Any]] = []
        self.participants_removed: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []

    async def send_message(
        self, content: str, mentions: list[str] | list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        msg = {
            "id": f"msg-{len(self.messages_sent)}",
            "content": content,
            "mentions": mentions or [],
        }
        self.messages_sent.append(msg)
        return msg

    async def send_event(
        self,
        content: str,
        message_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "id": f"evt-{len(self.events_sent)}",
            "content": content,
            "message_type": message_type,
            "metadata": metadata or {},
        }
        self.events_sent.append(event)
        return event

    async def add_participant(self, name: str, role: str = "member") -> dict[str, Any]:
        participant = {"id": f"p-{name}", "name": name, "role": role}
        self.participants_added.append(participant)
        return participant

    async def remove_participant(self, name: str) -> dict[str, Any]:
        participant = {"id": f"p-{name}", "name": name}
        self.participants_removed.append(participant)
        return participant

    @property
    def participants(self) -> list[dict[str, Any]]:
        return list(self._participants)

    async def get_participants(self) -> list[dict[str, Any]]:
        return list(self._participants)

    async def lookup_peers(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        return {
            "peers": [],
            "metadata": {"page": page, "page_size": page_size, "total": 0},
        }

    async def create_chatroom(self, task_id: str | None = None) -> str:
        return f"room-{uuid.uuid4()}"

    async def list_contacts(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        return {
            "contacts": [],
            "metadata": {
                "page": page,
                "page_size": page_size,
                "total_count": 0,
                "total_pages": 0,
            },
        }

    async def add_contact(
        self, handle: str, message: str | None = None
    ) -> dict[str, Any]:
        return {"id": str(uuid.uuid4()), "status": "pending"}

    async def remove_contact(
        self, handle: str | None = None, contact_id: str | None = None
    ) -> dict[str, Any]:
        return {"status": "removed"}

    async def list_contact_requests(
        self, page: int = 1, page_size: int = 50, sent_status: str = "pending"
    ) -> dict[str, Any]:
        return {
            "received": [],
            "sent": [],
            "metadata": {
                "page": page,
                "page_size": page_size,
                "received": {"total": 0, "total_pages": 0},
                "sent": {"total": 0, "total_pages": 0},
            },
        }

    async def respond_contact_request(
        self, action: str, handle: str | None = None, request_id: str | None = None
    ) -> dict[str, Any]:
        status_map = {
            "approve": "approved",
            "reject": "rejected",
            "cancel": "cancelled",
        }
        return {
            "id": request_id or str(uuid.uuid4()),
            "status": status_map.get(action, action),
        }

    async def list_memories(
        self,
        subject_id: str | None = None,
        scope: str | None = None,
        system: str | None = None,
        type: str | None = None,
        segment: str | None = None,
        content_query: str | None = None,
        page_size: int = 50,
        status: str | None = None,
    ) -> dict[str, Any]:
        return {
            "memories": [],
            "metadata": {"page_size": page_size, "total_count": 0},
        }

    async def store_memory(
        self,
        content: str,
        system: str,
        type: str,
        segment: str,
        thought: str,
        scope: str = "subject",
        subject_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "system": system,
            "type": type,
            "segment": segment,
            "scope": scope,
            "status": "active",
            "thought": thought,
            "inserted_at": "2025-01-01T00:00:00Z",
        }

    async def get_memory(self, memory_id: str) -> dict[str, Any]:
        return {
            "id": memory_id,
            "content": "Test memory content",
            "system": "long_term",
            "type": "semantic",
            "segment": "user",
            "scope": "subject",
            "status": "active",
            "thought": "Test thought",
            "subject_id": None,
            "source_agent_id": None,
            "inserted_at": "2025-01-01T00:00:00Z",
        }

    async def supersede_memory(self, memory_id: str) -> dict[str, Any]:
        return {"id": memory_id, "status": "superseded"}

    async def archive_memory(self, memory_id: str) -> dict[str, Any]:
        return {"id": memory_id, "status": "archived"}

    def get_tool_schemas(
        self,
        format: str,
        *,
        include_memory: bool = False,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        include_categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return []

    def get_anthropic_tool_schemas(
        self,
        *,
        include_memory: bool = False,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        include_categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return []

    def get_openai_tool_schemas(
        self,
        *,
        include_memory: bool = False,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        include_categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return []

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        call = {"tool_name": tool_name, "arguments": arguments}
        self.tool_calls.append(call)
        return {"status": "ok"}
