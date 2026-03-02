"""Runtime-checkable protocol conformance tests."""

from __future__ import annotations

from typing import Any

from thenvoi.core import protocols


class _DummyHistoryConverter:
    def convert(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return raw


class _DummyTools:
    async def send_message(
        self,
        content: str,
        mentions: list[str] | list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {"content": content, "mentions": mentions}

    async def send_event(
        self,
        content: str,
        message_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "content": content,
            "message_type": message_type,
            "metadata": metadata,
        }

    async def add_participant(self, name: str, role: str = "member") -> dict[str, Any]:
        return {"name": name, "role": role}

    async def remove_participant(self, name: str) -> dict[str, Any]:
        return {"name": name}

    async def get_participants(self) -> list[dict[str, Any]]:
        return []

    async def lookup_peers(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        return {"peers": [], "metadata": {"page": page, "page_size": page_size}}

    async def create_chatroom(self, task_id: str | None = None) -> str:
        return task_id or "room-1"

    async def list_contacts(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        return {"contacts": [], "metadata": {"page": page, "page_size": page_size}}

    async def add_contact(
        self,
        handle: str,
        message: str | None = None,
    ) -> dict[str, Any]:
        return {"handle": handle, "message": message}

    async def remove_contact(
        self,
        handle: str | None = None,
        contact_id: str | None = None,
    ) -> dict[str, Any]:
        return {"handle": handle, "contact_id": contact_id}

    async def list_contact_requests(
        self,
        page: int = 1,
        page_size: int = 50,
        sent_status: str = "pending",
    ) -> dict[str, Any]:
        return {
            "received": [],
            "sent": [],
            "metadata": {"page": page, "page_size": page_size, "status": sent_status},
        }

    async def respond_contact_request(
        self,
        action: str,
        handle: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return {"action": action, "handle": handle, "request_id": request_id}

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
            "metadata": {
                "subject_id": subject_id,
                "scope": scope,
                "system": system,
                "type": type,
                "segment": segment,
                "content_query": content_query,
                "page_size": page_size,
                "status": status,
            },
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
            "content": content,
            "system": system,
            "type": type,
            "segment": segment,
            "thought": thought,
            "scope": scope,
            "subject_id": subject_id,
            "metadata": metadata,
        }

    async def get_memory(self, memory_id: str) -> dict[str, Any]:
        return {"id": memory_id}

    async def supersede_memory(self, memory_id: str) -> dict[str, Any]:
        return {"id": memory_id, "status": "superseded"}

    async def archive_memory(self, memory_id: str) -> dict[str, Any]:
        return {"id": memory_id, "status": "archived"}

    def get_tool_schemas(
        self,
        format: str,
        *,
        include_memory: bool = False,
    ) -> list[dict[str, Any]]:
        return [{"format": format, "include_memory": include_memory}]

    def get_anthropic_tool_schemas(
        self,
        *,
        include_memory: bool = False,
    ) -> list[dict[str, Any]]:
        return [{"include_memory": include_memory}]

    def get_openai_tool_schemas(
        self,
        *,
        include_memory: bool = False,
    ) -> list[dict[str, Any]]:
        return [{"include_memory": include_memory}]

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return {"tool_name": tool_name, "arguments": arguments}


class _DummyFrameworkAdapter:
    async def on_event(self, inp: Any) -> None:
        del inp

    async def on_cleanup(self, room_id: str) -> None:
        del room_id

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        del agent_name, agent_description


class _DummyPreprocessor:
    async def process(self, ctx: Any, event: Any, agent_id: str) -> Any:
        del ctx, event, agent_id
        return None


def test_history_converter_protocol_runtime_checkable() -> None:
    converter = _DummyHistoryConverter()
    assert isinstance(converter, protocols.HistoryConverter)


def test_tool_protocols_runtime_checkable() -> None:
    tools = _DummyTools()
    assert isinstance(tools, protocols.MessagingToolsProtocol)
    assert isinstance(tools, protocols.ParticipantToolsProtocol)
    assert isinstance(tools, protocols.ChatToolsProtocol)
    assert isinstance(tools, protocols.ContactToolsProtocol)
    assert isinstance(tools, protocols.MemoryToolsProtocol)
    assert isinstance(tools, protocols.PlatformToolOperationsProtocol)
    assert isinstance(tools, protocols.ToolSchemaProviderProtocol)
    assert isinstance(tools, protocols.ToolDispatchProtocol)
    assert isinstance(tools, protocols.MessagingDispatchToolsProtocol)
    assert isinstance(tools, protocols.AnthropicSchemaToolsProtocol)
    assert isinstance(tools, protocols.AgentToolsProtocol)


def test_adapter_protocols_runtime_checkable() -> None:
    assert isinstance(_DummyFrameworkAdapter(), protocols.FrameworkAdapter)
    assert isinstance(_DummyPreprocessor(), protocols.Preprocessor)

