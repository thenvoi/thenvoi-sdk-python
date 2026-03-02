"""Tests for runtime memory service operations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.runtime.memory_service import MemoryService


def _memory_item(**overrides: Any) -> SimpleNamespace:
    data: dict[str, Any] = {
        "id": "mem-1",
        "content": "hello",
        "system": "chat",
        "type": "fact",
        "segment": "general",
        "scope": "subject",
        "status": "active",
        "thought": "remember this",
        "subject_id": "subject-1",
        "source_agent_id": "agent-1",
        "inserted_at": "2024-01-01T00:00:00Z",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _memory_rest_with_list_response(response: Any) -> MagicMock:
    rest = MagicMock()
    rest.agent_api_memories.list_agent_memories = AsyncMock(return_value=response)
    return rest


@pytest.mark.asyncio
async def test_list_memories_returns_serialized_items_and_metadata() -> None:
    response = SimpleNamespace(
        data=[_memory_item(id="mem-1"), _memory_item(id="mem-2", subject_id=None)],
        meta=SimpleNamespace(page_size=10, total_count=42),
    )
    rest = _memory_rest_with_list_response(response)
    service = MemoryService(rest)

    result = await service.list_memories(page_size=10)

    assert [item["id"] for item in result["memories"]] == ["mem-1", "mem-2"]
    assert result["memories"][0]["subject_id"] == "subject-1"
    assert result["memories"][1]["subject_id"] is None
    assert result["metadata"] == {"page_size": 10, "total_count": 42}


@pytest.mark.asyncio
async def test_list_memories_falls_back_when_meta_missing() -> None:
    response = SimpleNamespace(data=[], meta=None)
    rest = _memory_rest_with_list_response(response)
    service = MemoryService(rest)

    result = await service.list_memories(page_size=25)

    assert result == {"memories": [], "metadata": {"page_size": 25, "total_count": 0}}


@pytest.mark.asyncio
async def test_store_memory_builds_create_request_and_returns_data() -> None:
    response = SimpleNamespace(data=_memory_item(id="mem-created"))
    rest = MagicMock()
    rest.agent_api_memories.create_agent_memory = AsyncMock(return_value=response)
    service = MemoryService(rest)

    result = await service.store_memory(
        content="new memory",
        system="ops",
        type="note",
        segment="triage",
        thought="important",
        scope="subject",
        subject_id="subject-9",
        metadata={"priority": "high"},
    )

    assert result["id"] == "mem-created"
    assert result["content"] == "hello"
    create_call = rest.agent_api_memories.create_agent_memory.await_args.kwargs[
        "memory"
    ]
    assert create_call.content == "new memory"
    assert create_call.metadata.model_dump(exclude_none=True) == {"priority": "high"}


@pytest.mark.asyncio
async def test_store_memory_raises_when_response_has_no_data() -> None:
    rest = MagicMock()
    rest.agent_api_memories.create_agent_memory = AsyncMock(
        return_value=SimpleNamespace(data=None)
    )
    service = MemoryService(rest)

    with pytest.raises(RuntimeError, match="Failed to store memory"):
        await service.store_memory(
            content="new memory",
            system="ops",
            type="note",
            segment="triage",
            thought="important",
        )


@pytest.mark.asyncio
async def test_get_memory_returns_serialized_payload() -> None:
    rest = MagicMock()
    rest.agent_api_memories.get_agent_memory = AsyncMock(
        return_value=SimpleNamespace(data=_memory_item(id="mem-get"))
    )
    service = MemoryService(rest)

    result = await service.get_memory("mem-get")

    assert result["id"] == "mem-get"
    assert result["source_agent_id"] == "agent-1"


@pytest.mark.asyncio
async def test_get_memory_raises_when_response_has_no_data() -> None:
    rest = MagicMock()
    rest.agent_api_memories.get_agent_memory = AsyncMock(
        return_value=SimpleNamespace(data=None)
    )
    service = MemoryService(rest)

    with pytest.raises(RuntimeError, match="Failed to get memory"):
        await service.get_memory("mem-missing")


@pytest.mark.asyncio
async def test_supersede_and_archive_memory_return_status_payloads() -> None:
    rest = MagicMock()
    rest.agent_api_memories.supersede_agent_memory = AsyncMock(
        return_value=SimpleNamespace(
            data=SimpleNamespace(id="mem-1", status="superseded")
        )
    )
    rest.agent_api_memories.archive_agent_memory = AsyncMock(
        return_value=SimpleNamespace(
            data=SimpleNamespace(id="mem-1", status="archived")
        )
    )
    service = MemoryService(rest)

    superseded = await service.supersede_memory("mem-1")
    archived = await service.archive_memory("mem-1")

    assert superseded == {"id": "mem-1", "status": "superseded"}
    assert archived == {"id": "mem-1", "status": "archived"}


@pytest.mark.asyncio
async def test_supersede_and_archive_raise_without_response_data() -> None:
    rest = MagicMock()
    rest.agent_api_memories.supersede_agent_memory = AsyncMock(
        return_value=SimpleNamespace(data=None)
    )
    rest.agent_api_memories.archive_agent_memory = AsyncMock(
        return_value=SimpleNamespace(data=None)
    )
    service = MemoryService(rest)

    with pytest.raises(RuntimeError, match="Failed to supersede memory"):
        await service.supersede_memory("mem-1")
    with pytest.raises(RuntimeError, match="Failed to archive memory"):
        await service.archive_memory("mem-1")
