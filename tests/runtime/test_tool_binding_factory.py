"""Tests for shared tool binding factory helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from thenvoi.runtime.tool_binding_factory import (
    ToolBindingFactory,
    ToolBindingLookupError,
)
from thenvoi.runtime.tool_sessions import ToolSessionRegistry


class _DummyDispatchTools:
    def __init__(self) -> None:
        self.execute_tool_call = AsyncMock()


@pytest.mark.asyncio
async def test_invoke_room_tool_dispatches_via_lookup() -> None:
    factory = ToolBindingFactory()
    tools = _DummyDispatchTools()
    tools.execute_tool_call.return_value = {"status": "ok"}

    result = await factory.invoke_room_tool(
        room_id="room-1",
        get_tools=lambda room_id: tools if room_id == "room-1" else None,
        tool_name="thenvoi_send_event",
        arguments={"content": "thinking", "message_type": "thought"},
    )

    assert result == {"status": "ok"}
    tools.execute_tool_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_invoke_room_tool_raises_for_missing_tools() -> None:
    factory = ToolBindingFactory()
    with pytest.raises(ToolBindingLookupError, match="No tools available for room room-2"):
        await factory.invoke_room_tool(
            room_id="room-2",
            get_tools=lambda _room_id: None,
            tool_name="thenvoi_send_message",
            arguments={"content": "hi", "mentions": []},
        )


@pytest.mark.asyncio
async def test_invoke_session_tool_dispatches_via_registry() -> None:
    factory = ToolBindingFactory()
    registry: ToolSessionRegistry[Any] = ToolSessionRegistry()
    tools = _DummyDispatchTools()
    tools.execute_tool_call.return_value = "done"
    registry.set_tools("sess-1", tools)

    result = await factory.invoke_session_tool(
        session_id="sess-1",
        registry=registry,
        tool_name="thenvoi_get_participants",
        arguments={},
    )

    assert result == "done"
    tools.execute_tool_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_mcp_room_handler_success_and_hooks() -> None:
    factory = ToolBindingFactory()
    tools = _DummyDispatchTools()
    tools.execute_tool_call.return_value = {"ok": True}

    on_success = AsyncMock()

    handler = factory.build_mcp_room_handler(
        tool_name="thenvoi_send_message",
        get_tools=lambda room_id: tools if room_id == "room-1" else None,
        map_args=lambda args: {"content": args["content"], "mentions": []},
        format_success_payload=lambda args, result: {
            "content": args["content"],
            "result": result,
        },
        success_wrapper=lambda payload: {"ok": payload},
        error_wrapper=lambda message: {"error": message},
        on_success=lambda room_id, result: on_success(room_id, result),
    )

    result = await handler({"room_id": "room-1", "content": "hello"})

    assert result == {"ok": {"content": "hello", "result": {"ok": True}}}
    on_success.assert_awaited_once_with("room-1", {"ok": True})


@pytest.mark.asyncio
async def test_build_mcp_room_handler_error_wrapping() -> None:
    factory = ToolBindingFactory()

    handler = factory.build_mcp_room_handler(
        tool_name="thenvoi_send_message",
        get_tools=lambda _room_id: None,
        map_args=lambda args: args,
        format_success_payload=lambda _args, result: result,
        success_wrapper=lambda payload: {"ok": payload},
        error_wrapper=lambda message: {"error": message},
        missing_tools_message="missing tools",
        format_error_message=lambda error, room_id: f"{room_id}:{error}",
    )

    missing_result = await handler({"room_id": "room-1", "content": "x"})
    assert missing_result == {"error": "missing tools"}

