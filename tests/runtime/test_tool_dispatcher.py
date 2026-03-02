"""Tests for validated runtime tool dispatch."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from thenvoi.runtime.tool_bridge import ToolExecutionError
from thenvoi.runtime.tool_dispatcher import ToolDispatcher


class _DemoToolInput(BaseModel):
    value: int
    optional: str | None = None


@pytest.mark.asyncio
async def test_execute_or_raise_validates_and_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from thenvoi.runtime import tool_bridge

    dispatch = AsyncMock(return_value={"status": "ok"})
    monkeypatch.setattr(tool_bridge, "dispatch_platform_tool_call", dispatch)
    dispatcher = ToolDispatcher(tool_models={"demo_tool": _DemoToolInput})
    tools = object()

    result = await dispatcher.execute_or_raise(
        tools,
        "demo_tool",
        {"value": 7, "optional": None},
    )

    assert result == {"status": "ok"}
    dispatch.assert_awaited_once_with(tools, "demo_tool", {"value": 7})


@pytest.mark.asyncio
async def test_execute_or_raise_wraps_validation_errors() -> None:
    dispatcher = ToolDispatcher(tool_models={"demo_tool": _DemoToolInput})

    with pytest.raises(ToolExecutionError) as exc_info:
        await dispatcher.execute_or_raise(
            object(),
            "demo_tool",
            {"value": "bad"},
        )

    failure = exc_info.value.failure
    assert failure.tool_name == "demo_tool"
    assert failure.arguments == {"value": "bad"}
    assert "Invalid arguments for demo_tool" in failure.message


@pytest.mark.asyncio
async def test_execute_or_raise_rejects_unknown_tools() -> None:
    dispatcher = ToolDispatcher(tool_models={"demo_tool": _DemoToolInput})

    with pytest.raises(ToolExecutionError) as exc_info:
        await dispatcher.execute_or_raise(object(), "unknown_tool", {"value": 1})

    failure = exc_info.value.failure
    assert failure.tool_name == "unknown_tool"
    assert failure.arguments == {"value": 1}
    assert failure.message == "Unknown tool: unknown_tool"


@pytest.mark.asyncio
async def test_execute_or_raise_wraps_dispatch_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from thenvoi.runtime import tool_bridge

    dispatch = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(tool_bridge, "dispatch_platform_tool_call", dispatch)
    dispatcher = ToolDispatcher(tool_models={"demo_tool": _DemoToolInput})

    with pytest.raises(ToolExecutionError) as exc_info:
        await dispatcher.execute_or_raise(
            object(),
            "demo_tool",
            {"value": 7, "optional": None},
        )

    failure = exc_info.value.failure
    assert failure.tool_name == "demo_tool"
    assert failure.arguments == {"value": 7}
    assert failure.message == "Error executing demo_tool: boom"


@pytest.mark.asyncio
async def test_execute_aliases_execute_or_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = ToolDispatcher(tool_models={"demo_tool": _DemoToolInput})
    execute_or_raise = AsyncMock(return_value={"status": "ok"})
    monkeypatch.setattr(dispatcher, "execute_or_raise", execute_or_raise)
    tools: Any = object()

    result = await dispatcher.execute(tools, "demo_tool", {"value": 1})

    assert result == {"status": "ok"}
    execute_or_raise.assert_awaited_once_with(tools, "demo_tool", {"value": 1})
