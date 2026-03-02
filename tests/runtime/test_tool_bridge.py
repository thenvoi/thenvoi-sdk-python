"""Tests for the shared tool-bridge failure contract."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.runtime.tool_bridge import (
    ToolExecutionError,
    as_tool_execution_error,
    build_tool_failure,
    format_tool_error,
    invoke_platform_tool,
)


def test_build_tool_failure_from_exception() -> None:
    """Should normalize an arbitrary exception into ToolFailure."""
    error = RuntimeError("network down")

    failure = build_tool_failure(
        "thenvoi_send_message",
        {"content": "hello", "mentions": ["alice"]},
        error,
    )

    assert failure.tool_name == "thenvoi_send_message"
    assert failure.arguments["content"] == "hello"
    assert isinstance(failure.cause, RuntimeError)
    assert failure.message == "Error sending message: network down"


def test_as_tool_execution_error_is_idempotent() -> None:
    """Wrapping a ToolExecutionError should preserve the original instance."""
    original = as_tool_execution_error(
        "thenvoi_send_event",
        {"content": "thinking"},
        RuntimeError("oops"),
    )

    wrapped = as_tool_execution_error(
        "thenvoi_send_event",
        {"content": "thinking"},
        original,
    )

    assert wrapped is original


@pytest.mark.asyncio
async def test_invoke_platform_tool_wraps_argument_validation_errors() -> None:
    """Invalid argument shapes should raise ToolExecutionError."""
    tools = MagicMock()
    tools.execute_tool_call = AsyncMock()

    with pytest.raises(ToolExecutionError) as exc_info:
        await invoke_platform_tool(
            tools,
            "thenvoi_send_message",
            {"content": "hello", "mentions": "alice"},
        )

    failure = exc_info.value.failure
    assert failure.tool_name == "thenvoi_send_message"
    assert failure.arguments["mentions"] == "alice"
    assert failure.message == "Error sending message: mentions must be a list"


@pytest.mark.asyncio
async def test_invoke_platform_tool_wraps_execution_errors() -> None:
    """Runtime tool errors should raise ToolExecutionError with normalized message."""
    tools = MagicMock()
    tools.execute_tool_call = AsyncMock(side_effect=RuntimeError("connection failed"))

    with pytest.raises(ToolExecutionError) as exc_info:
        await invoke_platform_tool(
            tools,
            "thenvoi_send_message",
            {"content": "hello", "mentions": ["alice"]},
        )

    failure = exc_info.value.failure
    assert failure.tool_name == "thenvoi_send_message"
    assert failure.arguments["mentions"] == ["alice"]
    assert failure.message == "Error sending message: connection failed"
    assert format_tool_error("thenvoi_send_message", {}, exc_info.value) == failure.message


@pytest.mark.asyncio
async def test_invoke_platform_tool_prefers_execute_tool_call() -> None:
    """Canonical path should dispatch via execute_tool_call when available."""
    tools = MagicMock()
    tools.execute_tool_call = AsyncMock(return_value={"status": "ok"})
    tools.send_message = AsyncMock(side_effect=AssertionError("legacy path should not run"))

    result = await invoke_platform_tool(
        tools,
        "thenvoi_send_message",
        {"content": "hello", "mentions": ["alice"]},
    )

    assert result == {"status": "ok"}
    tools.execute_tool_call.assert_awaited_once_with(
        "thenvoi_send_message",
        {"content": "hello", "mentions": ["alice"]},
    )


@pytest.mark.asyncio
async def test_invoke_platform_tool_prefers_execute_tool_call_or_raise() -> None:
    """Typed dispatch path should be preferred when available."""
    tools = MagicMock()
    tools.execute_tool_call_or_raise = AsyncMock(return_value={"status": "typed-ok"})
    tools.execute_tool_call = AsyncMock(
        side_effect=AssertionError("legacy path should not run")
    )

    result = await invoke_platform_tool(
        tools,
        "thenvoi_send_message",
        {"content": "hello", "mentions": ["alice"]},
    )

    assert result == {"status": "typed-ok"}
    tools.execute_tool_call_or_raise.assert_awaited_once_with(
        "thenvoi_send_message",
        {"content": "hello", "mentions": ["alice"]},
    )


@pytest.mark.asyncio
async def test_invoke_platform_tool_preserves_string_success_payloads() -> None:
    """String tool payloads are treated as successful results."""
    tools = MagicMock()
    tools.execute_tool_call = AsyncMock(return_value="plain text payload")

    result = await invoke_platform_tool(
        tools,
        "thenvoi_send_message",
        {"content": "hello", "mentions": ["alice"]},
    )

    assert result == "plain text payload"
