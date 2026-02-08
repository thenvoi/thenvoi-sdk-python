"""PydanticAI adapter-specific tests.

Tests for PydanticAI adapter-specific behavior that isn't covered by conformance tests:
- History management (persistence between calls)
- Execution reporting (tool_call/tool_result event emission)
- Custom tool support
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from thenvoi.adapters.pydantic_ai import PydanticAIAdapter


def make_stream_events(
    result_messages: list | None = None,
    tool_calls: list[tuple[str, dict, str]] | None = None,
    tool_results: list[tuple[str, str, str]] | None = None,
) -> AsyncIterator:
    """Create a mock async iterator of stream events.

    Args:
        result_messages: Messages to return in AgentRunResultEvent
        tool_calls: List of (tool_name, args, tool_call_id) tuples
        tool_results: List of (tool_name, output, tool_call_id) tuples

    Returns:
        Async iterator of stream events
    """

    async def stream():
        # Emit tool call events
        if tool_calls:
            for tool_name, args, tool_call_id in tool_calls:
                event = MagicMock(spec=FunctionToolCallEvent)
                event.part = MagicMock()
                event.part.tool_name = tool_name
                event.part.args = args
                event.part.tool_call_id = tool_call_id
                yield event

        # Emit tool result events
        if tool_results:
            for tool_name, output, tool_call_id in tool_results:
                event = MagicMock(spec=FunctionToolResultEvent)
                event.result = MagicMock()
                event.result.tool_name = tool_name
                event.result.content = output
                event.tool_call_id = tool_call_id
                yield event

        # Always emit final result event
        result_event = MagicMock(spec=AgentRunResultEvent)
        result_event.result = MagicMock()
        result_event.result.all_messages.return_value = result_messages or []
        yield result_event

    return stream()


@pytest.fixture
def mock_pydantic_agent():
    """Create a mock Pydantic AI Agent."""
    agent = MagicMock()
    agent._function_tools = {
        "thenvoi_send_message": MagicMock(name="thenvoi_send_message"),
        "thenvoi_send_event": MagicMock(name="thenvoi_send_event"),
        "thenvoi_add_participant": MagicMock(name="thenvoi_add_participant"),
        "thenvoi_remove_participant": MagicMock(name="thenvoi_remove_participant"),
        "thenvoi_lookup_peers": MagicMock(name="thenvoi_lookup_peers"),
        "thenvoi_get_participants": MagicMock(name="thenvoi_get_participants"),
        "thenvoi_create_chatroom": MagicMock(name="thenvoi_create_chatroom"),
    }
    return agent


@pytest.fixture
def mock_tools():
    """Create mock AgentToolsProtocol."""
    tools = AsyncMock()
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.add_participant = AsyncMock(return_value={"id": "user-1"})
    tools.remove_participant = AsyncMock(return_value={"status": "removed"})
    tools.lookup_peers = AsyncMock(return_value={"peers": []})
    tools.get_participants = AsyncMock(return_value=[])
    tools.create_chatroom = AsyncMock(return_value="new-room-123")
    return tools


@pytest.fixture
def sample_message():
    """Create a sample platform message."""
    from datetime import datetime, timezone

    from thenvoi.core.types import PlatformMessage

    return PlatformMessage(
        id="msg-123",
        room_id="room-123",
        content="Hello, agent!",
        sender_id="user-456",
        sender_type="User",
        sender_name="Alice",
        message_type="text",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


class TestHistoryManagement:
    """Tests for message history management."""

    @pytest.mark.asyncio
    async def test_updates_history_after_run(
        self, sample_message, mock_tools, mock_pydantic_agent
    ):
        """Should update stored history with all messages from run."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started("TestBot", "Test bot")

        new_messages = [
            ModelRequest(parts=[UserPromptPart(content="Q1")]),
            ModelResponse(parts=[TextPart(content="A1")]),
        ]

        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(result_messages=new_messages)
        )

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        assert adapter._message_history["room-123"] == new_messages

    @pytest.mark.asyncio
    async def test_history_persists_between_calls(
        self, sample_message, mock_tools, mock_pydantic_agent
    ):
        """History should persist between on_message calls."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started("TestBot", "Test bot")

        # First call
        first_messages = [
            ModelRequest(parts=[UserPromptPart(content="Q1")]),
            ModelResponse(parts=[TextPart(content="A1")]),
        ]
        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(result_messages=first_messages)
        )

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        assert len(adapter._message_history["room-123"]) == 2

        # Second call - history should be passed
        second_messages = first_messages + [
            ModelRequest(parts=[UserPromptPart(content="Q2")]),
            ModelResponse(parts=[TextPart(content="A2")]),
        ]
        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(result_messages=second_messages)
        )

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        # Should have passed existing history to run_stream_events
        call_kwargs = adapter._agent.run_stream_events.call_args.kwargs
        assert "message_history" in call_kwargs
        assert len(call_kwargs["message_history"]) == 2  # From first call

    @pytest.mark.asyncio
    async def test_ensures_history_exists_for_non_bootstrap(
        self, sample_message, mock_tools, mock_pydantic_agent
    ):
        """Should create history if not bootstrap and room doesn't exist."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started("TestBot", "Test bot")

        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(result_messages=[])
        )

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=False,  # Not bootstrap
            room_id="new-room",
        )

        # Should have created empty history
        assert "new-room" in adapter._message_history


class TestExecutionReporting:
    """Tests for execution reporting (tool_call and tool_result events)."""

    @pytest.mark.asyncio
    async def test_emits_tool_call_events_when_enabled(
        self, sample_message, mock_tools, mock_pydantic_agent
    ):
        """Should emit tool_call events when enable_execution_reporting=True."""
        adapter = PydanticAIAdapter(
            model="openai:gpt-4o",
            enable_execution_reporting=True,
        )

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started("TestBot", "Test bot")

        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(
                result_messages=[],
                tool_calls=[("thenvoi_send_message", {"content": "Hello"}, "call-123")],
            )
        )

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        # Verify send_event was called with tool_call
        mock_tools.send_event.assert_any_call(
            content='{"name": "thenvoi_send_message", "args": {"content": "Hello"}, "tool_call_id": "call-123"}',
            message_type="tool_call",
        )

    @pytest.mark.asyncio
    async def test_emits_tool_result_events_when_enabled(
        self, sample_message, mock_tools, mock_pydantic_agent
    ):
        """Should emit tool_result events when enable_execution_reporting=True."""
        adapter = PydanticAIAdapter(
            model="openai:gpt-4o",
            enable_execution_reporting=True,
        )

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started("TestBot", "Test bot")

        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(
                result_messages=[],
                tool_results=[
                    ("thenvoi_send_message", "Message sent successfully", "call-123")
                ],
            )
        )

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        # Verify send_event was called with tool_result
        mock_tools.send_event.assert_any_call(
            content='{"name": "thenvoi_send_message", "output": "Message sent successfully", "tool_call_id": "call-123"}',
            message_type="tool_result",
        )

    @pytest.mark.asyncio
    async def test_no_events_when_reporting_disabled(
        self, sample_message, mock_tools, mock_pydantic_agent
    ):
        """Should NOT emit events when enable_execution_reporting=False (default)."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")  # Default is False

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started("TestBot", "Test bot")

        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(
                result_messages=[],
                tool_calls=[("thenvoi_send_message", {"content": "Hello"}, "call-123")],
                tool_results=[("thenvoi_send_message", "Message sent", "call-123")],
            )
        )

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        # Verify send_event was NOT called for tool_call or tool_result
        for call in mock_tools.send_event.call_args_list:
            _, kwargs = call
            assert kwargs.get("message_type") not in ["tool_call", "tool_result"]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_all_reported(
        self, sample_message, mock_tools, mock_pydantic_agent
    ):
        """Should emit events for all tool calls in sequence."""
        adapter = PydanticAIAdapter(
            model="openai:gpt-4o",
            enable_execution_reporting=True,
        )

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started("TestBot", "Test bot")

        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(
                result_messages=[],
                tool_calls=[
                    ("thenvoi_lookup_peers", {}, "call-1"),
                    ("thenvoi_add_participant", {"name": "Helper"}, "call-2"),
                    ("thenvoi_send_message", {"content": "Done"}, "call-3"),
                ],
                tool_results=[
                    ("thenvoi_lookup_peers", "[{...}]", "call-1"),
                    ("thenvoi_add_participant", "Added", "call-2"),
                    ("thenvoi_send_message", "Sent", "call-3"),
                ],
            )
        )

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        # Count tool_call and tool_result events
        tool_call_count = sum(
            1
            for call in mock_tools.send_event.call_args_list
            if call.kwargs.get("message_type") == "tool_call"
        )
        tool_result_count = sum(
            1
            for call in mock_tools.send_event.call_args_list
            if call.kwargs.get("message_type") == "tool_result"
        )

        assert tool_call_count == 3
        assert tool_result_count == 3

    @pytest.mark.asyncio
    async def test_event_failure_does_not_crash_run(
        self, sample_message, mock_pydantic_agent
    ):
        """Should continue running if send_event fails."""
        adapter = PydanticAIAdapter(
            model="openai:gpt-4o",
            enable_execution_reporting=True,
        )

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started("TestBot", "Test bot")

        # Mock tools where send_event fails
        failing_tools = AsyncMock()
        failing_tools.send_event = AsyncMock(side_effect=Exception("Network error"))

        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(
                result_messages=[ModelRequest(parts=[UserPromptPart(content="test")])],
                tool_calls=[("thenvoi_send_message", {"content": "Hello"}, "call-123")],
            )
        )

        # Should not raise
        await adapter.on_message(
            msg=sample_message,
            tools=failing_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        # History should still be updated
        assert "room-123" in adapter._message_history


class TestCustomTools:
    """Tests for custom tool support (PydanticAI-native functions)."""

    def test_accepts_additional_tools_parameter(self):
        """Adapter should accept list of callables."""

        async def my_tool(ctx, message: str) -> str:
            """A custom tool."""
            return f"Echo: {message}"

        adapter = PydanticAIAdapter(
            model="openai:gpt-4o",
            additional_tools=[my_tool],
        )

        assert len(adapter._custom_tools) == 1
        assert adapter._custom_tools[0] == my_tool

    def test_empty_additional_tools_by_default(self):
        """Should have empty custom tools list by default."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")
        assert adapter._custom_tools == []

    def test_multiple_custom_tools(self):
        """Should accept multiple custom tools."""

        async def tool_one(ctx, a: int) -> int:
            """Tool one."""
            return a + 1

        def tool_two(ctx, b: str) -> str:
            """Tool two."""
            return b.upper()

        async def tool_three(ctx, x: float, y: float) -> float:
            """Tool three."""
            return x + y

        adapter = PydanticAIAdapter(
            model="openai:gpt-4o",
            additional_tools=[tool_one, tool_two, tool_three],
        )

        assert len(adapter._custom_tools) == 3
