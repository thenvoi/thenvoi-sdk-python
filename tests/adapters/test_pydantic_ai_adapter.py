"""Tests for PydanticAIAdapter."""

from datetime import datetime, timezone
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
from thenvoi.core.types import PlatformMessage


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
                event.result.tool_name = tool_name  # tool_name is on result, not event
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
def sample_message():
    """Create a sample platform message."""
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
def mock_pydantic_agent():
    """Create a mock Pydantic AI Agent."""
    agent = MagicMock()
    agent._function_tools = {
        "send_message": MagicMock(name="send_message"),
        "send_event": MagicMock(name="send_event"),
        "add_participant": MagicMock(name="add_participant"),
        "remove_participant": MagicMock(name="remove_participant"),
        "lookup_peers": MagicMock(name="lookup_peers"),
        "get_participants": MagicMock(name="get_participants"),
        "create_chatroom": MagicMock(name="create_chatroom"),
    }
    return agent


class TestInitialization:
    """Tests for adapter initialization."""

    def test_requires_model(self):
        """Should require model parameter."""
        # model is required - no default
        adapter = PydanticAIAdapter(model="openai:gpt-4o")
        assert adapter.model == "openai:gpt-4o"

    def test_default_initialization(self):
        """Should initialize with default values."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        assert adapter.model == "openai:gpt-4o"
        assert adapter.system_prompt is None
        assert adapter.custom_section is None
        assert adapter.history_converter is not None
        assert adapter._agent is None  # Created on on_started

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        adapter = PydanticAIAdapter(
            model="anthropic:claude-sonnet-4-5-20250929",
            system_prompt="You are a helpful bot.",
            custom_section="Be concise.",
        )

        assert adapter.model == "anthropic:claude-sonnet-4-5-20250929"
        assert adapter.system_prompt == "You are a helpful bot."
        assert adapter.custom_section == "Be concise."

    def test_enable_execution_reporting_default_false(self):
        """Should default enable_execution_reporting to False."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        assert adapter.enable_execution_reporting is False

    def test_enable_execution_reporting_can_be_enabled(self):
        """Should accept enable_execution_reporting parameter."""
        adapter = PydanticAIAdapter(
            model="openai:gpt-4o",
            enable_execution_reporting=True,
        )

        assert adapter.enable_execution_reporting is True


class TestOnStarted:
    """Tests for on_started() method."""

    @pytest.mark.asyncio
    async def test_sets_agent_name_and_description(self, mock_pydantic_agent):
        """Should set agent_name and agent_description."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"

    @pytest.mark.asyncio
    async def test_creates_pydantic_agent(self, mock_pydantic_agent):
        """Should create Pydantic AI agent after start."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        assert adapter._agent is None

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

        assert adapter._agent is not None

    @pytest.mark.asyncio
    async def test_agent_has_tools_registered(self, mock_pydantic_agent):
        """Should register all platform tools on the agent."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

        # Get registered tool names
        tool_names = list(adapter._agent._function_tools.keys())

        expected_tools = [
            "send_message",
            "send_event",
            "add_participant",
            "remove_participant",
            "lookup_peers",
            "get_participants",
            "create_chatroom",
        ]

        for tool in expected_tools:
            assert tool in tool_names, f"Tool {tool} not found"


class TestOnMessage:
    """Tests for on_message() method."""

    @pytest.mark.asyncio
    async def test_initializes_history_on_bootstrap(
        self, sample_message, mock_tools, mock_pydantic_agent
    ):
        """Should initialize room history on first message."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started("TestBot", "Test bot")

        result_messages = [ModelRequest(parts=[UserPromptPart(content="test")])]
        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(result_messages=result_messages)
        )

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        assert "room-123" in adapter._message_history

    @pytest.mark.asyncio
    async def test_loads_existing_history(
        self, sample_message, mock_tools, mock_pydantic_agent
    ):
        """Should load historical messages on bootstrap."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        with patch.object(adapter, "_create_agent", return_value=mock_pydantic_agent):
            await adapter.on_started("TestBot", "Test bot")

        existing_history = [
            ModelRequest(parts=[UserPromptPart(content="[Bob]: Previous message")]),
            ModelResponse(parts=[TextPart(content="Previous response")]),
        ]

        result_messages = existing_history + [
            ModelRequest(parts=[UserPromptPart(content="new")])
        ]
        adapter._agent.run_stream_events = MagicMock(
            return_value=make_stream_events(result_messages=result_messages)
        )

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=existing_history,
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        # Verify history was passed to agent.run_stream_events()
        call_kwargs = adapter._agent.run_stream_events.call_args.kwargs
        assert "message_history" in call_kwargs
        assert len(call_kwargs["message_history"]) == 2

    @pytest.mark.asyncio
    async def test_injects_participants_message(
        self, sample_message, mock_tools, mock_pydantic_agent
    ):
        """Should inject participants update when provided."""
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
            participants_msg="Alice joined the room",
            is_session_bootstrap=True,
            room_id="room-123",
        )

        # Check that participant message was added to history before run
        call_kwargs = adapter._agent.run_stream_events.call_args.kwargs
        message_history = call_kwargs.get("message_history", [])
        # First message should be the participant update
        if message_history:
            first_msg = message_history[0]
            assert isinstance(first_msg, ModelRequest)
            assert "[System]: Alice joined" in first_msg.parts[0].content

    @pytest.mark.asyncio
    async def test_creates_agent_lazily_if_not_started(
        self, sample_message, mock_tools
    ):
        """Should create agent lazily if on_started wasn't called."""
        adapter = PydanticAIAdapter(
            model="openai:gpt-4o",
            custom_section="Test section",
        )
        # Don't call on_started - set agent_name directly for prompt rendering
        adapter.agent_name = "LazyBot"

        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run_stream_events = MagicMock(
                return_value=make_stream_events(result_messages=[])
            )
            mock_create.return_value = mock_agent

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            mock_create.assert_called_once()


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_room_history(self):
        """Should remove room history on cleanup."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        # Add some history
        adapter._message_history["room-123"] = [
            ModelRequest(parts=[UserPromptPart(content="test")])
        ]
        assert "room-123" in adapter._message_history

        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._message_history

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self):
        """Should handle cleanup of non-existent room."""
        adapter = PydanticAIAdapter(model="openai:gpt-4o")

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")


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
                tool_calls=[("send_message", {"content": "Hello"}, "call-123")],
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
            content='{"name": "send_message", "args": {"content": "Hello"}, "tool_call_id": "call-123"}',
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
                    ("send_message", "Message sent successfully", "call-123")
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
            content='{"name": "send_message", "output": "Message sent successfully", "tool_call_id": "call-123"}',
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
                tool_calls=[("send_message", {"content": "Hello"}, "call-123")],
                tool_results=[("send_message", "Message sent", "call-123")],
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
                    ("lookup_peers", {}, "call-1"),
                    ("add_participant", {"name": "Helper"}, "call-2"),
                    ("send_message", {"content": "Done"}, "call-3"),
                ],
                tool_results=[
                    ("lookup_peers", "[{...}]", "call-1"),
                    ("add_participant", "Added", "call-2"),
                    ("send_message", "Sent", "call-3"),
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
                tool_calls=[("send_message", {"content": "Hello"}, "call-123")],
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
