"""
Tests for LangGraph agent event functions.

These test the internal event handling functions that provide observability
by streaming tool calls and results back to the platform.
"""

import pytest
from unittest.mock import AsyncMock
from thenvoi.agent.langgraph.agent import (
    _send_platform_event,
    _handle_streaming_event,
)
from thenvoi.client.rest import ChatEventRequest


class TestSendPlatformEvent:
    """Tests for _send_platform_event function."""

    @pytest.mark.asyncio
    async def test_sends_event_with_correct_parameters(self):
        """Verify event is sent with correct room_id, content, and message_type."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        await _send_platform_event(
            api_client=mock_client,
            room_id="room-123",
            content="Test content",
            message_type="tool_call",
        )

        mock_client.agent_api.create_agent_chat_event.assert_called_once()
        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs

        assert call_kwargs["chat_id"] == "room-123"
        assert isinstance(call_kwargs["event"], ChatEventRequest)
        assert call_kwargs["event"].content == "Test content"
        assert call_kwargs["event"].message_type == "tool_call"

    @pytest.mark.asyncio
    async def test_sends_tool_result_message_type(self):
        """Verify tool_result message_type is passed correctly."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        await _send_platform_event(
            api_client=mock_client,
            room_id="room-456",
            content="Result: success",
            message_type="tool_result",
        )

        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs
        assert call_kwargs["event"].message_type == "tool_result"

    @pytest.mark.asyncio
    async def test_sends_error_message_type(self):
        """Verify error message_type is passed correctly."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        await _send_platform_event(
            api_client=mock_client,
            room_id="room-789",
            content="Error: something went wrong",
            message_type="error",
        )

        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs
        assert call_kwargs["event"].message_type == "error"


class TestHandleStreamingEvent:
    """Tests for _handle_streaming_event function."""

    @pytest.mark.asyncio
    async def test_handles_on_tool_start_with_dict_input(self):
        """on_tool_start with dict input sends formatted tool_call."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        event = {
            "event": "on_tool_start",
            "name": "send_message",
            "data": {"input": {"content": "Hello", "room_id": "room-123"}},
        }

        await _handle_streaming_event(event, "room-123", mock_client)

        mock_client.agent_api.create_agent_chat_event.assert_called_once()
        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs

        assert call_kwargs["chat_id"] == "room-123"
        assert call_kwargs["event"].message_type == "tool_call"
        assert "Calling send_message" in call_kwargs["event"].content
        assert "content=Hello" in call_kwargs["event"].content

    @pytest.mark.asyncio
    async def test_handles_on_tool_start_excludes_config_from_args(self):
        """on_tool_start should exclude config from args string."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        event = {
            "event": "on_tool_start",
            "name": "get_participants",
            "data": {
                "input": {"config": {"thread_id": "room-123"}, "room_id": "room-123"}
            },
        }

        await _handle_streaming_event(event, "room-123", mock_client)

        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs
        # config should NOT appear in the formatted content
        assert "config" not in call_kwargs["event"].content
        assert "room_id=room-123" in call_kwargs["event"].content

    @pytest.mark.asyncio
    async def test_handles_on_tool_start_with_string_input(self):
        """on_tool_start with string input converts to string."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        event = {
            "event": "on_tool_start",
            "name": "simple_tool",
            "data": {"input": "simple string input"},
        }

        await _handle_streaming_event(event, "room-123", mock_client)

        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs
        assert (
            "Calling simple_tool(simple string input)" in call_kwargs["event"].content
        )

    @pytest.mark.asyncio
    async def test_handles_on_tool_end_with_string_output(self):
        """on_tool_end with string output sends tool_result."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        event = {
            "event": "on_tool_end",
            "name": "send_message",
            "data": {"output": "Message sent successfully"},
        }

        await _handle_streaming_event(event, "room-456", mock_client)

        mock_client.agent_api.create_agent_chat_event.assert_called_once()
        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs

        assert call_kwargs["chat_id"] == "room-456"
        assert call_kwargs["event"].message_type == "tool_result"
        assert (
            "send_message result: Message sent successfully"
            in call_kwargs["event"].content
        )

    @pytest.mark.asyncio
    async def test_handles_on_tool_end_with_content_attribute(self):
        """on_tool_end extracts .content attribute from output object."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        # Simulate an object with .content attribute (like ToolMessage)
        output_obj = AsyncMock()
        output_obj.content = "Extracted content from object"

        event = {
            "event": "on_tool_end",
            "name": "some_tool",
            "data": {"output": output_obj},
        }

        await _handle_streaming_event(event, "room-123", mock_client)

        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs
        assert "Extracted content from object" in call_kwargs["event"].content

    @pytest.mark.asyncio
    async def test_handles_on_tool_end_with_none_output(self):
        """on_tool_end with None output sends empty string."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        event = {
            "event": "on_tool_end",
            "name": "void_tool",
            "data": {"output": None},
        }

        await _handle_streaming_event(event, "room-123", mock_client)

        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs
        assert "void_tool result: " in call_kwargs["event"].content

    @pytest.mark.asyncio
    async def test_handles_on_tool_end_truncates_long_output(self):
        """on_tool_end truncates output longer than 500 characters."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        # Create output longer than 500 characters
        long_output = "x" * 600

        event = {
            "event": "on_tool_end",
            "name": "verbose_tool",
            "data": {"output": long_output},
        }

        await _handle_streaming_event(event, "room-123", mock_client)

        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs
        content = call_kwargs["event"].content

        # Should be truncated to 500 chars + "..."
        assert content.endswith("...")
        # The full output would be "verbose_tool result: " + 600 x's
        # After truncation: "verbose_tool result: " + 500 x's + "..."
        assert len(content) < len(f"verbose_tool result: {long_output}")

    @pytest.mark.asyncio
    async def test_ignores_unrecognized_event_types(self):
        """Unrecognized event types should not send any events."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        event = {
            "event": "on_chat_model_stream",  # Not handled
            "name": "some_model",
            "data": {"chunk": "streaming data"},
        }

        await _handle_streaming_event(event, "room-123", mock_client)

        # Should not call create_agent_chat_event
        mock_client.agent_api.create_agent_chat_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_on_tool_start_with_empty_input(self):
        """on_tool_start with empty dict input sends tool call with no args."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        event = {
            "event": "on_tool_start",
            "name": "no_args_tool",
            "data": {"input": {}},
        }

        await _handle_streaming_event(event, "room-123", mock_client)

        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs
        assert "Calling no_args_tool()" in call_kwargs["event"].content

    @pytest.mark.asyncio
    async def test_handles_on_tool_start_without_input_key(self):
        """on_tool_start without input key defaults to empty dict."""
        mock_client = AsyncMock()
        mock_client.agent_api.create_agent_chat_event = AsyncMock()

        event = {
            "event": "on_tool_start",
            "name": "edge_case_tool",
            "data": {},  # No "input" key
        }

        await _handle_streaming_event(event, "room-123", mock_client)

        call_kwargs = mock_client.agent_api.create_agent_chat_event.call_args.kwargs
        assert "Calling edge_case_tool()" in call_kwargs["event"].content
