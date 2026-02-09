"""Anthropic adapter-specific tests.

Tests for Anthropic adapter-specific behavior that isn't covered by conformance tests:
- Helper methods (_extract_text_content, _serialize_content_blocks)
- History loading from existing data
- Tool execution reporting
- Custom tool execution (_process_tool_calls routing and error handling)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from thenvoi.adapters.anthropic import AnthropicAdapter


# --- Custom tool helpers for TestCustomToolExecution ---


class EchoInput(BaseModel):
    """Echo the message back."""

    message: str = Field(description="Message to echo")


class CalculatorInput(BaseModel):
    """Perform a calculation."""

    a: int = Field(description="First number")
    b: int = Field(description="Second number")


async def echo_message(args: EchoInput) -> str:
    return f"Echo: {args.message}"


async def calculate(args: CalculatorInput) -> dict:
    return {"result": args.a + args.b}


async def failing_tool(args: EchoInput) -> str:
    raise ValueError("Tool exploded")


class TestExtractTextContent:
    """Tests for _extract_text_content helper method."""

    def test_extracts_text_from_text_block(self):
        """Should extract text from TextBlock."""
        from anthropic.types import TextBlock

        adapter = AnthropicAdapter()
        content = [TextBlock(type="text", text="Hello world")]

        result = adapter._extract_text_content(content)

        assert result == "Hello world"

    def test_extracts_text_from_multiple_blocks(self):
        """Should join text from multiple TextBlocks."""
        from anthropic.types import TextBlock

        adapter = AnthropicAdapter()
        content = [
            TextBlock(type="text", text="Hello"),
            TextBlock(type="text", text="world"),
        ]

        result = adapter._extract_text_content(content)

        assert result == "Hello world"

    def test_returns_empty_string_for_empty_content(self):
        """Should return empty string when no text content."""
        adapter = AnthropicAdapter()
        result = adapter._extract_text_content([])
        assert result == ""

    def test_ignores_tool_use_blocks(self):
        """Should ignore ToolUseBlock when extracting text."""
        from anthropic.types import TextBlock, ToolUseBlock

        adapter = AnthropicAdapter()
        content = [
            TextBlock(type="text", text="Before"),
            ToolUseBlock(type="tool_use", id="t1", name="search", input={}),
            TextBlock(type="text", text="After"),
        ]

        result = adapter._extract_text_content(content)

        assert result == "Before After"

    def test_handles_empty_text_blocks(self):
        """Should handle TextBlocks with empty text."""
        from anthropic.types import TextBlock

        adapter = AnthropicAdapter()
        content = [
            TextBlock(type="text", text=""),
            TextBlock(type="text", text="Valid"),
        ]

        result = adapter._extract_text_content(content)

        assert result == "Valid"


class TestSerializeContentBlocks:
    """Tests for _serialize_content_blocks helper method."""

    def test_serializes_tool_use_block(self):
        """Should serialize ToolUseBlock to dict."""
        from anthropic.types import ToolUseBlock

        adapter = AnthropicAdapter()
        content = [
            ToolUseBlock(
                type="tool_use",
                id="toolu_123",
                name="get_weather",
                input={"location": "NYC"},
            )
        ]

        result = adapter._serialize_content_blocks(content)

        assert len(result) == 1
        assert result[0] == {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"location": "NYC"},
        }

    def test_serializes_text_block(self):
        """Should serialize TextBlock to dict."""
        from anthropic.types import TextBlock

        adapter = AnthropicAdapter()
        content = [TextBlock(type="text", text="Hello")]

        result = adapter._serialize_content_blocks(content)

        assert len(result) == 1
        assert result[0] == {"type": "text", "text": "Hello"}

    def test_skips_empty_text_blocks(self):
        """Should skip TextBlocks with empty text."""
        from anthropic.types import TextBlock

        adapter = AnthropicAdapter()
        content = [
            TextBlock(type="text", text=""),
            TextBlock(type="text", text="Valid"),
        ]

        result = adapter._serialize_content_blocks(content)

        assert len(result) == 1
        assert result[0]["text"] == "Valid"

    def test_serializes_mixed_blocks(self):
        """Should serialize mixed block types."""
        from anthropic.types import TextBlock, ToolUseBlock

        adapter = AnthropicAdapter()
        content = [
            TextBlock(type="text", text="Let me search"),
            ToolUseBlock(type="tool_use", id="t1", name="search", input={"q": "test"}),
        ]

        result = adapter._serialize_content_blocks(content)

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "tool_use"


class TestHistoryLoading:
    """Tests for history loading during bootstrap."""

    @pytest.mark.asyncio
    async def test_loads_existing_history_on_bootstrap(self):
        """Should load existing history into _message_history on bootstrap."""
        adapter = AnthropicAdapter()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        existing_history = [
            {"role": "user", "content": "[Alice]: Hello"},
            {"role": "user", "content": "[Bob]: Hi there"},
        ]

        mock_msg = MagicMock()
        mock_msg.id = "msg-123"
        mock_msg.format_for_llm.return_value = "[Alice]: New message"

        mock_tools = MagicMock()
        mock_tools.get_anthropic_tool_schemas.return_value = []

        # Mock the Anthropic API call
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []

        with patch.object(adapter, "_call_anthropic", return_value=mock_response):
            await adapter.on_message(
                msg=mock_msg,
                tools=mock_tools,
                history=existing_history,
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Check that history was loaded
        assert "room-123" in adapter._message_history
        # Should have existing history + new message
        assert len(adapter._message_history["room-123"]) >= 2


class TestToolExecutionReporting:
    """Tests for tool execution reporting."""

    @pytest.mark.asyncio
    async def test_reports_tool_calls_when_enabled(self):
        """Should send tool_call events when enable_execution_reporting=True."""
        from anthropic.types import TextBlock, ToolUseBlock

        adapter = AnthropicAdapter(enable_execution_reporting=True)
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        mock_msg = MagicMock()
        mock_msg.id = "msg-123"
        mock_msg.format_for_llm.return_value = "Test message"

        mock_tools = MagicMock()
        mock_tools.get_anthropic_tool_schemas.return_value = []
        mock_tools.execute_tool_call = AsyncMock(return_value={"result": "ok"})
        mock_tools.send_event = AsyncMock()

        # First response with tool use, second with end_turn
        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [
            ToolUseBlock(type="tool_use", id="t1", name="test_tool", input={"a": 1})
        ]

        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [TextBlock(type="text", text="Done")]

        call_count = [0]

        async def mock_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return tool_response
            return final_response

        with patch.object(adapter, "_call_anthropic", side_effect=mock_call):
            await adapter.on_message(
                msg=mock_msg,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Check that send_event was called for tool_call and tool_result
        send_event_calls = mock_tools.send_event.call_args_list
        message_types = [call.kwargs.get("message_type") for call in send_event_calls]

        assert "tool_call" in message_types
        assert "tool_result" in message_types

    @pytest.mark.asyncio
    async def test_does_not_report_when_disabled(self):
        """Should not send tool events when enable_execution_reporting=False."""
        from anthropic.types import TextBlock, ToolUseBlock

        adapter = AnthropicAdapter(enable_execution_reporting=False)
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        mock_msg = MagicMock()
        mock_msg.id = "msg-123"
        mock_msg.format_for_llm.return_value = "Test message"

        mock_tools = MagicMock()
        mock_tools.get_anthropic_tool_schemas.return_value = []
        mock_tools.execute_tool_call = AsyncMock(return_value={"result": "ok"})
        mock_tools.send_event = AsyncMock()

        # Response with tool use then end
        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [
            ToolUseBlock(type="tool_use", id="t1", name="test_tool", input={})
        ]

        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [TextBlock(type="text", text="Done")]

        call_count = [0]

        async def mock_call(*args, **kwargs):
            call_count[0] += 1
            return tool_response if call_count[0] == 1 else final_response

        with patch.object(adapter, "_call_anthropic", side_effect=mock_call):
            await adapter.on_message(
                msg=mock_msg,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Check that send_event was NOT called for tool events
        for call in mock_tools.send_event.call_args_list:
            msg_type = call.kwargs.get("message_type")
            assert msg_type not in ("tool_call", "tool_result")


class TestCustomToolExecution:
    """Tests for _process_tool_calls() routing and error handling."""

    @pytest.mark.asyncio
    async def test_merges_custom_tool_schemas(self):
        """Custom tool schemas should be appended to platform schemas in _call_anthropic."""
        adapter = AnthropicAdapter(
            additional_tools=[(EchoInput, echo_message)],
        )
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        mock_msg = MagicMock()
        mock_msg.id = "msg-123"
        mock_msg.format_for_llm.return_value = "Test message"

        mock_tools = MagicMock()
        mock_tools.get_anthropic_tool_schemas.return_value = [
            {"name": "thenvoi_send_message", "description": "Send", "input_schema": {}}
        ]

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []

        captured_tools = []

        async def capture_call(messages, tools):
            captured_tools.extend(tools)
            return mock_response

        with patch.object(adapter, "_call_anthropic", side_effect=capture_call):
            await adapter.on_message(
                msg=mock_msg,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        tool_names = [t["name"] for t in captured_tools]
        assert "thenvoi_send_message" in tool_names
        assert "echo" in tool_names

    @pytest.mark.asyncio
    async def test_routes_to_custom_tool(self):
        """find_custom_tool match → execute_custom_tool called, not tools.execute_tool_call."""
        from anthropic.types import ToolUseBlock

        adapter = AnthropicAdapter(
            additional_tools=[(EchoInput, echo_message)],
        )
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        mock_response = MagicMock()
        mock_response.content = [
            ToolUseBlock(
                type="tool_use", id="t1", name="echo", input={"message": "hi"}
            )
        ]

        mock_tools = MagicMock()
        mock_tools.execute_tool_call = AsyncMock()

        results = await adapter._process_tool_calls(mock_response, mock_tools)

        assert len(results) == 1
        assert results[0]["is_error"] is False
        assert "Echo: hi" in results[0]["content"]
        mock_tools.execute_tool_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_routes_to_platform_tool(self):
        """No custom match → tools.execute_tool_call called."""
        from anthropic.types import ToolUseBlock

        adapter = AnthropicAdapter(
            additional_tools=[(EchoInput, echo_message)],
        )
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        mock_response = MagicMock()
        mock_response.content = [
            ToolUseBlock(
                type="tool_use",
                id="t1",
                name="thenvoi_send_message",
                input={"content": "hello"},
            )
        ]

        mock_tools = MagicMock()
        mock_tools.execute_tool_call = AsyncMock(return_value={"status": "sent"})

        results = await adapter._process_tool_calls(mock_response, mock_tools)

        assert len(results) == 1
        assert results[0]["is_error"] is False
        mock_tools.execute_tool_call.assert_awaited_once_with(
            "thenvoi_send_message", {"content": "hello"}
        )

    @pytest.mark.asyncio
    async def test_custom_tool_error_sets_is_error(self):
        """Exception in custom tool → is_error=True in result."""
        from anthropic.types import ToolUseBlock

        adapter = AnthropicAdapter(
            additional_tools=[(EchoInput, failing_tool)],
        )
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        mock_response = MagicMock()
        mock_response.content = [
            ToolUseBlock(
                type="tool_use", id="t1", name="echo", input={"message": "hi"}
            )
        ]

        mock_tools = MagicMock()

        results = await adapter._process_tool_calls(mock_response, mock_tools)

        assert len(results) == 1
        assert results[0]["is_error"] is True
        assert "Tool exploded" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_preserves_tool_use_id_on_error(self):
        """tool_use_id should be preserved in result dict when tool fails."""
        from anthropic.types import ToolUseBlock

        adapter = AnthropicAdapter(
            additional_tools=[(EchoInput, failing_tool)],
        )
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        mock_response = MagicMock()
        mock_response.content = [
            ToolUseBlock(
                type="tool_use", id="toolu_abc123", name="echo", input={"message": "hi"}
            )
        ]

        mock_tools = MagicMock()

        results = await adapter._process_tool_calls(mock_response, mock_tools)

        assert results[0]["tool_use_id"] == "toolu_abc123"

    @pytest.mark.asyncio
    async def test_multiple_custom_tools_execution(self):
        """Two custom tools should both execute correctly via _process_tool_calls."""
        from anthropic.types import ToolUseBlock

        adapter = AnthropicAdapter(
            additional_tools=[
                (EchoInput, echo_message),
                (CalculatorInput, calculate),
            ],
        )
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        mock_response = MagicMock()
        mock_response.content = [
            ToolUseBlock(
                type="tool_use", id="t1", name="echo", input={"message": "hello"}
            ),
            ToolUseBlock(
                type="tool_use", id="t2", name="calculator", input={"a": 3, "b": 4}
            ),
        ]

        mock_tools = MagicMock()

        results = await adapter._process_tool_calls(mock_response, mock_tools)

        assert len(results) == 2
        assert results[0]["is_error"] is False
        assert "Echo: hello" in results[0]["content"]
        assert results[1]["is_error"] is False
        assert "7" in results[1]["content"]
