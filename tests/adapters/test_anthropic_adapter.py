"""Tests for AnthropicAdapter."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.adapters.anthropic import AnthropicAdapter
from thenvoi.core.types import PlatformMessage


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
    tools.get_tool_schemas = MagicMock(return_value=[])
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.execute_tool_call = AsyncMock(return_value={"status": "success"})
    return tools


class TestInitialization:
    """Tests for adapter initialization."""

    def test_default_initialization(self):
        """Should initialize with default values."""
        adapter = AnthropicAdapter()

        assert adapter.model == "claude-sonnet-4-5-20250929"
        assert adapter.max_tokens == 4096
        assert adapter.enable_execution_reporting is False
        assert adapter.history_converter is not None

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        adapter = AnthropicAdapter(
            model="claude-opus-4-20250514",
            max_tokens=8192,
            custom_section="Be helpful.",
            enable_execution_reporting=True,
        )

        assert adapter.model == "claude-opus-4-20250514"
        assert adapter.max_tokens == 8192
        assert adapter.custom_section == "Be helpful."
        assert adapter.enable_execution_reporting is True

    def test_system_prompt_override(self):
        """Should use custom system_prompt if provided."""
        adapter = AnthropicAdapter(
            system_prompt="You are a custom assistant.",
        )

        assert adapter.system_prompt == "You are a custom assistant."


class TestOnStarted:
    """Tests for on_started() method."""

    @pytest.mark.asyncio
    async def test_renders_system_prompt(self):
        """Should render system prompt from agent metadata."""
        adapter = AnthropicAdapter()

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"
        assert adapter._system_prompt != ""
        assert "TestBot" in adapter._system_prompt

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt_when_provided(self):
        """Should use custom system_prompt instead of rendered one."""
        adapter = AnthropicAdapter(system_prompt="Custom prompt here.")

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter._system_prompt == "Custom prompt here."


class TestOnMessage:
    """Tests for on_message() method."""

    @pytest.mark.asyncio
    async def test_initializes_history_on_bootstrap(self, sample_message, mock_tools):
        """Should initialize room history on first message."""
        adapter = AnthropicAdapter()
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_call_anthropic") as mock_call:
            # Create a mock response that ends the conversation
            mock_response = MagicMock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = []
            mock_call.return_value = mock_response

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            assert "room-123" in adapter._message_history
            assert len(adapter._message_history["room-123"]) >= 1

    @pytest.mark.asyncio
    async def test_loads_existing_history(self, sample_message, mock_tools):
        """Should load historical messages on bootstrap."""
        adapter = AnthropicAdapter()
        await adapter.on_started("TestBot", "Test bot")

        existing_history = [
            {"role": "user", "content": "[Bob]: Previous message"},
            {"role": "assistant", "content": "Previous response"},
        ]

        with patch.object(adapter, "_call_anthropic") as mock_call:
            mock_response = MagicMock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = []
            mock_call.return_value = mock_response

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=existing_history,
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            # Should have existing 2 + current message
            assert len(adapter._message_history["room-123"]) >= 3

    @pytest.mark.asyncio
    async def test_injects_participants_message(self, sample_message, mock_tools):
        """Should inject participants update when provided."""
        adapter = AnthropicAdapter()
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_call_anthropic") as mock_call:
            mock_response = MagicMock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = []
            mock_call.return_value = mock_response

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg="Alice joined the room",
                is_session_bootstrap=True,
                room_id="room-123",
            )

            # Find the participants message in history
            found = any(
                "[System]: Alice joined" in str(m.get("content", ""))
                for m in adapter._message_history["room-123"]
            )
            assert found


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_room_history(self, sample_message, mock_tools):
        """Should remove room history on cleanup."""
        adapter = AnthropicAdapter()
        await adapter.on_started("TestBot", "Test bot")

        # First add some history
        adapter._message_history["room-123"] = [{"role": "user", "content": "test"}]
        assert "room-123" in adapter._message_history

        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._message_history

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self):
        """Should handle cleanup of non-existent room."""
        adapter = AnthropicAdapter()

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")


class TestHelperMethods:
    """Tests for internal helper methods."""

    def test_extract_text_content(self):
        """Should extract text from TextBlock content."""
        from anthropic.types import TextBlock

        adapter = AnthropicAdapter()

        content = [
            TextBlock(type="text", text="Hello"),
            TextBlock(type="text", text="World"),
        ]

        result = adapter._extract_text_content(content)

        assert result == "Hello World"

    def test_extract_text_content_empty(self):
        """Should return empty string for empty content."""
        adapter = AnthropicAdapter()

        result = adapter._extract_text_content([])

        assert result == ""

    def test_serialize_content_blocks(self):
        """Should serialize ToolUseBlock and TextBlock."""
        from anthropic.types import TextBlock, ToolUseBlock

        adapter = AnthropicAdapter()

        content = [
            TextBlock(type="text", text="Some text"),
            ToolUseBlock(
                type="tool_use", id="tool-1", name="search", input={"q": "test"}
            ),
        ]

        result = adapter._serialize_content_blocks(content)

        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Some text"
        assert result[1]["type"] == "tool_use"
        assert result[1]["name"] == "search"


class TestToolExecution:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_reports_tool_calls_when_enabled(self, mock_tools):
        """Should send events when execution reporting is enabled."""
        from anthropic.types import ToolUseBlock

        adapter = AnthropicAdapter(enable_execution_reporting=True)

        mock_response = MagicMock()
        mock_response.content = [
            ToolUseBlock(
                type="tool_use",
                id="tool-1",
                name="send_message",
                input={"content": "Hello"},
            )
        ]

        mock_tools.execute_tool_call.return_value = {"status": "success"}

        await adapter._process_tool_calls(mock_response, mock_tools)

        # Should have sent tool_call and tool_result events
        assert mock_tools.send_event.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_tool_error(self, mock_tools):
        """Should handle tool execution errors gracefully."""
        from anthropic.types import ToolUseBlock

        adapter = AnthropicAdapter()

        mock_response = MagicMock()
        mock_response.content = [
            ToolUseBlock(
                type="tool_use",
                id="tool-1",
                name="failing_tool",
                input={},
            )
        ]

        mock_tools.execute_tool_call.side_effect = Exception("Tool failed!")

        results = await adapter._process_tool_calls(mock_response, mock_tools)

        assert len(results) == 1
        assert results[0]["is_error"] is True
        assert "Tool failed!" in results[0]["content"]


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_reports_error_on_api_failure(self, sample_message, mock_tools):
        """Should report error when Anthropic API fails."""
        adapter = AnthropicAdapter()
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_call_anthropic") as mock_call:
            mock_call.side_effect = Exception("API Error")

            with pytest.raises(Exception, match="API Error"):
                await adapter.on_message(
                    msg=sample_message,
                    tools=mock_tools,
                    history=[],
                    participants_msg=None,
                    is_session_bootstrap=True,
                    room_id="room-123",
                )

            # Should have tried to report error
            mock_tools.send_event.assert_called()
