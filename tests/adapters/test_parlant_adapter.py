"""Tests for ParlantAdapter."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.adapters.parlant import ParlantAdapter
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
    tools.get_openai_tool_schemas = MagicMock(return_value=[])
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.execute_tool_call = AsyncMock(return_value={"status": "success"})
    return tools


class TestInitialization:
    """Tests for adapter initialization."""

    def test_default_initialization(self):
        """Should initialize with default values."""
        adapter = ParlantAdapter()

        assert adapter.model == "gpt-4o"
        assert adapter.guidelines == []
        assert adapter.enable_execution_reporting is False
        assert adapter.history_converter is not None

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        guidelines = [
            {"condition": "User asks about refunds", "action": "Check order status"}
        ]
        adapter = ParlantAdapter(
            model="gpt-4o-mini",
            custom_section="Be helpful.",
            guidelines=guidelines,
            enable_execution_reporting=True,
        )

        assert adapter.model == "gpt-4o-mini"
        assert adapter.custom_section == "Be helpful."
        assert adapter.guidelines == guidelines
        assert adapter.enable_execution_reporting is True

    def test_system_prompt_override(self):
        """Should use custom system_prompt if provided."""
        adapter = ParlantAdapter(
            system_prompt="You are a custom assistant.",
        )

        assert adapter.system_prompt == "You are a custom assistant."


class TestOnStarted:
    """Tests for on_started() method."""

    @pytest.mark.asyncio
    async def test_renders_system_prompt(self):
        """Should render system prompt from agent metadata."""
        adapter = ParlantAdapter()

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"
        assert adapter._system_prompt != ""
        assert "TestBot" in adapter._system_prompt

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt_when_provided(self):
        """Should use custom system_prompt instead of rendered one."""
        adapter = ParlantAdapter(system_prompt="Custom prompt here.")

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter._system_prompt == "Custom prompt here."


class TestOnMessage:
    """Tests for on_message() method."""

    @pytest.mark.asyncio
    async def test_initializes_history_on_bootstrap(self, sample_message, mock_tools):
        """Should initialize room history on first message."""
        adapter = ParlantAdapter()
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_call_llm") as mock_call:
            mock_call.return_value = {"content": "Hello!", "tool_calls": []}

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
        adapter = ParlantAdapter()
        await adapter.on_started("TestBot", "Test bot")

        existing_history = [
            {"role": "user", "content": "[Bob]: Previous message"},
            {"role": "assistant", "content": "Previous response"},
        ]

        with patch.object(adapter, "_call_llm") as mock_call:
            mock_call.return_value = {"content": "Hello!", "tool_calls": []}

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
        adapter = ParlantAdapter()
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_call_llm") as mock_call:
            mock_call.return_value = {"content": "Hello!", "tool_calls": []}

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
                "[Participant Update]: Alice joined" in str(m.get("content", ""))
                for m in adapter._message_history["room-123"]
            )
            assert found


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_room_history(self, sample_message, mock_tools):
        """Should remove room history on cleanup."""
        adapter = ParlantAdapter()
        await adapter.on_started("TestBot", "Test bot")

        # First add some history
        adapter._message_history["room-123"] = [{"role": "user", "content": "test"}]
        assert "room-123" in adapter._message_history

        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._message_history

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self):
        """Should handle cleanup of non-existent room."""
        adapter = ParlantAdapter()

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")


class TestGuidelines:
    """Tests for behavioral guidelines."""

    def test_format_guidelines_empty(self):
        """Should return empty string for no guidelines."""
        adapter = ParlantAdapter()

        result = adapter._format_guidelines()

        assert result == ""

    def test_format_guidelines_single(self):
        """Should format a single guideline."""
        adapter = ParlantAdapter(
            guidelines=[
                {"condition": "User asks about refunds", "action": "Check order status"}
            ]
        )

        result = adapter._format_guidelines()

        assert "Behavioral Guidelines" in result
        assert "User asks about refunds" in result
        assert "Check order status" in result

    def test_format_guidelines_multiple(self):
        """Should format multiple guidelines."""
        adapter = ParlantAdapter(
            guidelines=[
                {
                    "condition": "User asks about refunds",
                    "action": "Check order status",
                },
                {"condition": "User is angry", "action": "Apologize first"},
                {"condition": "Issue is complex", "action": "Escalate to supervisor"},
            ]
        )

        result = adapter._format_guidelines()

        assert "1." in result
        assert "2." in result
        assert "3." in result
        assert "User asks about refunds" in result
        assert "User is angry" in result
        assert "Issue is complex" in result

    @pytest.mark.asyncio
    async def test_guidelines_included_in_messages(self):
        """Should include guidelines in message context."""
        adapter = ParlantAdapter(
            guidelines=[{"condition": "Test condition", "action": "Test action"}]
        )
        await adapter.on_started("TestBot", "Test bot")

        adapter._message_history["room-123"] = []
        messages = adapter._build_messages("room-123")

        # Should have system prompt + guidelines system message
        assert len(messages) >= 2
        guidelines_found = any(
            "Behavioral Guidelines" in m.get("content", "") for m in messages
        )
        assert guidelines_found


class TestBuildMessages:
    """Tests for _build_messages() method."""

    @pytest.mark.asyncio
    async def test_includes_system_prompt(self):
        """Should include system prompt in messages."""
        adapter = ParlantAdapter()
        await adapter.on_started("TestBot", "Test bot")

        adapter._message_history["room-123"] = []

        messages = adapter._build_messages("room-123")

        assert len(messages) >= 1
        assert messages[0]["role"] == "system"
        assert "TestBot" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_includes_conversation_history(self):
        """Should include conversation history in messages."""
        adapter = ParlantAdapter()
        await adapter.on_started("TestBot", "Test bot")

        adapter._message_history["room-123"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        messages = adapter._build_messages("room-123")

        # System prompt + 2 history messages
        assert len(messages) >= 3


class TestToolExecution:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_reports_tool_calls_when_enabled(self, mock_tools):
        """Should send events when execution reporting is enabled."""
        adapter = ParlantAdapter(enable_execution_reporting=True)

        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "send_message",
                    "arguments": '{"content": "Hello"}',
                },
            }
        ]

        mock_tools.execute_tool_call.return_value = {"status": "success"}

        await adapter._process_tool_calls(tool_calls, mock_tools)

        # Should have sent tool_call and tool_result events
        assert mock_tools.send_event.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_tool_error(self, mock_tools):
        """Should handle tool execution errors gracefully."""
        adapter = ParlantAdapter()

        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "failing_tool",
                    "arguments": "{}",
                },
            }
        ]

        mock_tools.execute_tool_call.side_effect = Exception("Tool failed!")

        results = await adapter._process_tool_calls(tool_calls, mock_tools)

        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert "Tool failed!" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_handles_invalid_json_arguments(self, mock_tools):
        """Should handle invalid JSON in tool arguments."""
        adapter = ParlantAdapter()

        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": "invalid json",
                },
            }
        ]

        mock_tools.execute_tool_call.return_value = {"status": "success"}

        results = await adapter._process_tool_calls(tool_calls, mock_tools)

        # Should still execute with empty arguments
        assert len(results) == 1
        mock_tools.execute_tool_call.assert_called_once_with("test_tool", {})


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_reports_error_on_api_failure(self, sample_message, mock_tools):
        """Should report error when LLM API fails."""
        adapter = ParlantAdapter()
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_call_llm") as mock_call:
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


class TestToolLoop:
    """Tests for the tool loop behavior."""

    @pytest.mark.asyncio
    async def test_stops_after_no_tool_calls(self, sample_message, mock_tools):
        """Should stop when LLM returns no tool calls."""
        adapter = ParlantAdapter()
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_call_llm") as mock_call:
            mock_call.return_value = {
                "content": "Done!",
                "tool_calls": [],
            }

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            # Should only call LLM once
            assert mock_call.call_count == 1

    @pytest.mark.asyncio
    async def test_continues_with_tool_calls(self, sample_message, mock_tools):
        """Should continue looping while there are tool calls."""
        adapter = ParlantAdapter()
        await adapter.on_started("TestBot", "Test bot")

        call_count = 0

        def mock_llm_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call returns a tool call
                return {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "send_message",
                                "arguments": '{"content": "Hi"}',
                            },
                        }
                    ],
                }
            else:
                # Second call returns no tool calls
                return {"content": "Done!", "tool_calls": []}

        with patch.object(adapter, "_call_llm") as mock_call:
            mock_call.side_effect = mock_llm_response

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            # Should call LLM twice (initial + after tool execution)
            assert mock_call.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_prevents_infinite_loop(
        self, sample_message, mock_tools
    ):
        """Should stop after max iterations to prevent infinite loops."""
        adapter = ParlantAdapter()
        adapter._max_tool_iterations = 3
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_call_llm") as mock_call:
            # Always return tool calls (would loop forever without limit)
            mock_call.return_value = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "send_message",
                            "arguments": '{"content": "Hi"}',
                        },
                    }
                ],
            }

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            # Should stop at max iterations
            assert mock_call.call_count == 3


class TestParlantSpecificFeatures:
    """Tests for Parlant-specific features."""

    @pytest.mark.asyncio
    async def test_guideline_based_response(self, sample_message, mock_tools):
        """Should use guidelines in context for responses."""
        adapter = ParlantAdapter(
            guidelines=[
                {
                    "condition": "User asks about pricing",
                    "action": "Refer to pricing page",
                },
                {"condition": "User needs help", "action": "Ask clarifying questions"},
            ]
        )
        await adapter.on_started("Support Agent", "Customer support")

        with patch.object(adapter, "_call_llm") as mock_call:
            mock_call.return_value = {"content": "How can I help?", "tool_calls": []}

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            # Check that guidelines were passed to LLM
            call_args = mock_call.call_args
            messages = call_args[0][0]  # First positional arg is messages

            # Find guidelines in messages
            guidelines_in_messages = any(
                "Behavioral Guidelines" in str(m.get("content", "")) for m in messages
            )
            assert guidelines_in_messages

    def test_adapter_without_guidelines(self):
        """Should work without guidelines (basic mode)."""
        adapter = ParlantAdapter()

        assert adapter.guidelines == []
        assert adapter._format_guidelines() == ""
