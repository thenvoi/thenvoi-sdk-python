"""Tests for GoogleADKAdapter.

Tests for shared adapter behavior (initialization defaults, custom kwargs,
history_converter, on_started agent_name/description, on_message callable,
cleanup safety) live in tests/framework_conformance/test_adapter_conformance.py.
This file contains Google ADK-specific behavior: system prompt rendering,
session management, tool bridging, custom tools, and error handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from thenvoi.adapters.google_adk import GoogleADKAdapter, _ThenvoiToolBridge
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
    """Create mock AgentToolsProtocol (MagicMock base, AsyncMock methods)."""
    tools = MagicMock()
    tools.get_tool_schemas = MagicMock(return_value=[])
    tools.get_openai_tool_schemas = MagicMock(
        return_value=[
            {
                "type": "function",
                "function": {
                    "name": "thenvoi_send_message",
                    "description": "Send a message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "mentions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["content", "mentions"],
                    },
                },
            }
        ]
    )
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.execute_tool_call = AsyncMock(return_value={"status": "success"})
    return tools


class TestInitialization:
    """Tests for adapter initialization."""

    def test_default_model(self):
        """Should default to gemini-2.5-flash."""
        adapter = GoogleADKAdapter()
        assert adapter.model == "gemini-2.5-flash"

    def test_custom_model(self):
        """Should accept custom model."""
        adapter = GoogleADKAdapter(model="gemini-2.5-pro")
        assert adapter.model == "gemini-2.5-pro"

    def test_system_prompt_override(self):
        """Should use custom system_prompt if provided."""
        adapter = GoogleADKAdapter(system_prompt="You are a custom assistant.")
        assert adapter.system_prompt == "You are a custom assistant."

    def test_custom_section(self):
        """Should store custom section."""
        adapter = GoogleADKAdapter(custom_section="Be helpful.")
        assert adapter.custom_section == "Be helpful."

    def test_execution_reporting_default(self):
        """Should default execution reporting to False."""
        adapter = GoogleADKAdapter()
        assert adapter.enable_execution_reporting is False

    def test_memory_tools_default(self):
        """Should default memory tools to False."""
        adapter = GoogleADKAdapter()
        assert adapter.enable_memory_tools is False

    def test_empty_custom_tools(self):
        """Should default to empty custom tools list."""
        adapter = GoogleADKAdapter()
        assert adapter._custom_tools == []

    def test_history_converter_set(self):
        """Should have history converter set by default."""
        adapter = GoogleADKAdapter()
        assert adapter.history_converter is not None


class TestOnStarted:
    """Tests for on_started() method."""

    @pytest.mark.asyncio
    async def test_renders_system_prompt(self):
        """Should render system prompt from agent metadata."""
        adapter = GoogleADKAdapter()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter._system_prompt != ""
        assert "TestBot" in adapter._system_prompt

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt_when_provided(self):
        """Should use custom system_prompt instead of rendered one."""
        adapter = GoogleADKAdapter(system_prompt="Custom prompt here.")
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter._system_prompt == "Custom prompt here."

    @pytest.mark.asyncio
    async def test_sets_agent_name(self):
        """Should set agent_name on the adapter."""
        adapter = GoogleADKAdapter()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"


class TestOnMessage:
    """Tests for on_message() method."""

    @pytest.mark.asyncio
    async def test_creates_session_on_bootstrap(self, sample_message, mock_tools):
        """Should create a new session on first message."""
        adapter = GoogleADKAdapter()
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_create_runner") as mock_create:
            mock_runner = AsyncMock()
            mock_runner.run_async = MagicMock(return_value=_empty_async_iter())
            mock_runner.close = AsyncMock()
            mock_create.return_value = mock_runner

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            assert "room-123" in adapter._room_sessions

    @pytest.mark.asyncio
    async def test_reuses_session_on_subsequent_messages(
        self, sample_message, mock_tools
    ):
        """Should reuse existing session for same room."""
        adapter = GoogleADKAdapter()
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_create_runner") as mock_create:
            mock_runner = AsyncMock()
            mock_runner.run_async = MagicMock(return_value=_empty_async_iter())
            mock_runner.close = AsyncMock()
            mock_create.return_value = mock_runner

            # First message (bootstrap)
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            session_id_1 = adapter._room_sessions["room-123"]

            # Second message (not bootstrap)
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=False,
                room_id="room-123",
            )

            session_id_2 = adapter._room_sessions["room-123"]
            assert session_id_1 == session_id_2

    @pytest.mark.asyncio
    async def test_injects_participants_message(self, sample_message, mock_tools):
        """Should include participants update in message."""
        adapter = GoogleADKAdapter()
        await adapter.on_started("TestBot", "Test bot")

        captured_messages = []

        with patch.object(adapter, "_create_runner") as mock_create:
            mock_runner = AsyncMock()

            def capture_run(**kwargs):
                captured_messages.append(kwargs.get("new_message"))
                return _empty_async_iter()

            mock_runner.run_async = capture_run
            mock_runner.close = AsyncMock()
            mock_create.return_value = mock_runner

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg="Alice joined the room",
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        assert len(captured_messages) == 1
        content = captured_messages[0]
        text = content.parts[0].text
        assert "[System]: Alice joined the room" in text

    @pytest.mark.asyncio
    async def test_injects_history_on_bootstrap(self, sample_message, mock_tools):
        """Should inject history transcript on bootstrap."""
        adapter = GoogleADKAdapter()
        await adapter.on_started("TestBot", "Test bot")

        history = [
            {"role": "user", "content": "[Bob]: Previous message"},
            {"role": "model", "content": "Previous response"},
        ]

        captured_messages = []

        with patch.object(adapter, "_create_runner") as mock_create:
            mock_runner = AsyncMock()

            def capture_run(**kwargs):
                captured_messages.append(kwargs.get("new_message"))
                return _empty_async_iter()

            mock_runner.run_async = capture_run
            mock_runner.close = AsyncMock()
            mock_create.return_value = mock_runner

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=history,
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        assert len(captured_messages) == 1
        text = captured_messages[0].parts[0].text
        assert "Previous conversation context" in text
        assert "[Bob]: Previous message" in text


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_room_session(self):
        """Should remove room session on cleanup."""
        adapter = GoogleADKAdapter()
        adapter._room_sessions["room-123"] = "session-abc"

        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._room_sessions

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room(self):
        """Should not raise when cleaning up nonexistent room."""
        adapter = GoogleADKAdapter()
        await adapter.on_cleanup("nonexistent-room")


class TestToolBridge:
    """Tests for _ThenvoiToolBridge."""

    def test_tool_name(self):
        """Should return the tool name."""
        bridge = _ThenvoiToolBridge(
            tool_name="thenvoi_send_message",
            tool_description="Send a message",
            parameters_schema={},
            tools_ref=[None],
            custom_tools_ref=[[]],
        )
        assert bridge.name == "thenvoi_send_message"

    def test_tool_description(self):
        """Should return the tool description."""
        bridge = _ThenvoiToolBridge(
            tool_name="test_tool",
            tool_description="A test tool",
            parameters_schema={},
            tools_ref=[None],
            custom_tools_ref=[[]],
        )
        assert bridge.description == "A test tool"

    @pytest.mark.asyncio
    async def test_executes_platform_tool(self):
        """Should delegate to AgentToolsProtocol."""
        mock_tools = MagicMock()
        mock_tools.execute_tool_call = AsyncMock(return_value={"status": "sent"})

        bridge = _ThenvoiToolBridge(
            tool_name="thenvoi_send_message",
            tool_description="Send a message",
            parameters_schema={},
            tools_ref=[mock_tools],
            custom_tools_ref=[[]],
        )

        result = await bridge.run_async(
            args={"content": "Hello", "mentions": ["@alice"]},
            tool_context=MagicMock(),
        )

        mock_tools.execute_tool_call.assert_called_once_with(
            "thenvoi_send_message", {"content": "Hello", "mentions": ["@alice"]}
        )
        assert "sent" in result

    @pytest.mark.asyncio
    async def test_returns_error_when_no_tools_context(self):
        """Should return error when tools context is not set."""
        bridge = _ThenvoiToolBridge(
            tool_name="test_tool",
            tool_description="Test",
            parameters_schema={},
            tools_ref=[None],
            custom_tools_ref=[[]],
        )

        result = await bridge.run_async(args={}, tool_context=MagicMock())
        assert isinstance(result, dict)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handles_tool_error(self):
        """Should return error string on tool failure."""
        mock_tools = MagicMock()
        mock_tools.execute_tool_call = AsyncMock(side_effect=Exception("Tool failed!"))

        bridge = _ThenvoiToolBridge(
            tool_name="failing_tool",
            tool_description="Fails",
            parameters_schema={},
            tools_ref=[mock_tools],
            custom_tools_ref=[[]],
        )

        result = await bridge.run_async(args={}, tool_context=MagicMock())
        assert "Error" in result
        assert "Tool failed!" in result

    def test_strips_additional_properties(self):
        """Should strip additionalProperties from schema for Gemini compatibility."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nested": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
            "required": ["name"],
        }

        cleaned = _ThenvoiToolBridge._convert_parameters(schema)

        assert "additionalProperties" not in cleaned
        assert "additionalProperties" not in cleaned["properties"]["nested"]
        assert cleaned["properties"]["name"] == {"type": "string"}
        assert cleaned["required"] == ["name"]


class TestBuildADKTools:
    """Tests for _build_adk_tools."""

    def test_builds_platform_tools(self, mock_tools):
        """Should create tool bridges from platform tool schemas."""
        adapter = GoogleADKAdapter()
        bridges = adapter._build_adk_tools(mock_tools)

        assert len(bridges) == 1
        assert bridges[0].name == "thenvoi_send_message"

    def test_includes_custom_tools(self, mock_tools):
        """Should include custom tools in the bridge list."""

        class EchoInput(BaseModel):
            """Echo back the message."""

            message: str = Field(description="Message to echo")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        adapter = GoogleADKAdapter(additional_tools=[(EchoInput, echo)])
        bridges = adapter._build_adk_tools(mock_tools)

        assert len(bridges) == 2
        tool_names = [b.name for b in bridges]
        assert "thenvoi_send_message" in tool_names
        assert "echo" in tool_names


class TestCustomTools:
    """Tests for custom tool support."""

    def test_accepts_additional_tools_parameter(self):
        """Adapter should accept list of (Model, func) tuples."""

        class EchoInput(BaseModel):
            """Echo back the message."""

            message: str = Field(description="Message to echo")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        adapter = GoogleADKAdapter(additional_tools=[(EchoInput, echo)])
        assert len(adapter._custom_tools) == 1

    @pytest.mark.asyncio
    async def test_custom_tool_execution_via_bridge(self):
        """Custom tool should be executed via bridge."""

        class EchoInput(BaseModel):
            """Echo back the message."""

            message: str = Field(description="Message to echo")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        mock_tools = MagicMock()
        mock_tools.execute_tool_call = AsyncMock()

        bridge = _ThenvoiToolBridge(
            tool_name="echo",
            tool_description="Echo back the message",
            parameters_schema={},
            tools_ref=[mock_tools],
            custom_tools_ref=[[(EchoInput, echo)]],
        )

        result = await bridge.run_async(
            args={"message": "Hello"},
            tool_context=MagicMock(),
        )

        # Should NOT have called platform tool
        mock_tools.execute_tool_call.assert_not_called()
        assert "Echo: Hello" in result


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_reports_error_on_runner_failure(self, sample_message, mock_tools):
        """Should report error when ADK runner fails."""
        adapter = GoogleADKAdapter()
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_create_runner") as mock_create:
            mock_runner = AsyncMock()

            async def failing_run(**kwargs):
                raise Exception("Runner Error")
                yield  # Make it an async generator  # noqa: E501

            mock_runner.run_async = failing_run
            mock_runner.close = AsyncMock()
            mock_create.return_value = mock_runner

            with pytest.raises(Exception, match="Runner Error"):
                await adapter.on_message(
                    msg=sample_message,
                    tools=mock_tools,
                    history=[],
                    participants_msg=None,
                    contacts_msg=None,
                    is_session_bootstrap=True,
                    room_id="room-123",
                )

            mock_tools.send_event.assert_called()


class TestHistoryTranscript:
    """Tests for _format_history_transcript."""

    def test_formats_text_messages(self):
        """Should format text messages."""
        adapter = GoogleADKAdapter()
        history = [
            {"role": "user", "content": "[Alice]: Hello"},
            {"role": "user", "content": "[Bob]: Hi there"},
        ]

        transcript = adapter._format_history_transcript(history)

        assert "[Alice]: Hello" in transcript
        assert "[Bob]: Hi there" in transcript

    def test_formats_tool_calls(self):
        """Should format tool call blocks."""
        adapter = GoogleADKAdapter()
        history = [
            {
                "role": "model",
                "content": [
                    {
                        "type": "function_call",
                        "name": "thenvoi_send_message",
                        "args": {"content": "Hello"},
                    }
                ],
            }
        ]

        transcript = adapter._format_history_transcript(history)

        assert "[Tool Call]" in transcript
        assert "thenvoi_send_message" in transcript

    def test_formats_tool_results(self):
        """Should format tool result blocks."""
        adapter = GoogleADKAdapter()
        history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "function_response",
                        "name": "thenvoi_send_message",
                        "output": '{"status": "sent"}',
                    }
                ],
            }
        ]

        transcript = adapter._format_history_transcript(history)

        assert "[Tool Result]" in transcript
        assert "thenvoi_send_message" in transcript

    def test_empty_history(self):
        """Should return empty string for empty history."""
        adapter = GoogleADKAdapter()
        assert adapter._format_history_transcript([]) == ""


# Helper for creating empty async iterators in tests
async def _empty_async_iter(**kwargs):
    return
    yield  # Make it an async generator
