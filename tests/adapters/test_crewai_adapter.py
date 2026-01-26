"""Tests for CrewAIAdapter."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from thenvoi.adapters.crewai import CrewAIAdapter
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
        adapter = CrewAIAdapter()

        assert adapter.model == "gpt-4o"
        assert adapter.role is None
        assert adapter.goal is None
        assert adapter.backstory is None
        assert adapter.enable_execution_reporting is False
        assert adapter.verbose is False
        assert adapter.history_converter is not None

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        adapter = CrewAIAdapter(
            model="gpt-4o-mini",
            role="Research Analyst",
            goal="Find and analyze information",
            backstory="Expert researcher with years of experience",
            custom_section="Be thorough.",
            enable_execution_reporting=True,
            verbose=True,
        )

        assert adapter.model == "gpt-4o-mini"
        assert adapter.role == "Research Analyst"
        assert adapter.goal == "Find and analyze information"
        assert adapter.backstory == "Expert researcher with years of experience"
        assert adapter.custom_section == "Be thorough."
        assert adapter.enable_execution_reporting is True
        assert adapter.verbose is True

    def test_system_prompt_override(self):
        """Should use custom system_prompt if provided."""
        adapter = CrewAIAdapter(
            system_prompt="You are a custom assistant.",
        )

        assert adapter.system_prompt == "You are a custom assistant."


class TestOnStarted:
    """Tests for on_started() method."""

    @pytest.mark.asyncio
    async def test_renders_system_prompt(self):
        """Should render system prompt from agent metadata."""
        adapter = CrewAIAdapter()

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"
        assert adapter._system_prompt != ""
        assert "TestBot" in adapter._system_prompt

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt_when_provided(self):
        """Should use custom system_prompt instead of rendered one."""
        adapter = CrewAIAdapter(system_prompt="Custom prompt here.")

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter._system_prompt == "Custom prompt here."

    @pytest.mark.asyncio
    async def test_includes_role_goal_backstory_in_prompt(self):
        """Should include role, goal, and backstory in system prompt."""
        adapter = CrewAIAdapter(
            role="Research Analyst",
            goal="Find information",
            backstory="Expert researcher",
        )

        await adapter.on_started(agent_name="TestBot", agent_description="")

        assert "Research Analyst" in adapter._system_prompt
        assert "Find information" in adapter._system_prompt
        assert "Expert researcher" in adapter._system_prompt

    @pytest.mark.asyncio
    async def test_uses_agent_name_as_default_role(self):
        """Should use agent name as role if role not provided."""
        adapter = CrewAIAdapter()

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        # Role section should contain the agent name
        assert "TestBot" in adapter._system_prompt


class TestOnMessage:
    """Tests for on_message() method."""

    @pytest.mark.asyncio
    async def test_initializes_history_on_bootstrap(self, sample_message, mock_tools):
        """Should initialize room history on first message."""
        adapter = CrewAIAdapter()
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
        adapter = CrewAIAdapter()
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
        adapter = CrewAIAdapter()
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
                "[Crew Update]: Alice joined" in str(m.get("content", ""))
                for m in adapter._message_history["room-123"]
            )
            assert found


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_room_history(self, sample_message, mock_tools):
        """Should remove room history on cleanup."""
        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")

        # First add some history
        adapter._message_history["room-123"] = [{"role": "user", "content": "test"}]
        assert "room-123" in adapter._message_history

        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._message_history

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self):
        """Should handle cleanup of non-existent room."""
        adapter = CrewAIAdapter()

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")


class TestBuildMessages:
    """Tests for _build_messages() method."""

    @pytest.mark.asyncio
    async def test_includes_system_prompt(self):
        """Should include system prompt in messages."""
        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")

        adapter._message_history["room-123"] = []

        messages = adapter._build_messages("room-123")

        assert len(messages) >= 1
        assert messages[0]["role"] == "system"
        assert "TestBot" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_includes_conversation_history(self):
        """Should include conversation history in messages."""
        adapter = CrewAIAdapter()
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
        adapter = CrewAIAdapter(enable_execution_reporting=True)

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
        adapter = CrewAIAdapter()

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
        adapter = CrewAIAdapter()

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
        adapter = CrewAIAdapter()
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
        adapter = CrewAIAdapter()
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
        adapter = CrewAIAdapter()
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


class TestVerboseMode:
    """Tests for verbose mode."""

    @pytest.mark.asyncio
    async def test_verbose_mode_logs_iterations(self, sample_message, mock_tools):
        """Verbose mode should enable detailed logging."""
        adapter = CrewAIAdapter(verbose=True)
        await adapter.on_started("TestBot", "Test bot")

        with patch.object(adapter, "_call_llm") as mock_call:
            mock_call.return_value = {"content": "Done!", "tool_calls": []}

            # Should not raise - verbose just enables more logging
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )


class EchoInput(BaseModel):
    """Echo back the provided message."""

    message: str = Field(description="Message to echo")


class CalculatorInput(BaseModel):
    """Perform math calculations."""

    operation: str = Field(description="add, subtract, multiply, divide")
    left: float
    right: float


async def echo_message(args: EchoInput) -> str:
    """Async echo tool."""
    return f"Echo: {args.message}"


def calculate(args: CalculatorInput) -> str:
    """Sync calculator tool."""
    ops = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b,
    }
    return str(ops[args.operation](args.left, args.right))


async def failing_tool(args: EchoInput) -> str:
    """Tool that always fails."""
    raise ValueError("Service unavailable")


class TestCustomTools:
    """Tests for custom tool support."""

    def test_accepts_additional_tools_parameter(self):
        """Adapter should accept list of (Model, func) tuples."""
        adapter = CrewAIAdapter(
            additional_tools=[(EchoInput, echo_message)],
        )

        assert len(adapter._custom_tools) == 1
        assert adapter._custom_tools[0][0] is EchoInput

    def test_accepts_multiple_custom_tools(self):
        """Adapter should accept multiple custom tools."""
        adapter = CrewAIAdapter(
            additional_tools=[
                (EchoInput, echo_message),
                (CalculatorInput, calculate),
            ],
        )

        assert len(adapter._custom_tools) == 2

    def test_defaults_to_empty_custom_tools(self):
        """Adapter should have empty custom tools by default."""
        adapter = CrewAIAdapter()

        assert adapter._custom_tools == []

    @pytest.mark.asyncio
    async def test_merges_custom_tool_schemas_openai_format(
        self, sample_message, mock_tools
    ):
        """Custom tools should appear in schema list with OpenAI format."""
        adapter = CrewAIAdapter(
            additional_tools=[(EchoInput, echo_message)],
        )
        await adapter.on_started("TestBot", "Test bot")

        # Mock platform tools returning some schemas
        mock_tools.get_openai_tool_schemas = MagicMock(
            return_value=[
                {
                    "type": "function",
                    "function": {"name": "send_message", "description": "Send"},
                }
            ]
        )

        captured_tools = []

        with patch.object(adapter, "_call_llm") as mock_call:

            async def capture_call(messages, tools):
                captured_tools.extend(tools)
                return {"content": "Done!", "tool_calls": []}

            mock_call.side_effect = capture_call

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Should have both platform and custom tool
        assert len(captured_tools) == 2
        # Verify OpenAI format
        echo_tool = next(t for t in captured_tools if t["function"]["name"] == "echo")
        assert echo_tool["type"] == "function"
        assert "parameters" in echo_tool["function"]

    @pytest.mark.asyncio
    async def test_routes_to_custom_tool(self, mock_tools):
        """Tool call for custom tool should execute custom function."""
        adapter = CrewAIAdapter(
            additional_tools=[(EchoInput, echo_message)],
        )

        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": '{"message": "Hello world"}',
                },
            }
        ]

        results = await adapter._process_tool_calls(tool_calls, mock_tools)

        # Should NOT have called platform execute_tool_call
        mock_tools.execute_tool_call.assert_not_called()

        # Should have result from custom tool
        assert len(results) == 1
        assert "Echo: Hello world" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_routes_to_platform_tool(self, mock_tools):
        """Tool call for platform tool should use execute_tool_call."""
        adapter = CrewAIAdapter(
            additional_tools=[(EchoInput, echo_message)],
        )

        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "send_message",
                    "arguments": '{"content": "Hello", "mentions": ["User"]}',
                },
            }
        ]

        mock_tools.execute_tool_call.return_value = {"status": "sent"}

        results = await adapter._process_tool_calls(tool_calls, mock_tools)

        # Should have called platform execute_tool_call
        mock_tools.execute_tool_call.assert_called_once()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_custom_tool_error_sets_error_content(self, mock_tools):
        """Custom tool exception should result in error content."""
        adapter = CrewAIAdapter(
            additional_tools=[(EchoInput, failing_tool)],
        )

        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": '{"message": "test"}',
                },
            }
        ]

        results = await adapter._process_tool_calls(tool_calls, mock_tools)

        assert len(results) == 1
        assert "Service unavailable" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_preserves_tool_call_id_on_error(self, mock_tools):
        """tool_call_id should be preserved even when custom tool fails."""
        adapter = CrewAIAdapter(
            additional_tools=[(EchoInput, failing_tool)],
        )

        tool_calls = [
            {
                "id": "call-abc-123",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": '{"message": "test"}',
                },
            }
        ]

        results = await adapter._process_tool_calls(tool_calls, mock_tools)

        assert results[0]["tool_call_id"] == "call-abc-123"

    @pytest.mark.asyncio
    async def test_multiple_custom_tools_execution(self, mock_tools):
        """Multiple custom tools should be callable."""
        adapter = CrewAIAdapter(
            additional_tools=[
                (EchoInput, echo_message),
                (CalculatorInput, calculate),
            ],
        )

        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "echo",
                    "arguments": '{"message": "Hello"}',
                },
            },
            {
                "id": "call-2",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"operation": "add", "left": 5, "right": 3}',
                },
            },
        ]

        results = await adapter._process_tool_calls(tool_calls, mock_tools)

        assert len(results) == 2
        assert "Echo: Hello" in results[0]["content"]
        assert "8.0" in results[1]["content"]
