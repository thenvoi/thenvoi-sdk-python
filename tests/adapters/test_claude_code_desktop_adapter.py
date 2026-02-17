"""Tests for ClaudeCodeDesktopAdapter."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.adapters.claude_code_desktop import ClaudeCodeDesktopAdapter
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
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.add_participant = AsyncMock(return_value={"id": "user-1"})
    tools.remove_participant = AsyncMock(return_value={"status": "removed"})
    tools.lookup_peers = AsyncMock(return_value={"peers": []})
    tools.get_participants = AsyncMock(return_value=[])
    tools.create_chatroom = AsyncMock(return_value="new-room-abc")
    return tools


class TestClaudeCodeDesktopInitialization:
    """Tests for adapter initialization."""

    def test_default_initialization(self):
        """Should initialize with default values."""
        adapter = ClaudeCodeDesktopAdapter()

        assert adapter.custom_section is None
        assert adapter.cli_path is None  # Auto-detect from PATH or env var
        assert adapter.cli_timeout == 120000  # 2 minutes default
        assert adapter._session_ids == {}
        assert adapter.allowed_tools == []

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        adapter = ClaudeCodeDesktopAdapter(
            custom_section="Be helpful.",
            cli_path="/custom/path/to/claude",
            cli_timeout=60000,
        )

        assert adapter.custom_section == "Be helpful."
        assert adapter.cli_path == "/custom/path/to/claude"
        assert adapter.cli_timeout == 60000

    def test_initialization_with_allowed_tools(self):
        """Should accept allowed_tools parameter."""
        adapter = ClaudeCodeDesktopAdapter(
            allowed_tools=["Read", "Write", "Edit"],
        )

        assert adapter.allowed_tools == ["Read", "Write", "Edit"]

    def test_allowed_tools_none_defaults_to_empty_list(self):
        """Should default allowed_tools to empty list when None."""
        adapter = ClaudeCodeDesktopAdapter(allowed_tools=None)

        assert adapter.allowed_tools == []


class TestClaudeCodeDesktopCLIDetection:
    """Tests for CLI detection."""

    def test_find_cli_with_explicit_path(self):
        """Should use explicit CLI path when provided."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/explicit/path/claude")

        assert adapter._get_cli_path() == "/explicit/path/claude"

    def test_find_cli_from_env_var(self):
        """Should use CLAUDE_CODE_PATH env var when set."""
        adapter = ClaudeCodeDesktopAdapter()

        with patch.dict("os.environ", {"CLAUDE_CODE_PATH": "/env/path/claude"}):
            assert adapter._get_cli_path() == "/env/path/claude"

    def test_find_cli_in_path(self):
        """Should find claude in PATH."""
        adapter = ClaudeCodeDesktopAdapter()

        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            assert adapter._get_cli_path() == "/usr/local/bin/claude"

    def test_raise_error_when_cli_not_found(self):
        """Should raise error when CLI not found."""
        adapter = ClaudeCodeDesktopAdapter()

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("shutil.which", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="Claude Code CLI not found"):
                adapter._get_cli_path()


class TestClaudeCodeDesktopOnStarted:
    """Tests for on_started() method."""

    @pytest.mark.asyncio
    async def test_stores_agent_info(self):
        """Should store agent name and description on start."""
        adapter = ClaudeCodeDesktopAdapter()

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"


class TestClaudeCodeDesktopSessionManagement:
    """Tests for session ID management."""

    def test_stores_session_ids_per_room(self):
        """Should store session IDs per room."""
        adapter = ClaudeCodeDesktopAdapter()

        adapter._session_ids["room-1"] = "session-abc"
        adapter._session_ids["room-2"] = "session-def"

        assert adapter._session_ids["room-1"] == "session-abc"
        assert adapter._session_ids["room-2"] == "session-def"


class TestClaudeCodeDesktopOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_session_id(self):
        """Should cleanup session ID when room is cleaned up."""
        adapter = ClaudeCodeDesktopAdapter()
        adapter._session_ids["room-123"] = "session-abc"

        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._session_ids

    @pytest.mark.asyncio
    async def test_cleanup_non_existent_room_is_safe(self):
        """Should handle cleanup for non-existent room safely."""
        adapter = ClaudeCodeDesktopAdapter()

        # Should not raise
        await adapter.on_cleanup("non-existent-room")


class TestClaudeCodeDesktopCLIInvocation:
    """Tests for CLI invocation."""

    def test_builds_correct_cli_command(self):
        """Should build correct CLI command with stream-json flags."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")

        cmd = adapter._build_cli_command(session_id=None)

        assert cmd[0] == "/usr/bin/claude"
        assert "--print" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        # No --allowedTools when no tools configured
        assert "--allowedTools" not in cmd

    def test_builds_command_with_session_resume(self):
        """Should build command with session resume when session_id exists."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")

        cmd = adapter._build_cli_command(session_id="session-123")

        assert "--resume" in cmd
        assert "session-123" in cmd

    def test_builds_command_with_allowed_tools(self):
        """Should include --allowedTools flag when tools are configured."""
        adapter = ClaudeCodeDesktopAdapter(
            cli_path="/usr/bin/claude",
            allowed_tools=["Read", "Write", "Edit"],
        )

        cmd = adapter._build_cli_command(session_id=None)

        assert "--allowedTools" in cmd
        tools_idx = cmd.index("--allowedTools")
        assert cmd[tools_idx + 1] == "Read,Write,Edit"

    def test_builds_command_no_allowed_tools_when_empty(self):
        """Should not include --allowedTools when no tools configured."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")

        cmd = adapter._build_cli_command(session_id=None)

        assert "--allowedTools" not in cmd

    def test_builds_command_always_includes_verbose(self):
        """--verbose is always included (required by --output-format stream-json)."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")

        cmd = adapter._build_cli_command(session_id=None)

        assert "--verbose" in cmd

    def test_parses_stream_json_response(self):
        """Should parse NDJSON stream-json response from CLI."""
        adapter = ClaudeCodeDesktopAdapter()

        # Simulate stream-json NDJSON output (multiple lines)
        lines = [
            json.dumps({"type": "system", "subtype": "init", "session_id": "s-1"}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "I'll help you."}],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "Hello! I can help you.",
                    "session_id": "new-session-123",
                    "total_cost_usd": 0.024,
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                }
            ),
        ]
        ndjson_output = "\n".join(lines)

        result = adapter._parse_cli_response(ndjson_output)

        assert result["result"] == "Hello! I can help you."
        assert result["session_id"] == "new-session-123"
        assert result["total_cost_usd"] == 0.024

    def test_parses_stream_json_with_tool_calls(self):
        """Should capture tool calls from assistant messages in NDJSON."""
        adapter = ClaudeCodeDesktopAdapter()

        lines = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "Let me read that file."},
                            {
                                "type": "tool_use",
                                "name": "Read",
                                "input": {"file_path": "/tmp/test.md"},
                            },
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "File contents here.",
                    "session_id": "s-456",
                }
            ),
        ]
        ndjson_output = "\n".join(lines)

        result = adapter._parse_cli_response(ndjson_output)

        assert result["result"] == "File contents here."
        assert result["session_id"] == "s-456"

    def test_parses_single_json_fallback(self):
        """Should fall back to single JSON parsing for backward compat."""
        adapter = ClaudeCodeDesktopAdapter()

        json_output = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "Hello! I can help you.",
                "session_id": "new-session-123",
                "total_cost_usd": 0.024,
            }
        )

        result = adapter._parse_cli_response(json_output)

        assert result["result"] == "Hello! I can help you."
        assert result["session_id"] == "new-session-123"

    def test_handles_error_response(self):
        """Should handle error response from CLI."""
        adapter = ClaudeCodeDesktopAdapter()

        json_output = json.dumps(
            {
                "type": "result",
                "subtype": "error_response",
                "result": "An error occurred",
                "is_error": True,
            }
        )

        result = adapter._parse_cli_response(json_output)

        assert result["is_error"] is True
        assert result["result"] == "An error occurred"

    def test_handles_unparseable_output(self):
        """Should return error when output is not valid JSON or NDJSON."""
        adapter = ClaudeCodeDesktopAdapter()

        result = adapter._parse_cli_response("not json at all")

        assert result["is_error"] is True
        assert result["result"] == "not json at all"

    def test_handles_ndjson_with_blank_lines(self):
        """Should skip blank lines in NDJSON output."""
        adapter = ClaudeCodeDesktopAdapter()

        lines = [
            "",
            json.dumps({"type": "system", "subtype": "init"}),
            "",
            json.dumps(
                {
                    "type": "result",
                    "result": "Done.",
                    "session_id": "s-789",
                }
            ),
            "",
        ]
        ndjson_output = "\n".join(lines)

        result = adapter._parse_cli_response(ndjson_output)

        assert result["result"] == "Done."
        assert result["session_id"] == "s-789"


class TestClaudeCodeDesktopPromptGeneration:
    """Tests for prompt generation."""

    def test_generates_prompt_with_room_context(self):
        """Should generate prompt with room_id context."""
        adapter = ClaudeCodeDesktopAdapter()
        adapter.agent_name = "TestBot"
        adapter.agent_description = "A test bot"

        prompt = adapter._generate_prompt(
            room_id="room-123",
            message="Hello!",
            history="",
            participants_msg=None,
        )

        assert "room_id: room-123" in prompt
        assert "Hello!" in prompt
        assert "TestBot" in prompt

    def test_includes_custom_section(self):
        """Should include custom section in prompt."""
        adapter = ClaudeCodeDesktopAdapter(custom_section="Always be polite.")
        adapter.agent_name = "TestBot"
        adapter.agent_description = "A test bot"

        prompt = adapter._generate_prompt(
            room_id="room-123",
            message="Hello!",
            history="",
            participants_msg=None,
        )

        assert "Always be polite." in prompt

    def test_includes_history(self):
        """Should include history in prompt."""
        adapter = ClaudeCodeDesktopAdapter()
        adapter.agent_name = "TestBot"
        adapter.agent_description = "A test bot"

        prompt = adapter._generate_prompt(
            room_id="room-123",
            message="Hello!",
            history="[Alice]: Previous message",
            participants_msg=None,
        )

        assert "[Alice]: Previous message" in prompt

    def test_includes_participants_message(self):
        """Should include participants update message."""
        adapter = ClaudeCodeDesktopAdapter()
        adapter.agent_name = "TestBot"
        adapter.agent_description = "A test bot"

        prompt = adapter._generate_prompt(
            room_id="room-123",
            message="Hello!",
            history="",
            participants_msg="Bob has joined the chat",
        )

        assert "Bob has joined the chat" in prompt


class TestClaudeCodeDesktopCreateChatroom:
    """Tests for create_chatroom action."""

    @pytest.mark.asyncio
    async def test_execute_create_chatroom_action(self, mock_tools):
        """Should execute create_chatroom action via tools."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {"action": "create_chatroom", "task_id": "task-123"}
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.create_chatroom.assert_called_once_with("task-123")

    @pytest.mark.asyncio
    async def test_execute_create_chatroom_without_task_id(self, mock_tools):
        """Should pass None when task_id is not provided."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {"action": "create_chatroom"}
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.create_chatroom.assert_called_once_with(None)


class TestClaudeCodeDesktopExecuteAction:
    """Tests for _execute_action with all action types."""

    @pytest.mark.asyncio
    async def test_execute_send_message(self, mock_tools):
        """Should call tools.send_message with content and mentions."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {
            "action": "send_message",
            "content": "Hello!",
            "mentions": ["user-1"],
        }
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.send_message.assert_called_once_with("Hello!", ["user-1"])

    @pytest.mark.asyncio
    async def test_execute_send_message_defaults(self, mock_tools):
        """Should default to empty content and mentions when not provided."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {"action": "send_message"}
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.send_message.assert_called_once_with("", [])

    @pytest.mark.asyncio
    async def test_execute_send_event(self, mock_tools):
        """Should call tools.send_event with content and message_type."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {
            "action": "send_event",
            "content": "Processing...",
            "message_type": "thought",
        }
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.send_event.assert_called_once_with("Processing...", "thought")

    @pytest.mark.asyncio
    async def test_execute_send_event_defaults(self, mock_tools):
        """Should default message_type to 'thought'."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {"action": "send_event", "content": "Thinking"}
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.send_event.assert_called_once_with("Thinking", "thought")

    @pytest.mark.asyncio
    async def test_execute_add_participant(self, mock_tools):
        """Should call tools.add_participant with name and role."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {
            "action": "add_participant",
            "name": "Weather Agent",
            "role": "admin",
        }
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.add_participant.assert_called_once_with("Weather Agent", "admin")

    @pytest.mark.asyncio
    async def test_execute_add_participant_defaults(self, mock_tools):
        """Should default role to 'member'."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {"action": "add_participant", "name": "Bot"}
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.add_participant.assert_called_once_with("Bot", "member")

    @pytest.mark.asyncio
    async def test_execute_remove_participant(self, mock_tools):
        """Should call tools.remove_participant with name."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {"action": "remove_participant", "name": "Weather Agent"}
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.remove_participant.assert_called_once_with("Weather Agent")

    @pytest.mark.asyncio
    async def test_execute_get_participants(self, mock_tools):
        """Should call tools.get_participants."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {"action": "get_participants"}
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.get_participants.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_lookup_peers(self, mock_tools):
        """Should call tools.lookup_peers with page and page_size."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {"action": "lookup_peers", "page": 2, "page_size": 25}
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.lookup_peers.assert_called_once_with(2, 25)

    @pytest.mark.asyncio
    async def test_execute_lookup_peers_defaults(self, mock_tools):
        """Should default page=1 and page_size=50."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {"action": "lookup_peers"}
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.lookup_peers.assert_called_once_with(1, 50)

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, mock_tools):
        """Should send error event for unknown action."""
        adapter = ClaudeCodeDesktopAdapter()

        action_data = {"action": "do_something_weird"}
        await adapter._execute_action(action_data, mock_tools)

        mock_tools.send_event.assert_called_once()
        call_args = mock_tools.send_event.call_args
        assert "Unknown action type: do_something_weird" in call_args[1]["content"]
        assert call_args[1]["message_type"] == "error"

    @pytest.mark.asyncio
    async def test_execute_action_handles_exception(self, mock_tools):
        """Should send error event when action execution raises."""
        adapter = ClaudeCodeDesktopAdapter()
        mock_tools.send_message = AsyncMock(side_effect=RuntimeError("connection lost"))

        action_data = {"action": "send_message", "content": "test"}
        await adapter._execute_action(action_data, mock_tools)

        # send_event should be called with the error
        mock_tools.send_event.assert_called_once()
        call_args = mock_tools.send_event.call_args
        assert "connection lost" in call_args[1]["content"]
        assert call_args[1]["message_type"] == "error"


class TestClaudeCodeDesktopExecuteCLI:
    """Tests for _execute_cli subprocess management."""

    @pytest.mark.asyncio
    async def test_execute_cli_success(self):
        """Should execute CLI and return parsed response."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")

        ndjson_output = json.dumps(
            {
                "type": "result",
                "result": "Hello!",
                "session_id": "s-1",
            }
        )

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(ndjson_output.encode(), b""))
        mock_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            result = await adapter._execute_cli("test prompt", None)

        assert result["result"] == "Hello!"
        assert result["session_id"] == "s-1"

    @pytest.mark.asyncio
    async def test_execute_cli_nonzero_exit(self):
        """Should return error dict when CLI exits with non-zero code."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Error: model not found")
        )
        mock_process.returncode = 1

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            result = await adapter._execute_cli("test prompt", None)

        assert result["is_error"] is True
        assert "model not found" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_cli_timeout(self):
        """Should return error and kill process on timeout."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude", cli_timeout=1000)

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        with (
            patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_process,
            ),
            patch("asyncio.wait_for", side_effect=asyncio.TimeoutError),
        ):
            result = await adapter._execute_cli("test prompt", None)

        assert result["is_error"] is True
        assert "timed out" in result["result"]
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_cli_general_exception(self):
        """Should return error on unexpected exception."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("command not found"),
        ):
            result = await adapter._execute_cli("test prompt", None)

        assert result["is_error"] is True
        assert "command not found" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_cli_passes_prompt_via_stdin(self):
        """Should send prompt to CLI via stdin."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")

        ndjson_output = json.dumps({"type": "result", "result": "ok"})
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(ndjson_output.encode(), b""))
        mock_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            await adapter._execute_cli("my test prompt", None)

        mock_process.communicate.assert_called_once_with(input=b"my test prompt")


class TestClaudeCodeDesktopProcessResponse:
    """Tests for _process_response."""

    @pytest.mark.asyncio
    async def test_process_error_response(self, mock_tools):
        """Should send error event for is_error responses."""
        adapter = ClaudeCodeDesktopAdapter()

        response = {"result": "something broke", "is_error": True}
        await adapter._process_response(response, mock_tools, "room-1")

        mock_tools.send_event.assert_called_once()
        call_args = mock_tools.send_event.call_args
        assert "something broke" in call_args[1]["content"]

    @pytest.mark.asyncio
    async def test_process_json_action_response(self, mock_tools):
        """Should extract and execute JSON action from result."""
        adapter = ClaudeCodeDesktopAdapter()

        response = {"result": '{"action": "send_message", "content": "Hi there!"}'}
        await adapter._process_response(response, mock_tools, "room-1")

        mock_tools.send_message.assert_called_once_with("Hi there!", [])

    @pytest.mark.asyncio
    async def test_process_plain_text_response(self, mock_tools):
        """Should send plain text as message when no JSON action found."""
        adapter = ClaudeCodeDesktopAdapter()

        response = {"result": "Just a plain response."}
        await adapter._process_response(response, mock_tools, "room-1")

        mock_tools.send_message.assert_called_once_with("Just a plain response.", [])

    @pytest.mark.asyncio
    async def test_process_empty_result(self, mock_tools):
        """Should not send anything for empty result."""
        adapter = ClaudeCodeDesktopAdapter()

        response = {"result": ""}
        await adapter._process_response(response, mock_tools, "room-1")

        mock_tools.send_message.assert_not_called()
        mock_tools.send_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_json_in_code_block(self, mock_tools):
        """Should extract JSON action from markdown code block."""
        adapter = ClaudeCodeDesktopAdapter()

        response = {
            "result": 'Here is my action:\n```json\n{"action": "send_message", "content": "Hello!"}\n```'
        }
        await adapter._process_response(response, mock_tools, "room-1")

        mock_tools.send_message.assert_called_once_with("Hello!", [])


class TestClaudeCodeDesktopOnMessage:
    """Tests for on_message end-to-end flow."""

    @pytest.mark.asyncio
    async def test_on_message_bootstrap(self, sample_message, mock_tools):
        """Should use no session_id on bootstrap and capture new session_id."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")
        adapter.agent_name = "TestBot"
        adapter.agent_description = "A test bot"

        cli_response = json.dumps(
            {
                "type": "result",
                "result": '{"action": "send_message", "content": "Hi!"}',
                "session_id": "new-session-abc",
                "total_cost_usd": 0.01,
            }
        )

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(cli_response.encode(), b""))
        mock_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history="previous context",
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Should have captured session_id
        assert adapter._session_ids["room-123"] == "new-session-abc"
        # Should have sent the message
        mock_tools.send_message.assert_called_once_with("Hi!", [])

    @pytest.mark.asyncio
    async def test_on_message_resume(self, sample_message, mock_tools):
        """Should use stored session_id for --resume on subsequent messages."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")
        adapter.agent_name = "TestBot"
        adapter.agent_description = "A test bot"
        adapter._session_ids["room-123"] = "existing-session"

        cli_response = json.dumps(
            {
                "type": "result",
                "result": "Plain text reply",
                "session_id": "existing-session",
            }
        )

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(cli_response.encode(), b""))
        mock_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_exec:
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history="",
                participants_msg=None,
                is_session_bootstrap=False,
                room_id="room-123",
            )

        # Verify --resume was passed with session id
        call_args = mock_exec.call_args[0]
        assert "--resume" in call_args
        assert "existing-session" in call_args

    @pytest.mark.asyncio
    async def test_on_message_cli_error_sends_event_and_raises(
        self, sample_message, mock_tools
    ):
        """Should send error event and re-raise when CLI fails unexpectedly."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")
        adapter.agent_name = "TestBot"
        adapter.agent_description = "A test bot"

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("cli missing"),
        ):
            # _execute_cli catches OSError and returns is_error dict,
            # so on_message should NOT raise. It sends an error event instead.
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history="",
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        mock_tools.send_event.assert_called_once()


class TestClaudeCodeDesktopCleanupAll:
    """Tests for cleanup_all method."""

    @pytest.mark.asyncio
    async def test_cleanup_all_clears_sessions(self):
        """Should clear all session IDs."""
        adapter = ClaudeCodeDesktopAdapter()
        adapter._session_ids["room-1"] = "s-1"
        adapter._session_ids["room-2"] = "s-2"

        await adapter.cleanup_all()

        assert adapter._session_ids == {}


class TestClaudeCodeDesktopExtractActions:
    """Tests for _extract_actions (multi-action support)."""

    def test_single_json_object(self):
        """Should extract a single JSON action object."""
        adapter = ClaudeCodeDesktopAdapter()

        result = '{"action": "send_message", "content": "Hello!"}'
        actions = adapter._extract_actions(result)

        assert len(actions) == 1
        assert actions[0]["action"] == "send_message"
        assert actions[0]["content"] == "Hello!"

    def test_json_array_of_actions(self):
        """Should extract multiple actions from a JSON array."""
        adapter = ClaudeCodeDesktopAdapter()

        result = json.dumps(
            [
                {
                    "action": "send_event",
                    "content": "Thinking...",
                    "message_type": "thought",
                },
                {"action": "send_message", "content": "Here is my answer."},
            ]
        )
        actions = adapter._extract_actions(result)

        assert len(actions) == 2
        assert actions[0]["action"] == "send_event"
        assert actions[1]["action"] == "send_message"

    def test_single_code_block(self):
        """Should extract action from a single code block."""
        adapter = ClaudeCodeDesktopAdapter()

        result = 'Here is my response:\n```json\n{"action": "send_message", "content": "Hi!"}\n```'
        actions = adapter._extract_actions(result)

        assert len(actions) == 1
        assert actions[0]["content"] == "Hi!"

    def test_multiple_code_blocks(self):
        """Should extract actions from multiple code blocks."""
        adapter = ClaudeCodeDesktopAdapter()

        result = (
            'First action:\n```json\n{"action": "send_event", "content": "Working...", "message_type": "thought"}\n```\n'
            'Second action:\n```json\n{"action": "send_message", "content": "Done!"}\n```'
        )
        actions = adapter._extract_actions(result)

        assert len(actions) == 2
        assert actions[0]["action"] == "send_event"
        assert actions[1]["action"] == "send_message"

    def test_array_in_code_block(self):
        """Should extract actions from a JSON array inside a code block."""
        adapter = ClaudeCodeDesktopAdapter()

        arr = json.dumps(
            [
                {"action": "add_participant", "name": "Bot", "role": "member"},
                {"action": "send_message", "content": "Bot added!"},
            ]
        )
        result = f"```json\n{arr}\n```"
        actions = adapter._extract_actions(result)

        assert len(actions) == 2
        assert actions[0]["action"] == "add_participant"
        assert actions[1]["action"] == "send_message"

    def test_plain_text_returns_empty(self):
        """Should return empty list for plain text."""
        adapter = ClaudeCodeDesktopAdapter()

        actions = adapter._extract_actions("Just a plain response")

        assert actions == []

    def test_json_without_action_key_returns_empty(self):
        """Should return empty list for JSON without 'action' key."""
        adapter = ClaudeCodeDesktopAdapter()

        actions = adapter._extract_actions('{"message": "no action key"}')

        assert actions == []

    def test_array_filters_non_action_items(self):
        """Should skip array items that lack an 'action' key."""
        adapter = ClaudeCodeDesktopAdapter()

        result = json.dumps(
            [
                {"action": "send_message", "content": "Valid"},
                {"not_action": "invalid"},
                "just a string",
                {"action": "get_participants"},
            ]
        )
        actions = adapter._extract_actions(result)

        assert len(actions) == 2
        assert actions[0]["action"] == "send_message"
        assert actions[1]["action"] == "get_participants"

    def test_invalid_json_in_code_block_skipped(self):
        """Should skip invalid JSON code blocks and continue."""
        adapter = ClaudeCodeDesktopAdapter()

        result = (
            "```json\n{invalid json\n```\n"
            '```json\n{"action": "send_message", "content": "valid"}\n```'
        )
        actions = adapter._extract_actions(result)

        assert len(actions) == 1
        assert actions[0]["content"] == "valid"


class TestClaudeCodeDesktopProcessResponseMultiAction:
    """Tests for _process_response with multiple actions."""

    @pytest.mark.asyncio
    async def test_executes_multiple_actions(self, mock_tools):
        """Should execute all actions from a JSON array result."""
        adapter = ClaudeCodeDesktopAdapter()

        response = {
            "result": json.dumps(
                [
                    {
                        "action": "send_event",
                        "content": "Thinking...",
                        "message_type": "thought",
                    },
                    {"action": "send_message", "content": "Here is my answer."},
                ]
            )
        }
        await adapter._process_response(response, mock_tools, "room-1")

        mock_tools.send_event.assert_called_once_with("Thinking...", "thought")
        mock_tools.send_message.assert_called_once_with("Here is my answer.", [])

    @pytest.mark.asyncio
    async def test_executes_actions_from_multiple_code_blocks(self, mock_tools):
        """Should execute actions from multiple code blocks."""
        adapter = ClaudeCodeDesktopAdapter()

        response = {
            "result": (
                '```json\n{"action": "send_event", "content": "Processing...", "message_type": "thought"}\n```\n'
                '```json\n{"action": "send_message", "content": "Done!"}\n```'
            )
        }
        await adapter._process_response(response, mock_tools, "room-1")

        mock_tools.send_event.assert_called_once_with("Processing...", "thought")
        mock_tools.send_message.assert_called_once_with("Done!", [])


class TestClaudeCodeDesktopSanitizeError:
    """Tests for _sanitize_error."""

    def test_redacts_file_paths(self):
        """Should redact file paths from error messages."""
        adapter = ClaudeCodeDesktopAdapter()

        error = "Failed to read /home/user/.config/secrets.json: permission denied"
        sanitized = adapter._sanitize_error(error)

        assert "/home/user/.config/secrets.json" not in sanitized
        assert "[redacted]" in sanitized
        assert "permission denied" in sanitized

    def test_redacts_api_keys(self):
        """Should redact API keys starting with sk-."""
        adapter = ClaudeCodeDesktopAdapter()

        error = "Authentication failed for key sk-abcdef1234567890"
        sanitized = adapter._sanitize_error(error)

        assert "sk-abcdef1234567890" not in sanitized
        assert "[redacted]" in sanitized

    def test_redacts_token_assignments(self):
        """Should redact token/secret/password assignments."""
        adapter = ClaudeCodeDesktopAdapter()

        error = "Config error: api_token = xyzzy12345 is invalid"
        sanitized = adapter._sanitize_error(error)

        assert "xyzzy12345" not in sanitized
        assert "[redacted]" in sanitized

    def test_truncates_long_messages(self):
        """Should truncate messages exceeding max_length."""
        adapter = ClaudeCodeDesktopAdapter()

        error = "A" * 300
        sanitized = adapter._sanitize_error(error, max_length=100)

        assert len(sanitized) == 103  # 100 + "..."
        assert sanitized.endswith("...")

    def test_preserves_short_safe_messages(self):
        """Should not alter short messages without sensitive data."""
        adapter = ClaudeCodeDesktopAdapter()

        error = "Connection refused"
        sanitized = adapter._sanitize_error(error)

        assert sanitized == "Connection refused"

    def test_process_response_error_is_sanitized(self, mock_tools):
        """Should sanitize error in _process_response."""
        adapter = ClaudeCodeDesktopAdapter()

        response = {
            "result": "CLI error: failed at /usr/local/bin/claude with token=secret123",
            "is_error": True,
        }

        import asyncio

        asyncio.get_event_loop().run_until_complete(
            adapter._process_response(response, mock_tools, "room-1")
        )

        call_args = mock_tools.send_event.call_args
        content = call_args[1]["content"]
        assert "/usr/local/bin/claude" not in content
        assert "secret123" not in content
        assert "[redacted]" in content

    @pytest.mark.asyncio
    async def test_execute_action_error_is_sanitized(self, mock_tools):
        """Should sanitize error in _execute_action exception handler."""
        adapter = ClaudeCodeDesktopAdapter()
        mock_tools.send_message = AsyncMock(
            side_effect=RuntimeError("failed at /tmp/secret with key=abc123")
        )

        action_data = {"action": "send_message", "content": "test"}
        await adapter._execute_action(action_data, mock_tools)

        call_args = mock_tools.send_event.call_args
        content = call_args[1]["content"]
        assert "/tmp/secret" not in content
        assert "[redacted]" in content
