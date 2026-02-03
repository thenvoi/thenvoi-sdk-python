"""Tests for ClaudeCodeDesktopAdapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

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

    @pytest.mark.asyncio
    async def test_builds_correct_cli_command(self):
        """Should build correct CLI command with all flags."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")

        cmd = adapter._build_cli_command("Test prompt", session_id=None)

        assert cmd[0] == "/usr/bin/claude"
        assert "--print" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--no-session-persistence" in cmd

    @pytest.mark.asyncio
    async def test_builds_command_with_session_resume(self):
        """Should build command with session resume when session_id exists."""
        adapter = ClaudeCodeDesktopAdapter(cli_path="/usr/bin/claude")

        cmd = adapter._build_cli_command("Test prompt", session_id="session-123")

        assert "--resume" in cmd
        assert "session-123" in cmd
        assert "--no-session-persistence" not in cmd

    @pytest.mark.asyncio
    async def test_parses_json_response(self):
        """Should parse JSON response from CLI."""
        adapter = ClaudeCodeDesktopAdapter()

        json_output = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "Hello! I can help you.",
                "session_id": "new-session-123",
                "total_cost_usd": 0.024,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        )

        result = adapter._parse_cli_response(json_output)

        assert result["result"] == "Hello! I can help you."
        assert result["session_id"] == "new-session-123"
        assert result["total_cost_usd"] == 0.024

    @pytest.mark.asyncio
    async def test_handles_error_response(self):
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
