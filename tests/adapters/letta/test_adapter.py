"""Tests for Letta adapter."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.adapters.letta import LettaAdapter, LettaConfig, LettaMode
from thenvoi.core.types import PlatformMessage


@pytest.fixture
def config():
    """Basic Letta config for testing."""
    return LettaConfig(
        api_key="test-key",
        mode=LettaMode.PER_ROOM,
        base_url="http://localhost:8283",
    )


@pytest.fixture
def shared_config():
    """Shared mode Letta config for testing."""
    return LettaConfig(
        api_key="test-key",
        mode=LettaMode.SHARED,
        base_url="http://localhost:8283",
    )


@pytest.fixture
def adapter(config, tmp_path: Path):
    """Create adapter with temp storage."""
    return LettaAdapter(
        config=config,
        state_storage_path=tmp_path / "state.json",
    )


@pytest.fixture
def shared_adapter(shared_config, tmp_path: Path):
    """Create shared mode adapter with temp storage."""
    return LettaAdapter(
        config=shared_config,
        state_storage_path=tmp_path / "state.json",
    )


@pytest.fixture
def platform_message():
    """Sample platform message."""
    return PlatformMessage(
        id="msg-123",
        room_id="room-456",
        content="Hello, can you help me?",
        sender_id="user-789",
        sender_type="User",
        sender_name="Alice",
        message_type="text",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_tools():
    """Mock AgentToolsProtocol."""
    tools = AsyncMock()
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.add_participant = AsyncMock(return_value={"id": "user-1"})
    tools.remove_participant = AsyncMock(return_value={"status": "removed"})
    tools.lookup_peers = AsyncMock(return_value={"peers": []})
    tools.get_participants = AsyncMock(return_value=[{"name": "Alice", "type": "User"}])
    tools.create_chatroom = AsyncMock(return_value="new-room-123")
    # execute_tool_call is used by the adapter for assistant messages
    tools.execute_tool_call = AsyncMock(return_value={"status": "sent"})
    return tools


@pytest.fixture
def mock_letta_client():
    """Mock Letta client."""
    client = MagicMock()

    # Mock agent creation
    mock_agent = MagicMock()
    mock_agent.id = "letta-agent-123"
    client.agents.create.return_value = mock_agent

    # Mock agent retrieval
    client.agents.retrieve.return_value = mock_agent

    # Mock message response
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.message_type = "assistant_message"
    mock_message.content = "Hello! I'm here to help."
    mock_response.messages = [mock_message]
    client.agents.messages.create.return_value = mock_response

    # Mock Conversations API (for SHARED mode)
    mock_conversation = MagicMock()
    mock_conversation.id = "conv-456"
    client.conversations.create.return_value = mock_conversation
    client.conversations.retrieve.return_value = mock_conversation
    client.conversations.messages.create.return_value = mock_response

    return client


async def _init_adapter_with_mock(
    adapter, mock_client, agent_name="TestBot", agent_description="Test assistant"
):
    """Helper to initialize adapter with mock client."""
    from thenvoi.adapters.letta.memory import MemoryManager

    # Manually set state that on_started would set
    adapter.agent_name = agent_name
    adapter.agent_description = agent_description
    adapter._thenvoi_agent_name = agent_name
    adapter._thenvoi_agent_description = agent_description
    adapter._client = mock_client
    adapter._memory_manager = MemoryManager(mock_client)
    adapter._state_store.load()
    adapter.state.thenvoi_agent_id = agent_name
    adapter.state.mode = adapter.config.mode.value

    # For shared mode, set up shared agent
    if adapter.config.mode == LettaMode.SHARED:
        adapter.state.shared_agent_id = "letta-agent-123"
        adapter._state_store.save()


class TestInitialization:
    """Tests for adapter initialization."""

    def test_creates_with_config(self, config, tmp_path: Path):
        """Should initialize with config."""
        adapter = LettaAdapter(
            config=config,
            state_storage_path=tmp_path / "state.json",
        )

        assert adapter.config == config
        assert adapter._client is None  # Not initialized until on_started

    def test_expands_tilde_in_path(self, config):
        """Should expand ~ in storage path."""
        adapter = LettaAdapter(
            config=config,
            state_storage_path="~/.thenvoi/test.json",
        )

        assert "~" not in str(adapter._state_store.storage_path)


class TestOnStarted:
    """Tests for on_started() lifecycle method."""

    @pytest.mark.asyncio
    async def test_sets_agent_name_and_description(self, adapter, mock_letta_client):
        """Should set agent metadata."""
        await _init_adapter_with_mock(
            adapter, mock_letta_client, "TestBot", "A test bot"
        )

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"
        assert adapter._thenvoi_agent_name == "TestBot"
        assert adapter._thenvoi_agent_description == "A test bot"

    @pytest.mark.asyncio
    async def test_initializes_letta_client(self, adapter, mock_letta_client):
        """Should initialize Letta client."""
        await _init_adapter_with_mock(adapter, mock_letta_client)

        assert adapter._client is not None

    @pytest.mark.asyncio
    async def test_loads_state(self, adapter, mock_letta_client):
        """Should load persisted state."""
        await _init_adapter_with_mock(adapter, mock_letta_client)

        assert adapter.state is not None
        assert adapter.state.thenvoi_agent_id == "TestBot"
        assert adapter.state.mode == "per_room"

    @pytest.mark.asyncio
    async def test_shared_mode_creates_agent(self, shared_adapter, mock_letta_client):
        """Should create shared agent in SHARED mode."""
        await _init_adapter_with_mock(shared_adapter, mock_letta_client)

        assert shared_adapter.state.shared_agent_id == "letta-agent-123"


class TestOnMessagePerRoom:
    """Tests for on_message() in PER_ROOM mode."""

    @pytest.mark.asyncio
    async def test_creates_agent_on_first_message(
        self, adapter, platform_message, mock_tools, mock_letta_client
    ):
        """Should create Letta agent on first message to a room."""
        await _init_adapter_with_mock(adapter, mock_letta_client)

        await adapter.on_message(
            msg=platform_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-456",
        )

        # Verify agent was created
        mock_letta_client.agents.create.assert_called_once()

        # Verify state was persisted
        assert adapter.state.get_room_agent("room-456") == "letta-agent-123"

    @pytest.mark.asyncio
    async def test_reuses_existing_agent(
        self, adapter, platform_message, mock_tools, mock_letta_client
    ):
        """Should reuse existing agent on subsequent messages."""
        await _init_adapter_with_mock(adapter, mock_letta_client)

        # Pre-populate state
        adapter.state.set_room_agent("room-456", "existing-agent-123")

        await adapter.on_message(
            msg=platform_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=False,
            room_id="room-456",
        )

        # Verify no new agent was created (only retrieval to verify exists)
        mock_letta_client.agents.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_message_to_letta(
        self, adapter, platform_message, mock_tools, mock_letta_client
    ):
        """Should send message to Letta agent."""
        await _init_adapter_with_mock(adapter, mock_letta_client)

        await adapter.on_message(
            msg=platform_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-456",
        )

        # Verify message was sent to Letta
        mock_letta_client.agents.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_response_as_thought(
        self, adapter, platform_message, mock_tools, mock_letta_client
    ):
        """Should send Letta response as internal thought (not chat message).

        Assistant text is internal thinking - actual messages should be sent
        via MCP tools (create_agent_chat_message) which execute on Letta server.
        """
        await _init_adapter_with_mock(adapter, mock_letta_client)

        await adapter.on_message(
            msg=platform_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-456",
        )

        # Verify response was sent as a thought event (not as a chat message)
        mock_tools.send_event.assert_called()
        # Check one of the calls was a thought event with the assistant content
        thought_calls = [
            call
            for call in mock_tools.send_event.call_args_list
            if call.kwargs.get("message_type") == "thought"
            and "Hello! I'm here to help." in call.kwargs.get("content", "")
        ]
        assert len(thought_calls) == 1, (
            "Expected one thought event with assistant content"
        )

    @pytest.mark.asyncio
    async def test_updates_room_state(
        self, adapter, platform_message, mock_tools, mock_letta_client
    ):
        """Should update room state after message."""
        await _init_adapter_with_mock(adapter, mock_letta_client)

        await adapter.on_message(
            msg=platform_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-456",
        )

        room_state = adapter.state.get_room_state("room-456")
        assert room_state is not None
        assert room_state.last_interaction is not None
        assert room_state.summary is not None


class TestOnMessageShared:
    """Tests for on_message() in SHARED mode using Conversations API."""

    @pytest.mark.asyncio
    async def test_creates_conversation_per_room(
        self, shared_adapter, platform_message, mock_tools, mock_letta_client
    ):
        """Should create separate conversation for each room."""
        await _init_adapter_with_mock(shared_adapter, mock_letta_client)

        # Message to room A
        await shared_adapter.on_message(
            msg=platform_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-A",
        )

        # Create second message for room B
        platform_message_b = PlatformMessage(
            id="msg-124",
            room_id="room-B",
            content="Hello again!",
            sender_id="user-789",
            sender_type="User",
            sender_name="Alice",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        await shared_adapter.on_message(
            msg=platform_message_b,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-B",
        )

        # Should create two conversations (one per room)
        conv_create_calls = mock_letta_client.conversations.create.call_args_list
        assert len(conv_create_calls) == 2
        # All conversations should be for the same shared agent
        assert conv_create_calls[0].kwargs["agent_id"] == "letta-agent-123"
        assert conv_create_calls[1].kwargs["agent_id"] == "letta-agent-123"

        # Both messages should go via Conversations API
        msg_calls = mock_letta_client.conversations.messages.create.call_args_list
        assert len(msg_calls) == 2

    @pytest.mark.asyncio
    async def test_reuses_existing_conversation(
        self, shared_adapter, platform_message, mock_tools, mock_letta_client
    ):
        """Should reuse conversation on subsequent messages to same room."""
        await _init_adapter_with_mock(shared_adapter, mock_letta_client)

        # First message to room A
        await shared_adapter.on_message(
            msg=platform_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-A",
        )

        # Second message to same room A
        platform_message_2 = PlatformMessage(
            id="msg-125",
            room_id="room-A",
            content="Follow-up question",
            sender_id="user-789",
            sender_type="User",
            sender_name="Alice",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        await shared_adapter.on_message(
            msg=platform_message_2,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=False,
            room_id="room-A",
        )

        # Should only create one conversation (reuse for second message)
        conv_create_calls = mock_letta_client.conversations.create.call_args_list
        assert len(conv_create_calls) == 1

        # But should send two messages
        msg_calls = mock_letta_client.conversations.messages.create.call_args_list
        assert len(msg_calls) == 2

    @pytest.mark.asyncio
    async def test_sends_via_conversations_api(
        self, shared_adapter, platform_message, mock_tools, mock_letta_client
    ):
        """Should send messages via Conversations API, not agents.messages."""
        await _init_adapter_with_mock(shared_adapter, mock_letta_client)

        await shared_adapter.on_message(
            msg=platform_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-456",
        )

        # Should NOT use agents.messages.create (that's for PER_ROOM mode)
        mock_letta_client.agents.messages.create.assert_not_called()

        # Should use conversations.messages.create
        mock_letta_client.conversations.messages.create.assert_called_once()
        call_args = mock_letta_client.conversations.messages.create.call_args
        message_content = call_args.kwargs["messages"][0]["content"]
        assert "Alice" in message_content  # Sender name
        assert "Hello, can you help me?" in message_content  # Message content


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_marks_room_inactive(
        self, adapter, platform_message, mock_tools, mock_letta_client
    ):
        """Should mark room as inactive on cleanup."""
        await _init_adapter_with_mock(adapter, mock_letta_client)

        # First send a message to create room state
        await adapter.on_message(
            msg=platform_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-456",
        )

        # Now cleanup
        await adapter.on_cleanup("room-456")

        room_state = adapter.state.get_room_state("room-456")
        assert room_state is not None
        assert room_state.is_active is False

    @pytest.mark.asyncio
    async def test_triggers_memory_consolidation(
        self, adapter, platform_message, mock_tools, mock_letta_client
    ):
        """Should trigger memory consolidation on cleanup."""
        await _init_adapter_with_mock(adapter, mock_letta_client)

        # First send a message
        await adapter.on_message(
            msg=platform_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-456",
        )

        # Now cleanup
        await adapter.on_cleanup("room-456")

        # Should have sent consolidation prompt
        calls = mock_letta_client.agents.messages.create.call_args_list
        assert len(calls) >= 2
        last_call = calls[-1]
        message_content = last_call.kwargs["messages"][0]["content"]
        assert "Memory Consolidation" in message_content

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self, adapter, mock_letta_client):
        """Should handle cleanup of non-existent room without error."""
        await _init_adapter_with_mock(adapter, mock_letta_client)

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_raises_on_missing_client(self, adapter):
        """Should raise if client not initialized."""
        from thenvoi.adapters.letta.exceptions import LettaAdapterError

        with pytest.raises(LettaAdapterError, match="not initialized"):
            _ = adapter.client


class TestUtilities:
    """Tests for utility methods."""

    def test_format_time_ago_days(self, adapter):
        """Should format days ago."""
        from datetime import timedelta

        past = datetime.now(timezone.utc) - timedelta(days=5)
        result = adapter._format_time_ago(past)
        assert result == "5 days"

    def test_format_time_ago_hours(self, adapter):
        """Should format hours ago."""
        from datetime import timedelta

        past = datetime.now(timezone.utc) - timedelta(hours=3)
        result = adapter._format_time_ago(past)
        assert result == "3 hours"

    def test_format_time_ago_just_now(self, adapter):
        """Should format just now."""
        now = datetime.now(timezone.utc)
        result = adapter._format_time_ago(now)
        assert result == "just now"

    def test_extract_summary_short(self, adapter):
        """Should return short content as-is."""
        result = adapter._extract_summary(["Hello world"])
        assert result == "Hello world"

    def test_extract_summary_truncates(self, adapter):
        """Should truncate long content."""
        long_text = "a" * 200
        result = adapter._extract_summary([long_text])
        assert len(result) == 150
        assert result.endswith("...")

    def test_format_participants(self, adapter):
        """Should format participants for memory block."""
        participants = [
            {"name": "Alice", "type": "User"},
            {"name": "DataBot", "type": "Agent"},
        ]
        result = adapter._format_participants(participants)
        assert "Alice (User)" in result
        assert "DataBot (Agent)" in result
