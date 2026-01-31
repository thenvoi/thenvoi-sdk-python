"""Tests for Letta memory manager."""

import pytest
from unittest.mock import MagicMock, patch

from thenvoi.adapters.letta.memory import MemoryBlocks, MemoryManager


class TestMemoryBlocks:
    """Tests for MemoryBlocks constants."""

    def test_has_standard_blocks(self):
        """Should define standard memory block labels."""
        assert MemoryBlocks.PERSONA == "persona"
        assert MemoryBlocks.PARTICIPANTS == "participants"
        assert MemoryBlocks.ROOM_CONTEXTS == "room_contexts"


class TestMemoryManager:
    """Tests for MemoryManager class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Letta client."""
        client = MagicMock()
        client.agents = MagicMock()
        client.agents.blocks = MagicMock()
        return client

    @pytest.fixture
    def manager(self, mock_client):
        """Create memory manager with mock client."""
        return MemoryManager(mock_client)

    def test_init(self, mock_client):
        """Should initialize with client."""
        manager = MemoryManager(mock_client)
        assert manager._client is mock_client


class TestFormatParticipants:
    """Tests for _format_participants method."""

    @pytest.fixture
    def manager(self):
        """Create memory manager with mock client."""
        return MemoryManager(MagicMock())

    def test_formats_participants_list(self, manager):
        """Should format participants as markdown list."""
        participants = [
            {"name": "Alice", "type": "User"},
            {"name": "DataBot", "type": "Agent"},
        ]
        result = manager._format_participants(participants)

        assert "## Current Room Participants" in result
        assert "- Alice (User)" in result
        assert "- DataBot (Agent)" in result
        assert "EXACT name" in result

    def test_handles_empty_list(self, manager):
        """Should handle empty participants list."""
        result = manager._format_participants([])

        assert "## Current Room Participants" in result
        assert "EXACT name" in result

    def test_handles_missing_fields(self, manager):
        """Should handle participants with missing fields."""
        participants = [
            {"name": "Alice"},  # Missing type
            {"type": "Agent"},  # Missing name
        ]
        result = manager._format_participants(participants)

        assert "- Alice (Unknown)" in result
        assert "- Unknown (Agent)" in result


class TestParseRoomContexts:
    """Tests for _parse_room_contexts method."""

    @pytest.fixture
    def manager(self):
        """Create memory manager with mock client."""
        return MemoryManager(MagicMock())

    def test_parses_multiple_rooms(self, manager):
        """Should parse multiple room contexts."""
        value = """## Room: room-123
Topic: Budget planning
Key points: Q4 review

## Room: room-456
Topic: Product roadmap
"""
        result = manager._parse_room_contexts(value)

        assert "room-123" in result
        assert "room-456" in result
        assert "Budget planning" in result["room-123"]
        assert "Product roadmap" in result["room-456"]

    def test_handles_empty_value(self, manager):
        """Should handle empty or None value."""
        assert manager._parse_room_contexts("") == {}
        assert manager._parse_room_contexts("No room contexts yet") == {}

    def test_handles_single_room(self, manager):
        """Should handle single room context."""
        value = """## Room: room-123
Topic: Discussion
Key points: Important stuff"""
        result = manager._parse_room_contexts(value)

        assert len(result) == 1
        assert "room-123" in result
        assert "Discussion" in result["room-123"]


class TestFormatRoomContexts:
    """Tests for _format_room_contexts method."""

    @pytest.fixture
    def manager(self):
        """Create memory manager with mock client."""
        return MemoryManager(MagicMock())

    def test_formats_contexts_dict(self, manager):
        """Should format contexts dict as markdown."""
        contexts = {
            "room-123": "Topic: Budget review",
            "room-456": "Topic: Product planning",
        }
        result = manager._format_room_contexts(contexts)

        assert "## Room: room-123" in result
        assert "## Room: room-456" in result
        assert "Budget review" in result
        assert "Product planning" in result

    def test_handles_empty_dict(self, manager):
        """Should return default message for empty dict."""
        result = manager._format_room_contexts({})

        assert "No room contexts yet" in result

    def test_sorts_rooms(self, manager):
        """Should sort rooms alphabetically."""
        contexts = {
            "z-room": "Topic: Z",
            "a-room": "Topic: A",
        }
        result = manager._format_room_contexts(contexts)

        # a-room should come before z-room
        a_idx = result.index("a-room")
        z_idx = result.index("z-room")
        assert a_idx < z_idx


class TestUpdateParticipants:
    """Tests for update_participants async method."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Letta client."""
        client = MagicMock()
        client.agents = MagicMock()
        client.agents.blocks = MagicMock()
        client.agents.blocks.modify = MagicMock()
        return client

    @pytest.fixture
    def manager(self, mock_client):
        """Create memory manager with mock client."""
        return MemoryManager(mock_client)

    @pytest.mark.asyncio
    async def test_updates_participants_block(self, manager, mock_client):
        """Should call blocks.update with participants."""
        participants = [{"name": "Alice", "type": "User"}]

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None
            await manager.update_participants("agent-123", participants)

            mock_to_thread.assert_called_once()
            call_args = mock_to_thread.call_args
            assert call_args[0][0] == mock_client.agents.blocks.update

    @pytest.mark.asyncio
    async def test_handles_errors_gracefully(self, manager, mock_client):
        """Should log warning on error, not raise."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = Exception("API error")

            # Should not raise
            await manager.update_participants("agent-123", [])


class TestGetRoomContexts:
    """Tests for get_room_contexts async method."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Letta client."""
        client = MagicMock()
        client.agents = MagicMock()
        client.agents.blocks = MagicMock()
        return client

    @pytest.fixture
    def manager(self, mock_client):
        """Create memory manager with mock client."""
        return MemoryManager(mock_client)

    @pytest.mark.asyncio
    async def test_returns_parsed_contexts(self, manager, mock_client):
        """Should parse and return room contexts."""
        mock_block = MagicMock()
        mock_block.label = "room_contexts"
        mock_block.value = "## Room: room-123\nTopic: Test"

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_block
            result = await manager.get_room_contexts("agent-123")

            assert "room-123" in result
            assert "Test" in result["room-123"]

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self, manager, mock_client):
        """Should return empty dict on error."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = Exception("API error")
            result = await manager.get_room_contexts("agent-123")

            assert result == {}


class TestUpdateRoomContext:
    """Tests for update_room_context async method."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Letta client."""
        client = MagicMock()
        client.agents = MagicMock()
        client.agents.blocks = MagicMock()
        client.agents.blocks.modify = MagicMock()
        client.agents.blocks.retrieve = MagicMock()
        return client

    @pytest.fixture
    def manager(self, mock_client):
        """Create memory manager with mock client."""
        return MemoryManager(mock_client)

    @pytest.mark.asyncio
    async def test_updates_specific_room(self, manager, mock_client):
        """Should update context for specific room."""
        mock_memory = MagicMock()
        mock_memory.blocks = []  # Empty initially

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_memory
            await manager.update_room_context(
                agent_id="agent-123",
                room_id="room-456",
                topic="Project planning",
                key_points=["Deadline is next week", "Need more resources"],
            )

            # Should have been called twice: retrieve and modify
            assert mock_to_thread.call_count == 2


class TestConsolidateRoomMemory:
    """Tests for consolidate_room_memory async method."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Letta client."""
        client = MagicMock()
        client.agents = MagicMock()
        client.agents.blocks = MagicMock()
        client.agents.blocks.modify = MagicMock()
        client.agents.blocks.retrieve = MagicMock()
        return client

    @pytest.fixture
    def manager(self, mock_client):
        """Create memory manager with mock client."""
        return MemoryManager(mock_client)

    @pytest.mark.asyncio
    async def test_consolidates_with_summary(self, manager, mock_client):
        """Should store condensed summary."""
        mock_memory = MagicMock()
        mock_memory.blocks = []

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_memory
            await manager.consolidate_room_memory(
                agent_id="agent-123",
                room_id="room-456",
                summary="Important decisions were made about the Q4 budget",
            )

            assert mock_to_thread.call_count == 2
