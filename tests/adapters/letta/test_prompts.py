"""Tests for Letta adapter prompts."""

from thenvoi.adapters.letta.modes import LettaMode
from thenvoi.adapters.letta.prompts import (
    build_consolidation_prompt,
    build_room_entry_context,
    get_system_prompt,
)


class TestGetSystemPrompt:
    """Tests for get_system_prompt()."""

    def test_per_room_prompt(self):
        """Should return per-room system prompt."""
        prompt = get_system_prompt(LettaMode.PER_ROOM)

        assert "isolated to this room only" in prompt
        assert "memory_replace" in prompt
        assert "create_agent_chat_message" in prompt

    def test_shared_prompt(self):
        """Should return shared mode system prompt."""
        prompt = get_system_prompt(LettaMode.SHARED)

        assert "multiple chat rooms" in prompt
        assert "room_contexts" in prompt
        assert "Do NOT leak information" in prompt


class TestBuildRoomEntryContext:
    """Tests for build_room_entry_context()."""

    def test_basic_context(self):
        """Should build basic room context."""
        result = build_room_entry_context(
            room_id="room-123456789",
            participants=["Alice", "Bob"],
        )

        assert "[Room:" in result
        assert "Alice, Bob" in result

    def test_truncates_room_id(self):
        """Should truncate long room IDs."""
        result = build_room_entry_context(
            room_id="very-long-room-id-that-should-be-truncated",
            participants=["Alice"],
        )

        assert "very-long-room-i" in result
        assert "truncated" not in result

    def test_includes_time_context(self):
        """Should include time context when provided."""
        result = build_room_entry_context(
            room_id="room-123",
            participants=["Alice"],
            last_interaction_ago="2 weeks",
            previous_summary="Q4 budget review",
        )

        assert "Last interaction was 2 weeks" in result
        assert "Q4 budget review" in result

    def test_includes_time_without_summary(self):
        """Should include time even without summary."""
        result = build_room_entry_context(
            room_id="room-123",
            participants=["Alice"],
            last_interaction_ago="3 days",
        )

        assert "Last interaction was 3 days" in result

    def test_includes_participant_changes(self):
        """Should include participant changes."""
        result = build_room_entry_context(
            room_id="room-123",
            participants=["Alice", "Bob"],
            participant_changes="Carol joined the room",
        )

        assert "[Update: Carol joined the room]" in result


class TestBuildConsolidationPrompt:
    """Tests for build_consolidation_prompt()."""

    def test_includes_room_id(self):
        """Should include room ID in prompt."""
        result = build_consolidation_prompt("room-abc-123")

        assert "room-abc-123" in result

    def test_includes_consolidation_instructions(self):
        """Should include consolidation instructions."""
        result = build_consolidation_prompt("room-123")

        assert "Memory Consolidation" in result
        assert "Key decisions" in result
        assert "Important facts" in result
        assert "Action items" in result
        assert "memory_replace" in result or "memory_rethink" in result
