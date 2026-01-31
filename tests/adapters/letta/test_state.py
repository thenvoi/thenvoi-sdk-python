"""Tests for Letta adapter state management."""

from pathlib import Path


from thenvoi.adapters.letta.state import (
    LettaAdapterState,
    RoomState,
    StateStore,
)


class TestRoomState:
    """Tests for RoomState dataclass."""

    def test_create_room_state(self):
        """Should create room state with defaults."""
        state = RoomState(room_id="room-123")

        assert state.room_id == "room-123"
        assert state.is_active is True
        assert state.letta_agent_id is None
        assert state.letta_conversation_id is None
        assert state.summary is None
        assert state.participants_snapshot == []

    def test_mark_interaction(self):
        """Should update state after an interaction."""
        state = RoomState(room_id="room-123")
        state.mark_interaction("msg-456", ["Alice", "Bob"])

        assert state.last_message_id == "msg-456"
        assert state.participants_snapshot == ["Alice", "Bob"]
        assert state.last_interaction is not None
        assert state.is_active is True

    def test_serialization_roundtrip(self):
        """Should serialize and deserialize correctly."""
        state = RoomState(
            room_id="room-123",
            letta_agent_id="agent-456",
            letta_conversation_id="conv-789",
            summary="Discussing Q4 sales",
            participants_snapshot=["Alice", "Bob"],
            is_active=False,
        )
        state.mark_interaction("msg-789", ["Alice", "Bob", "Carol"])

        data = state.to_dict()
        restored = RoomState.from_dict(data)

        assert restored.room_id == state.room_id
        assert restored.letta_agent_id == state.letta_agent_id
        assert restored.letta_conversation_id == state.letta_conversation_id
        assert restored.summary == state.summary
        assert restored.participants_snapshot == state.participants_snapshot
        assert restored.last_message_id == state.last_message_id
        assert restored.is_active == state.is_active

    def test_serialization_with_none_values(self):
        """Should handle None values in serialization."""
        state = RoomState(room_id="room-123")

        data = state.to_dict()
        restored = RoomState.from_dict(data)

        assert restored.room_id == "room-123"
        assert restored.letta_agent_id is None
        assert restored.last_interaction is None


class TestLettaAdapterState:
    """Tests for LettaAdapterState dataclass."""

    def test_get_or_create_room_state(self):
        """Should create room state if not exists."""
        state = LettaAdapterState()

        room_state = state.get_or_create_room_state("room-123")

        assert room_state.room_id == "room-123"
        assert "room-123" in state.room_states

    def test_get_or_create_returns_existing(self):
        """Should return existing room state."""
        state = LettaAdapterState()
        room_state = state.get_or_create_room_state("room-123")
        room_state.summary = "Test summary"

        room_state2 = state.get_or_create_room_state("room-123")

        assert room_state2.summary == "Test summary"
        assert room_state is room_state2

    def test_set_and_get_room_agent(self):
        """Should associate and retrieve agent ID for room."""
        state = LettaAdapterState()

        state.set_room_agent("room-123", "agent-456")

        assert state.get_room_agent("room-123") == "agent-456"

    def test_get_room_agent_nonexistent(self):
        """Should return None for nonexistent room."""
        state = LettaAdapterState()

        assert state.get_room_agent("nonexistent") is None

    def test_set_and_get_room_conversation(self):
        """Should associate and retrieve conversation ID for room (SHARED mode)."""
        state = LettaAdapterState()

        state.set_room_conversation("room-123", "conv-456")

        assert state.get_room_conversation("room-123") == "conv-456"

    def test_get_room_conversation_nonexistent(self):
        """Should return None for nonexistent room."""
        state = LettaAdapterState()

        assert state.get_room_conversation("nonexistent") is None

    def test_mark_room_inactive(self):
        """Should mark room as inactive."""
        state = LettaAdapterState()
        state.get_or_create_room_state("room-123")

        state.mark_room_inactive("room-123")

        assert state.room_states["room-123"].is_active is False

    def test_mark_room_active(self):
        """Should mark room as active."""
        state = LettaAdapterState()
        state.get_or_create_room_state("room-123")
        state.mark_room_inactive("room-123")

        state.mark_room_active("room-123")

        assert state.room_states["room-123"].is_active is True

    def test_get_active_rooms(self):
        """Should return only active rooms."""
        state = LettaAdapterState()
        state.get_or_create_room_state("room-1")
        state.get_or_create_room_state("room-2")
        state.get_or_create_room_state("room-3")
        state.mark_room_inactive("room-2")

        active = state.get_active_rooms()

        assert set(active) == {"room-1", "room-3"}

    def test_get_inactive_rooms(self):
        """Should return only inactive rooms."""
        state = LettaAdapterState()
        state.get_or_create_room_state("room-1")
        state.get_or_create_room_state("room-2")
        state.get_or_create_room_state("room-3")
        state.mark_room_inactive("room-2")

        inactive = state.get_inactive_rooms()

        assert inactive == ["room-2"]

    def test_serialization_roundtrip(self):
        """Should serialize and deserialize full state."""
        state = LettaAdapterState(
            shared_agent_id="shared-agent-123",
            thenvoi_agent_id="thenvoi-agent-456",
            mode="shared",
        )
        state.set_room_agent("room-1", "agent-1")
        state.set_room_agent("room-2", "agent-2")
        state.set_room_conversation("room-3", "conv-3")
        state.mark_room_inactive("room-2")

        data = state.to_dict()
        restored = LettaAdapterState.from_dict(data)

        assert restored.shared_agent_id == state.shared_agent_id
        assert restored.thenvoi_agent_id == state.thenvoi_agent_id
        assert restored.mode == state.mode
        assert restored.get_room_agent("room-1") == "agent-1"
        assert restored.get_room_agent("room-2") == "agent-2"
        assert restored.get_room_conversation("room-3") == "conv-3"
        assert restored.room_states["room-2"].is_active is False


class TestStateStore:
    """Tests for StateStore persistence."""

    def test_create_new_state(self, tmp_path: Path):
        """Should create new state when file doesn't exist."""
        store = StateStore(tmp_path / "state.json")
        state = store.load()

        assert state is not None
        assert state.shared_agent_id is None
        assert len(state.room_states) == 0

    def test_persist_and_load(self, tmp_path: Path):
        """Should persist and reload state correctly."""
        store = StateStore(tmp_path / "state.json")
        state = store.load()

        state.set_room_agent("room-123", "agent-456")
        state.shared_agent_id = "shared-agent"
        store.save()

        # Load fresh
        store2 = StateStore(tmp_path / "state.json")
        state2 = store2.load()

        assert state2.get_room_agent("room-123") == "agent-456"
        assert state2.shared_agent_id == "shared-agent"

    def test_state_property_loads_if_needed(self, tmp_path: Path):
        """State property should auto-load."""
        store = StateStore(tmp_path / "state.json")

        # Access via property without explicit load
        state = store.state

        assert state is not None

    def test_handles_corrupted_file(self, tmp_path: Path):
        """Should handle corrupted JSON gracefully."""
        state_file = tmp_path / "state.json"
        state_file.write_text("{ invalid json }")

        store = StateStore(state_file)
        state = store.load()

        # Should create fresh state
        assert state is not None
        assert len(state.room_states) == 0

    def test_atomic_save(self, tmp_path: Path):
        """Should use atomic write to prevent corruption."""
        store = StateStore(tmp_path / "state.json")
        state = store.load()

        state.set_room_agent("room-123", "agent-456")
        store.save()

        # Verify no temp file left behind
        assert not (tmp_path / "state.tmp").exists()
        assert (tmp_path / "state.json").exists()

    def test_reset(self, tmp_path: Path):
        """Should reset state and delete file."""
        store = StateStore(tmp_path / "state.json")
        state = store.load()
        state.set_room_agent("room-123", "agent-456")
        store.save()

        store.reset()

        assert store.state.shared_agent_id is None
        assert len(store.state.room_states) == 0
        assert not (tmp_path / "state.json").exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        """Should create parent directories if needed."""
        nested_path = tmp_path / "deep" / "nested" / "state.json"
        store = StateStore(nested_path)
        state = store.load()

        state.set_room_agent("room-123", "agent-456")
        store.save()

        assert nested_path.exists()

    def test_tilde_expansion(self):
        """Should expand ~ in path."""
        store = StateStore("~/.thenvoi/test_state.json")

        assert "~" not in str(store.storage_path)
        assert store.storage_path.is_absolute()

    def test_save_if_dirty(self, tmp_path: Path):
        """Should only save when marked dirty."""
        store = StateStore(tmp_path / "state.json")
        state = store.load()

        # Not dirty - should not save
        store.save_if_dirty()
        assert not (tmp_path / "state.json").exists()

        # Mark dirty and save
        state.set_room_agent("room-123", "agent-456")
        store.mark_dirty()
        store.save_if_dirty()

        assert (tmp_path / "state.json").exists()
