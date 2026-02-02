"""Tests for Letta adapter modes and configuration."""

import pytest

from thenvoi.adapters.letta.modes import LettaConfig, LettaMode


class TestLettaMode:
    """Tests for LettaMode enum."""

    def test_per_room_mode_value(self):
        """Should have correct value for PER_ROOM."""
        assert LettaMode.PER_ROOM.value == "per_room"

    def test_shared_mode_value(self):
        """Should have correct value for SHARED."""
        assert LettaMode.SHARED.value == "shared"


class TestLettaConfig:
    """Tests for LettaConfig dataclass."""

    def test_api_key_defaults_to_none(self):
        """Should default api_key to None."""
        config = LettaConfig()
        assert config.api_key is None

    def test_accepts_api_key(self):
        """Should accept api_key when provided."""
        config = LettaConfig(api_key="sk-let-test")
        assert config.api_key == "sk-let-test"

    def test_per_room_mode_config(self):
        """Should accept per-room mode config."""
        config = LettaConfig(
            api_key="sk-let-test",
            mode=LettaMode.PER_ROOM,
            model="gpt-4o",
        )
        assert config.mode == LettaMode.PER_ROOM
        assert config.model == "gpt-4o"

    def test_shared_mode_config(self):
        """Should accept shared mode config."""
        config = LettaConfig(
            api_key="sk-let-test",
            mode=LettaMode.SHARED,
            shared_agent_id="agent-123",
        )
        assert config.mode == LettaMode.SHARED
        assert config.shared_agent_id == "agent-123"

    def test_shared_mode_without_agent_id(self):
        """Should allow shared mode without agent_id (will create one)."""
        config = LettaConfig(mode=LettaMode.SHARED)
        assert config.shared_agent_id is None

    def test_default_values(self):
        """Should have sensible defaults."""
        config = LettaConfig()

        assert config.mode == LettaMode.PER_ROOM
        assert config.base_url == "https://api.letta.com"
        assert config.model == "openai/gpt-4o"
        assert config.embedding_model == "openai/text-embedding-3-small"
        assert config.api_timeout == 30
        assert config.tool_execution_timeout == 30
        assert config.max_reasoning_steps == 10
        assert config.custom_tools == []
        assert config.persona is None
        assert config.shared_agent_id is None
        assert config.api_key is None

    def test_custom_base_url(self):
        """Should accept custom base URL for self-hosted."""
        config = LettaConfig(base_url="http://localhost:8283")

        assert config.base_url == "http://localhost:8283"

    def test_custom_tools_list(self):
        """Should accept custom tools list."""
        mock_tool = {"name": "test_tool"}
        config = LettaConfig(custom_tools=[mock_tool, "builtin_tool"])

        assert len(config.custom_tools) == 2
        assert config.custom_tools[0] == mock_tool
        assert config.custom_tools[1] == "builtin_tool"


class TestLettaConfigMemoryBlockLimits:
    """Tests for memory block limit defaults and validation."""

    def test_default_memory_block_limits(self):
        """Default memory block limits should be set."""
        config = LettaConfig()
        assert config.persona_limit == 2000
        assert config.participants_limit == 2000
        assert config.room_contexts_limit == 5000

    def test_accepts_none_limits(self):
        """Should accept None as unlimited."""
        config = LettaConfig(
            persona_limit=None,
            participants_limit=None,
            room_contexts_limit=None,
        )
        assert config.persona_limit is None
        assert config.participants_limit is None
        assert config.room_contexts_limit is None

    def test_accepts_positive_limits(self):
        """Should accept positive limits."""
        config = LettaConfig(
            persona_limit=500,
            participants_limit=1000,
            room_contexts_limit=3000,
        )
        assert config.persona_limit == 500
        assert config.participants_limit == 1000
        assert config.room_contexts_limit == 3000

    def test_accepts_limit_of_one(self):
        """Should accept limit of 1 (minimum valid positive)."""
        config = LettaConfig(persona_limit=1)
        assert config.persona_limit == 1


class TestLettaConfigValidation:
    """Tests for LettaConfig validation errors."""

    def test_rejects_zero_persona_limit(self):
        """Should reject persona_limit of 0."""
        with pytest.raises(ValueError, match="persona_limit must be positive"):
            LettaConfig(persona_limit=0)

    def test_rejects_negative_persona_limit(self):
        """Should reject negative persona_limit."""
        with pytest.raises(ValueError, match="persona_limit must be positive"):
            LettaConfig(persona_limit=-100)

    def test_rejects_zero_participants_limit(self):
        """Should reject participants_limit of 0."""
        with pytest.raises(ValueError, match="participants_limit must be positive"):
            LettaConfig(participants_limit=0)

    def test_rejects_negative_participants_limit(self):
        """Should reject negative participants_limit."""
        with pytest.raises(ValueError, match="participants_limit must be positive"):
            LettaConfig(participants_limit=-50)

    def test_rejects_zero_room_contexts_limit(self):
        """Should reject room_contexts_limit of 0."""
        with pytest.raises(ValueError, match="room_contexts_limit must be positive"):
            LettaConfig(room_contexts_limit=0)

    def test_rejects_negative_room_contexts_limit(self):
        """Should reject negative room_contexts_limit."""
        with pytest.raises(ValueError, match="room_contexts_limit must be positive"):
            LettaConfig(room_contexts_limit=-1)
