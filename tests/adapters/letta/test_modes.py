"""Tests for Letta adapter modes and configuration."""

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
