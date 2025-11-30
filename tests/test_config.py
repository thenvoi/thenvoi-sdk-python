"""
Config loading tests - verify agent configuration management.

Tests cover all error paths and edge cases for loading agent credentials
from YAML configuration files.
"""

import pytest
from thenvoi.config import load_agent_config


def test_load_valid_config_success(tmp_path, monkeypatch):
    """Should successfully load agent credentials from valid config."""
    # Create a valid config file
    config_content = """
simple_agent:
  agent_id: test-agent-123
  api_key: test-key-abc
    """
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text(config_content)

    # Mock get_config_path to return our test file
    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Load config
    agent_id, api_key = load_agent_config("simple_agent")

    # Verify
    assert agent_id == "test-agent-123"
    assert api_key == "test-key-abc"


def test_missing_config_file(tmp_path, monkeypatch):
    """Should raise FileNotFoundError with helpful message when config missing."""
    # Point to non-existent file
    non_existent = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: non_existent)

    # Attempt to load config
    with pytest.raises(FileNotFoundError) as exc_info:
        load_agent_config("any_agent")

    # Verify error message is helpful
    assert "agent_config.yaml not found" in str(exc_info.value)
    assert "agent_config.yaml.example" in str(exc_info.value)


def test_agent_key_not_in_config(tmp_path, monkeypatch):
    """Should raise ValueError when requested agent key doesn't exist."""
    # Create config without the requested agent
    config_content = """
simple_agent:
  agent_id: test-agent-123
  api_key: test-key-abc
    """
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text(config_content)

    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Try to load non-existent agent
    with pytest.raises(ValueError) as exc_info:
        load_agent_config("nonexistent_agent")

    # Verify error message mentions the agent key
    assert "nonexistent_agent" in str(exc_info.value)
    assert "not found" in str(exc_info.value)


def test_missing_agent_id_field(tmp_path, monkeypatch):
    """Should raise ValueError when agent_id field is missing."""
    # Create config with api_key but no agent_id
    config_content = """
simple_agent:
  api_key: test-key-abc
    """
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text(config_content)

    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Try to load config
    with pytest.raises(ValueError) as exc_info:
        load_agent_config("simple_agent")

    # Verify error message mentions missing field
    assert "agent_id" in str(exc_info.value)
    assert "Missing required fields" in str(exc_info.value)


def test_missing_api_key_field(tmp_path, monkeypatch):
    """Should raise ValueError when api_key field is missing."""
    # Create config with agent_id but no api_key
    config_content = """
simple_agent:
  agent_id: test-agent-123
    """
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text(config_content)

    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Try to load config
    with pytest.raises(ValueError) as exc_info:
        load_agent_config("simple_agent")

    # Verify error message mentions missing field
    assert "api_key" in str(exc_info.value)
    assert "Missing required fields" in str(exc_info.value)


def test_missing_both_fields(tmp_path, monkeypatch):
    """Should raise ValueError listing both missing fields."""
    # Create config with neither field
    config_content = """
simple_agent:
  some_other_field: value
    """
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text(config_content)

    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Try to load config
    with pytest.raises(ValueError) as exc_info:
        load_agent_config("simple_agent")

    # Verify both fields mentioned
    error_msg = str(exc_info.value)
    assert "agent_id" in error_msg
    assert "api_key" in error_msg
    assert "Missing required fields" in error_msg


def test_empty_agent_id(tmp_path, monkeypatch):
    """Should raise ValueError when agent_id is empty string."""
    # Create config with empty agent_id
    config_content = """
simple_agent:
  agent_id: ""
  api_key: test-key-abc
    """
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text(config_content)

    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Try to load config
    with pytest.raises(ValueError) as exc_info:
        load_agent_config("simple_agent")

    # Verify error treats empty as missing
    assert "agent_id" in str(exc_info.value)
    assert "Missing required fields" in str(exc_info.value)


def test_empty_api_key(tmp_path, monkeypatch):
    """Should raise ValueError when api_key is empty string."""
    # Create config with empty api_key
    config_content = """
simple_agent:
  agent_id: test-agent-123
  api_key: ""
    """
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text(config_content)

    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Try to load config
    with pytest.raises(ValueError) as exc_info:
        load_agent_config("simple_agent")

    # Verify error treats empty as missing
    assert "api_key" in str(exc_info.value)
    assert "Missing required fields" in str(exc_info.value)


def test_empty_yaml_file(tmp_path, monkeypatch):
    """Should raise ValueError when YAML file is empty."""
    # Create empty config file
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text("")

    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Try to load config
    with pytest.raises(ValueError) as exc_info:
        load_agent_config("simple_agent")

    # Verify error message indicates agent not found
    assert "simple_agent" in str(exc_info.value)
    assert "not found" in str(exc_info.value)


def test_invalid_yaml_syntax(tmp_path, monkeypatch):
    """Should raise RuntimeError when YAML is malformed."""
    # Create file with invalid YAML
    config_content = """
simple_agent:
  agent_id: test-agent-123
  api_key: [unclosed bracket
    """
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text(config_content)

    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Try to load config
    with pytest.raises(RuntimeError) as exc_info:
        load_agent_config("simple_agent")

    # Verify it's wrapped in RuntimeError
    assert "Error loading agent config" in str(exc_info.value)


def test_load_multiple_agents_from_same_file(tmp_path, monkeypatch):
    """Should successfully load different agents from same config file."""
    # Create config with multiple agents
    config_content = """
agent_one:
  agent_id: agent-1-id
  api_key: agent-1-key

agent_two:
  agent_id: agent-2-id
  api_key: agent-2-key
    """
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text(config_content)

    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Load first agent
    agent_id_1, api_key_1 = load_agent_config("agent_one")
    assert agent_id_1 == "agent-1-id"
    assert api_key_1 == "agent-1-key"

    # Load second agent
    agent_id_2, api_key_2 = load_agent_config("agent_two")
    assert agent_id_2 == "agent-2-id"
    assert api_key_2 == "agent-2-key"


def test_config_with_extra_fields_ignored(tmp_path, monkeypatch):
    """Should ignore extra fields and only return agent_id and api_key."""
    # Create config with extra fields (forward compatibility)
    config_content = """
simple_agent:
  agent_id: test-agent-123
  api_key: test-key-abc
  description: "This is a test agent"
  extra_field: "should be ignored"
  version: 2
    """
    config_file = tmp_path / "agent_config.yaml"
    config_file.write_text(config_content)

    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: config_file)

    # Load config
    agent_id, api_key = load_agent_config("simple_agent")

    # Verify only required fields returned
    assert agent_id == "test-agent-123"
    assert api_key == "test-key-abc"
    # Function returns tuple of (agent_id, api_key), nothing else
    assert isinstance((agent_id, api_key), tuple)
    assert len((agent_id, api_key)) == 2
