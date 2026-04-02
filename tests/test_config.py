"""Config loading tests for typed agent configuration."""

from __future__ import annotations

import warnings
from uuid import UUID

import pytest

from thenvoi.config import AgentConfig, build_adapter_from_config, load_agent_config
from thenvoi.core.exceptions import ThenvoiConfigError


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """Provide a writable config path wired into the loader."""
    path = tmp_path / "agent_config.yaml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: path)
    return path


def test_load_valid_config_success(config_file) -> None:
    config_file.write_text(
        """
simple_agent:
  agent_id: "11111111-1111-1111-1111-111111111111"
  api_key: test-key-abc
        """.strip()
    )

    config = load_agent_config("simple_agent")

    assert isinstance(config, AgentConfig)
    assert config.agent_id == UUID("11111111-1111-1111-1111-111111111111")
    assert config.api_key == "test-key-abc"


def test_tuple_unpack_still_works_with_warning(config_file) -> None:
    config_file.write_text(
        """
simple_agent:
  agent_id: "11111111-1111-1111-1111-111111111111"
  api_key: test-key-abc
        """.strip()
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent_id, api_key = load_agent_config("simple_agent")

    assert agent_id == "11111111-1111-1111-1111-111111111111"
    assert api_key == "test-key-abc"
    assert any(item.category is DeprecationWarning for item in caught)


def test_missing_config_file_raises_thenvoi_config_error(tmp_path, monkeypatch) -> None:
    non_existent = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr("thenvoi.config.loader.get_config_path", lambda: non_existent)

    with pytest.raises(ThenvoiConfigError, match="Config file not found"):
        load_agent_config("any_agent")


def test_agent_key_not_in_config(config_file) -> None:
    config_file.write_text(
        """
simple_agent:
  agent_id: "11111111-1111-1111-1111-111111111111"
  api_key: test-key-abc
        """.strip()
    )

    with pytest.raises(
        ThenvoiConfigError, match="Agent 'nonexistent_agent' was not found"
    ):
        load_agent_config("nonexistent_agent")


def test_missing_required_fields(config_file) -> None:
    config_file.write_text(
        """
simple_agent:
  api_key: test-key-abc
        """.strip()
    )

    with pytest.raises(ThenvoiConfigError, match="Missing required field 'agent_id'"):
        load_agent_config("simple_agent")


def test_invalid_uuid_raises_thenvoi_config_error(config_file) -> None:
    config_file.write_text(
        """
simple_agent:
  agent_id: not-a-uuid
  api_key: test-key-abc
        """.strip()
    )

    with pytest.raises(ThenvoiConfigError, match="Invalid agent_id"):
        load_agent_config("simple_agent")


def test_invalid_yaml_syntax(config_file) -> None:
    config_file.write_text(
        """
simple_agent:
  agent_id: [unclosed
        """.strip()
    )

    with pytest.raises(ThenvoiConfigError, match="Invalid YAML"):
        load_agent_config("simple_agent")


def test_load_multiple_agents_from_same_file(config_file) -> None:
    config_file.write_text(
        """
agent_one:
  agent_id: "11111111-1111-1111-1111-111111111111"
  api_key: agent-1-key

agent_two:
  agent_id: "22222222-2222-2222-2222-222222222222"
  api_key: agent-2-key
        """.strip()
    )

    first = load_agent_config("agent_one")
    second = load_agent_config("agent_two")

    assert first.api_key == "agent-1-key"
    assert second.api_key == "agent-2-key"


def test_config_with_extra_fields_preserved(config_file) -> None:
    config_file.write_text(
        """
simple_agent:
  agent_id: "11111111-1111-1111-1111-111111111111"
  api_key: test-key-abc
  description: This is a test agent
        """.strip()
    )

    config = load_agent_config("simple_agent")

    assert config.extra["description"] == "This is a test agent"


def test_load_with_explicit_config_path(tmp_path) -> None:
    config_path = tmp_path / "custom_config.yaml"
    config_path.write_text(
        """
reviewer:
  agent_id: "33333333-3333-3333-3333-333333333333"
  api_key: reviewer-key
        """.strip()
    )

    config = load_agent_config("reviewer", config_path=config_path)

    assert config.agent_id == UUID("33333333-3333-3333-3333-333333333333")
    assert config.api_key == "reviewer-key"


def test_flat_format_fallback_with_explicit_config_path(tmp_path) -> None:
    config_path = tmp_path / "single_agent.yaml"
    config_path.write_text(
        """
agent_id: "44444444-4444-4444-4444-444444444444"
api_key: flat-key-abc
role: planner
        """.strip()
    )

    config = load_agent_config("any_key", config_path=config_path)

    assert config.agent_id == UUID("44444444-4444-4444-4444-444444444444")
    assert config.api_key == "flat-key-abc"
    assert config.extra["role"] == "planner"


def test_env_expansion_and_defaults(config_file, monkeypatch) -> None:
    monkeypatch.setenv("THENVOI_API_KEY", "expanded-key")
    config_file.write_text(
        """
planner:
  agent_id: "55555555-5555-5555-5555-555555555555"
  api_key: "${THENVOI_API_KEY}"
  prompt: "${MISSING_PROMPT:-fallback prompt}"
        """.strip()
    )

    config = load_agent_config("planner")

    assert config.api_key == "expanded-key"
    assert config.prompt == "fallback prompt"


def test_unresolved_env_var_raises(config_file) -> None:
    config_file.write_text(
        """
planner:
  agent_id: "55555555-5555-5555-5555-555555555555"
  api_key: "${MISSING_API_KEY}"
        """.strip()
    )

    with pytest.raises(
        ThenvoiConfigError, match="Environment variable 'MISSING_API_KEY' is not set"
    ):
        load_agent_config("planner")


def test_prompt_path_is_resolved_relative_to_config(tmp_path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt_file = prompt_dir / "planner.md"
    prompt_file.write_text("Be concise.")
    config_path = tmp_path / "agent_config.yaml"
    config_path.write_text(
        """
planner:
  agent_id: "66666666-6666-6666-6666-666666666666"
  api_key: test-key
  prompt_path: prompts/planner.md
        """.strip()
    )

    config = load_agent_config("planner", config_path=config_path)

    assert config.prompt_path == prompt_file.resolve()


def test_search_order_prefers_agent_config_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    agent_config = tmp_path / "agent_config.yaml"
    agents_config = tmp_path / "agents.yaml"
    agent_config.write_text(
        """
planner:
  agent_id: "77777777-7777-7777-7777-777777777777"
  api_key: first-key
        """.strip()
    )
    agents_config.write_text(
        """
planner:
  agent_id: "88888888-8888-8888-8888-888888888888"
  api_key: second-key
        """.strip()
    )

    config = load_agent_config("planner")

    assert config.api_key == "first-key"


def test_build_adapter_from_config_unknown_type_suggests_match() -> None:
    with pytest.raises(ThenvoiConfigError, match="Did you mean 'anthropic'\?"):
        build_adapter_from_config({"type": "anthopic"})


def test_build_adapter_from_config_requires_type() -> None:
    with pytest.raises(ThenvoiConfigError, match="Missing adapter type"):
        build_adapter_from_config({})


def test_build_adapter_from_config_rejects_langgraph_yaml() -> None:
    with pytest.raises(
        ThenvoiConfigError, match="LangGraph requires a Python-built adapter"
    ):
        build_adapter_from_config({"type": "langgraph"})
