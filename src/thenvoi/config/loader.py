"""Agent configuration management utilities."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from thenvoi.config.types import AgentConfig
from thenvoi.core.exceptions import ThenvoiConfigError

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")
_DEFAULT_CONFIG_FILENAMES = ("agent_config.yaml", "agents.yaml")


def get_config_path() -> Path:
    """Get the path to the agent configuration file."""
    cwd = Path.cwd()
    for filename in _DEFAULT_CONFIG_FILENAMES:
        path = cwd / filename
        if path.exists():
            return path
    return cwd / _DEFAULT_CONFIG_FILENAMES[0]


def _expand_env_var_in_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(name)
        if env_value:
            return env_value
        if env_value == "" and default is not None:
            return default
        if env_value == "":
            raise ThenvoiConfigError(
                f"Environment variable '{name}' is empty. Set {name} or provide a default with ${{{name}:-value}}."
            )
        if default is not None:
            return default
        raise ThenvoiConfigError(
            f"Environment variable '{name}' is not set. Set {name} or provide a default with ${{{name}:-value}}."
        )

    return _ENV_VAR_PATTERN.sub(replace, value)


def _expand_env_vars(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _expand_env_vars(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    if isinstance(data, str):
        return _expand_env_var_in_string(data)
    return data


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ThenvoiConfigError(
            f"Invalid YAML in {path}. Fix the YAML syntax and try again."
        ) from exc
    except OSError as exc:
        raise ThenvoiConfigError(
            f"Could not read config file at {path}. Check that the file exists and is readable."
        ) from exc

    if not isinstance(loaded, dict):
        raise ThenvoiConfigError(
            f"Config file at {path} must contain a YAML mapping. Use key/value YAML at the top level."
        )
    return _expand_env_vars(loaded)


def _select_agent_section(
    config: dict[str, Any], agent_key: str, path: Path
) -> dict[str, Any]:
    agent_config = config.get(agent_key)
    if agent_config is None and "agent_id" in config:
        agent_config = config

    if agent_config is None:
        raise ThenvoiConfigError(
            f"Agent '{agent_key}' was not found in {path}. Add that agent key to the config file or use the flat single-agent format."
        )
    if not isinstance(agent_config, dict):
        raise ThenvoiConfigError(
            f"Agent '{agent_key}' config must be a mapping. Use YAML key/value fields under that agent key."
        )
    return agent_config


def _required_string(
    agent_config: dict[str, Any], field_name: str, agent_key: str
) -> str:
    value = agent_config.get(field_name)
    if not value:
        raise ThenvoiConfigError(
            f"Missing required field '{field_name}' for agent '{agent_key}'. Add {field_name} to the config entry."
        )
    if not isinstance(value, str):
        raise ThenvoiConfigError(
            f"Field '{field_name}' for agent '{agent_key}' must be a string. Update the config value to a string."
        )
    return value


def _optional_string_list(
    agent_config: dict[str, Any], field_name: str, *, default: list[str] | None = None
) -> list[str] | None:
    value = agent_config.get(field_name, default)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ThenvoiConfigError(
            f"Field '{field_name}' must be a list of strings. Update {field_name} to a YAML list of strings."
        )
    return list(value)


def _build_agent_config(
    agent_key: str, agent_config: dict[str, Any], config_path: Path
) -> AgentConfig:
    agent_id = _required_string(agent_config, "agent_id", agent_key)
    api_key = _required_string(agent_config, "api_key", agent_key)

    try:
        agent_uuid = UUID(agent_id)
    except ValueError as exc:
        raise ThenvoiConfigError(
            f"Invalid agent_id for agent '{agent_key}'. Use a valid UUID string."
        ) from exc

    prompt_path_value = agent_config.get("prompt_path")
    if prompt_path_value is not None and not isinstance(prompt_path_value, str):
        raise ThenvoiConfigError(
            "Field 'prompt_path' must be a string path. Set prompt_path to a relative or absolute file path."
        )

    prompt_path = Path(prompt_path_value) if prompt_path_value else None
    if prompt_path is not None and not prompt_path.is_absolute():
        prompt_path = (config_path.parent / prompt_path).resolve()

    adapter = agent_config.get("adapter")
    if adapter is not None and not isinstance(adapter, dict):
        raise ThenvoiConfigError(
            "Field 'adapter' must be a mapping. Put adapter options under adapter: in YAML."
        )

    known_fields = {
        "agent_id",
        "api_key",
        "adapter",
        "capabilities",
        "include_categories",
        "include_tools",
        "exclude_tools",
        "emit",
        "prompt",
        "prompt_path",
    }

    prompt = agent_config.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise ThenvoiConfigError(
            "Field 'prompt' must be a string. Use a YAML string or block scalar for prompt."
        )

    return AgentConfig(
        agent_id=agent_uuid,
        api_key=api_key,
        adapter=adapter,
        capabilities=_optional_string_list(agent_config, "capabilities", default=[])
        or [],
        include_categories=_optional_string_list(agent_config, "include_categories"),
        include_tools=_optional_string_list(agent_config, "include_tools"),
        exclude_tools=_optional_string_list(agent_config, "exclude_tools"),
        emit=_optional_string_list(agent_config, "emit", default=[]) or [],
        prompt=prompt,
        prompt_path=prompt_path,
        extra={
            key: value for key, value in agent_config.items() if key not in known_fields
        },
    )


def load_agent_config(
    agent_key: str,
    *,
    config_path: str | Path | None = None,
) -> AgentConfig:
    """Load typed agent config from YAML."""
    path = Path(config_path) if config_path is not None else get_config_path()
    logger.debug("Loading config from: %s", path)

    if not path.exists():
        raise ThenvoiConfigError(
            f"Config file not found at {path}. Copy agent_config.yaml.example to agent_config.yaml and configure your agent."
        )

    config = _read_yaml(path)
    selected = _select_agent_section(config, agent_key, path)
    return _build_agent_config(agent_key, selected, path)
