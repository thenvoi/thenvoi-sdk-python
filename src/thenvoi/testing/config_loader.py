"""Shared runner configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from thenvoi.config.runtime import (
    resolve_agent_credentials,
    resolve_platform_urls as _resolve_platform_urls,
)

# Compatibility export for runner modules that import this symbol here.
resolve_platform_urls = _resolve_platform_urls


def load_yaml_mapping(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and require a top-level mapping."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    try:
        with open(path, encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read config file: {exc}") from exc

    if config is None:
        raise ValueError("Config file is empty")
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a mapping at the top level")
    return dict(config)


def load_runner_config(config_path: str | Path, agent_key: str) -> dict[str, Any]:
    """Load runner config from YAML and inject validated credentials."""
    path = Path(config_path).resolve()
    config = load_yaml_mapping(path)

    agent_id, api_key = resolve_agent_credentials(agent_key, config_path=path)
    agent_section = config.get(agent_key)
    if agent_section is not None and not isinstance(agent_section, dict):
        raise ValueError(
            f"Agent section '{agent_key}' must be a mapping, got {type(agent_section).__name__}"
        )

    result = dict(agent_section) if agent_section else dict(config)
    result["agent_id"] = agent_id
    result["api_key"] = api_key
    return result


def load_config(config_path: str | Path, agent_key: str) -> dict[str, Any]:
    """Compatibility alias for existing runner imports."""
    return load_runner_config(config_path, agent_key)
