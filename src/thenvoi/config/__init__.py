"""Agent configuration helpers public namespace."""

from __future__ import annotations

from pathlib import Path

from thenvoi.config.loader import (
    get_config_path as _get_config_path,
    load_agent_config as _load_agent_config,
)
from thenvoi.config.runtime import (
    resolve_agent_credentials as _resolve_agent_credentials,
    resolve_platform_urls as _resolve_platform_urls,
)


def get_config_path() -> Path:
    """Return the default local configuration file path."""
    return _get_config_path()


def load_agent_config(
    agent_key: str,
    *,
    config_path: str | Path | None = None,
) -> tuple[str, str]:
    """Load credentials for an agent key from configured sources."""
    return _load_agent_config(agent_key, config_path=config_path)


def resolve_agent_credentials(
    agent_key: str,
    *,
    config_path: str | Path | None = None,
) -> tuple[str, str]:
    """Resolve validated credentials for the given agent key."""
    return _resolve_agent_credentials(agent_key, config_path=config_path)


def resolve_platform_urls(
    *,
    ws_env_key: str = "THENVOI_WS_URL",
    rest_env_key: str = "THENVOI_REST_URL",
    ws_default: str | None = None,
    rest_default: str | None = None,
) -> tuple[str, str]:
    """Resolve runtime platform URLs from environment variables."""
    return _resolve_platform_urls(
        ws_env_key=ws_env_key,
        rest_env_key=rest_env_key,
        ws_default=ws_default,
        rest_default=rest_default,
    )

__all__ = [
    "load_agent_config",
    "get_config_path",
    "resolve_agent_credentials",
    "resolve_platform_urls",
]
