"""Runtime credential and endpoint resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path

from thenvoi.config.loader import load_agent_config


def resolve_platform_urls(
    *,
    ws_env_key: str = "THENVOI_WS_URL",
    rest_env_key: str = "THENVOI_REST_URL",
    ws_default: str | None = None,
    rest_default: str | None = None,
) -> tuple[str, str]:
    """Resolve Thenvoi platform URLs from environment with optional defaults."""
    ws_url = os.getenv(ws_env_key, ws_default)
    rest_url = os.getenv(rest_env_key, rest_default)
    if not ws_url:
        raise ValueError(f"{ws_env_key} environment variable is required")
    if not rest_url:
        raise ValueError(f"{rest_env_key} environment variable is required")
    return ws_url, rest_url


def resolve_agent_credentials(
    agent_key: str,
    *,
    config_path: str | Path | None = None,
) -> tuple[str, str]:
    """Load validated agent credentials from config or environment sources."""
    path = Path(config_path).resolve() if config_path is not None else None
    return load_agent_config(agent_key, config_path=path)
