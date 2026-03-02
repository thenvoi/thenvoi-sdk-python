"""Shared dynamic tool loading for runner scripts."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def load_custom_tools(
    tools_dir: Path,
    config_dir: Path,
    tool_names: Sequence[str],
    *,
    logger: logging.Logger,
) -> list[Any]:
    """Load selected tools from a local TOOL_REGISTRY module safely."""
    resolved_tools_dir = tools_dir.resolve()
    resolved_config_dir = config_dir.resolve()

    try:
        resolved_tools_dir.relative_to(resolved_config_dir.parent)
    except ValueError:
        logger.warning(
            "Tools directory %s is outside allowed path, skipping",
            resolved_tools_dir,
        )
        return []

    tools_init = resolved_tools_dir / "__init__.py"
    if not tools_init.exists():
        return []

    try:
        spec = importlib.util.spec_from_file_location("tools", tools_init)
        if spec is None or spec.loader is None:
            logger.warning("Could not create module spec for tools")
            return []

        tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tools_module)

        tool_registry = getattr(tools_module, "TOOL_REGISTRY", {})
        return [tool_registry[name] for name in tool_names if name in tool_registry]
    except Exception:
        logger.exception("Could not load custom tools")
        return []
