"""
Agent configuration management utilities.

This module provides functions to load agent credentials from a YAML
configuration file at the project root. Agents must be created manually
on the platform as external agents before use.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def get_config_path() -> Path:
    """
    Get the path to the agent configuration file.

    Looks for agent_config.yaml in the current working directory (project root).
    """
    return Path(os.getcwd()) / "agent_config.yaml"


def load_agent_config(agent_key: str) -> Tuple[str, str]:
    """
    Load agent credentials from YAML file at project root.

    Agents must be created manually on the platform as external agents.
    This function loads the agent_id and api_key for validation.

    Args:
        agent_key: The key identifying the agent in the config file

    Returns:
        Tuple of (agent_id, api_key)

    Raises:
        FileNotFoundError: If agent_config.yaml doesn't exist
        ValueError: If required fields (agent_id, api_key) are missing or empty
    """
    config_path = get_config_path()
    logger.debug(f"Loading config from: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(
            f"agent_config.yaml not found at {config_path}. "
            "Copy agent_config.yaml.example to agent_config.yaml and configure your agents."
        )

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        agent_config = config.get(agent_key, {})

        if not agent_config:
            raise ValueError(
                f"Agent '{agent_key}' not found in {config_path}. "
                f"Please add the agent configuration."
            )

        # Require agent_id and api_key
        agent_id = agent_config.get("agent_id")
        api_key = agent_config.get("api_key")

        missing_fields = []
        if not agent_id:
            missing_fields.append("agent_id")
        if not api_key:
            missing_fields.append("api_key")

        if missing_fields:
            raise ValueError(
                f"Missing required fields for agent '{agent_key}': {', '.join(missing_fields)}. "
                f"Please create an external agent on the platform and add the credentials to {config_path}"
            )

        return agent_id, api_key
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading agent config: {e}")
