#!/usr/bin/env python3
"""
Healthcheck script for Thenvoi agent container.

Verifies:
1. Agent configuration is valid
2. REST API is reachable
3. Agent can authenticate with the platform

Exit codes:
- 0: Healthy
- 1: Unhealthy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


def main() -> int:
    """Run healthcheck and return exit code."""
    # Check config file exists and is valid
    config_path = os.environ.get("AGENT_CONFIG")
    if not config_path:
        print("AGENT_CONFIG not set")
        return 1

    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return 1

    if not config or not config.get("agent_id") or not config.get("api_key"):
        print("Missing agent_id or api_key in config")
        return 1

    # Check REST API connectivity
    rest_url = os.environ.get("THENVOI_REST_URL", "https://api.thenvoi.com")
    if not rest_url:
        print("THENVOI_REST_URL is empty")
        return 1

    try:
        import httpx

        # Verify agent can authenticate with the platform
        api_key = config["api_key"]
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{rest_url.rstrip('/')}/api/v1/agent/me",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if response.status_code == 200:
                print("Healthy: Agent authenticated successfully")
                return 0
            else:
                print(f"Unhealthy: API returned {response.status_code}")
                return 1
    except ImportError:
        # httpx not available, fall back to basic check
        print("Healthy: Config valid (httpx not available for API check)")
        return 0
    except Exception as e:
        print(f"Unhealthy: API check failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
