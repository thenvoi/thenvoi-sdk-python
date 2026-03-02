#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[claude_sdk]", "pyyaml"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
YAML-based agent runner for Thenvoi Claude SDK.

Reads agent configuration from a YAML file and runs the agent.
Designed for Docker deployment without writing Python code.

Usage:
    AGENT_CONFIG=/app/config/agent.yaml python runner.py
"""

from __future__ import annotations

import asyncio

from thenvoi.testing.runner_core import (
    build_claude_sdk_runner_artifacts_contract,
    build_runner_execution_contract,
    RunnerSpec,
    configure_runner_logging,
    create_runner_agent,
    create_shutdown_event,
    get_agent_key,
    get_platform_urls,
    get_runner_config_path,
    log_claude_sdk_runner_startup,
    run_agent_lifecycle,
    run_runner_with_adapter,
    validate_required_mounts,
)
from thenvoi.testing.config_loader import load_runner_config

# Required mount points per SRS NFR-007
REQUIRED_MOUNTS = [
    "/workspace/repo",
    "/workspace/notes",
    "/workspace/state",
]

RUNNER_SPEC = RunnerSpec(
    required_mounts=tuple(REQUIRED_MOUNTS),
    mount_hint=(
        "Ensure docker-compose.yml mounts: "
        f"{', '.join(f'{mount} (rw)' for mount in REQUIRED_MOUNTS)}. "
        "See README.md for mount contract details."
    ),
)

logger = configure_runner_logging(__name__)


async def main() -> None:
    """Run the agent from YAML configuration."""
    context = build_runner_execution_contract(
        RUNNER_SPEC,
        logger=logger,
        load_config=load_runner_config,
        create_shutdown_event_fn=create_shutdown_event,
        get_config_path=get_runner_config_path,
        get_agent_key_fn=get_agent_key,
        get_platform_urls_fn=get_platform_urls,
        validate_mounts_fn=validate_required_mounts,
    ).unwrap(operation="Claude SDK example runner bootstrap")
    artifacts = build_claude_sdk_runner_artifacts_contract(
        context,
        logger=logger,
    ).unwrap(operation="Claude SDK example runner artifact build")

    def _log_startup() -> None:
        log_claude_sdk_runner_startup(
            logger=logger,
            plan=artifacts.plan,
            startup_note="Press Ctrl+C to stop",
        )

    await run_runner_with_adapter(
        context,
        adapter=artifacts.adapter,
        logger=logger,
        on_started=_log_startup,
        create_agent_fn=create_runner_agent,
        run_lifecycle_fn=run_agent_lifecycle,
    )


if __name__ == "__main__":
    asyncio.run(main())
