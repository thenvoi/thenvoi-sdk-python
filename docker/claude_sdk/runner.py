"""Production runner for Thenvoi Claude SDK agents.

Reads agent configuration from a YAML file and runs the agent with
retry logic and graceful shutdown support.  Designed for Docker
deployment — all configuration is via environment variables.

Environment variables:
    AGENT_CONFIG   Path to the YAML config file (required)
    AGENT_KEY      Key to look up in keyed config (default: "agent")
    AGENT_ROLE     Role override (planner, reviewer)
    WORKSPACE      Working directory override
    THENVOI_WS_URL     Platform WebSocket URL
    THENVOI_REST_URL   Platform REST URL
    ANTHROPIC_API_KEY  Anthropic API key
"""

from __future__ import annotations

import asyncio

from repo_init import initialize_repo
from thenvoi.testing.runner_core import (
    build_claude_sdk_runner_artifacts_contract,
    build_runner_execution_contract,
    RunnerSpec,
    configure_runner_logging,
    create_runner_agent,
    create_shutdown_event,
    get_agent_key,
    get_lock_timeout_seconds,
    get_platform_urls,
    get_runner_config_path,
    log_claude_sdk_runner_startup,
    log_repo_init_status,
    run_agent_lifecycle,
    run_runner_with_adapter,
    validate_required_mounts,
)
from thenvoi.testing.config_loader import load_runner_config

# Required mount points
REQUIRED_MOUNTS = [
    "/workspace/repo",
    "/workspace/notes",
    "/workspace/state",
    "/workspace/context",
]

RUNNER_SPEC = RunnerSpec(
    required_mounts=tuple(REQUIRED_MOUNTS),
    mount_hint=(
        "Ensure docker-compose.yml mounts: "
        f"{', '.join(f'{mount} (rw)' for mount in REQUIRED_MOUNTS)}."
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
        repo_initializer=initialize_repo,
        get_lock_timeout_fn=get_lock_timeout_seconds,
    ).unwrap(operation="Claude SDK docker runner bootstrap")

    repo_path = getattr(context.repo_init, "repo_path", None)
    context_bundle = getattr(context.repo_init, "context_bundle", "")
    artifacts = build_claude_sdk_runner_artifacts_contract(
        context,
        logger=logger,
        workspace_fallback=repo_path,
        prompt_extra_sections=[context_bundle],
    ).unwrap(operation="Claude SDK docker runner artifact build")

    def _log_startup() -> None:
        log_claude_sdk_runner_startup(logger=logger, plan=artifacts.plan)
        log_repo_init_status(logger=logger, repo_init=context.repo_init)

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
