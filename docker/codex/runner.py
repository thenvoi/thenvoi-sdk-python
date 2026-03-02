"""Production runner for Thenvoi Codex agents.

Reads agent configuration from a YAML file and runs the Codex adapter
with retry logic and graceful shutdown support.  Designed for Docker
deployment — all configuration is via environment variables.

Environment variables:
    AGENT_CONFIG               Path to the YAML config file (required)
    AGENT_KEY                  Key to look up in keyed config (default: "agent")
    CODEX_CWD                  Working directory for Codex (default: /workspace/repo)
    CODEX_TRANSPORT            Transport mode: stdio or ws (default: stdio)
    CODEX_MODEL                Model ID override
    CODEX_ROLE                 Role name; loads prompt from {config_dir}/prompts/{role}.md
    CODEX_SANDBOX              Sandbox mode (default: external-sandbox)
    CODEX_REASONING_EFFORT     Reasoning effort level
    CODEX_APPROVAL_MODE        Approval mode: manual, auto_accept, auto_decline
    CODEX_TURN_TASK_MARKERS    Emit turn task markers: true/false (default: false)
    THENVOI_WS_URL             Platform WebSocket URL
    THENVOI_REST_URL           Platform REST URL
    OPENAI_API_KEY             OpenAI API key
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

from repo_init import initialize_repo
from thenvoi.testing.runner_core import (
    build_runner_execution_contract,
    RunnerSpec,
    compose_runner_prompt,
    configure_runner_logging,
    create_runner_agent,
    create_shutdown_event,
    get_agent_key,
    get_lock_timeout_seconds,
    get_platform_urls,
    get_runner_config_path,
    log_repo_init_status,
    optional_text,
    parse_choice,
    read_bool_env,
    run_agent_lifecycle,
    run_runner_with_adapter,
    validate_required_mounts,
)
from thenvoi.testing.config_loader import load_runner_config

# Required mount points
REQUIRED_MOUNTS = [
    "/workspace/repo",
]

RUNNER_SPEC = RunnerSpec(
    required_mounts=tuple(REQUIRED_MOUNTS),
    mount_hint="Ensure docker-compose.yml mounts /workspace/repo.",
    agent_key_envs=("AGENT_KEY", "CODEX_AGENT_KEY"),
)

logger = configure_runner_logging(__name__)

CodexTransport = Literal["stdio", "ws"]
CodexApprovalMode = Literal["manual", "auto_accept", "auto_decline"]
CodexReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

_TRANSPORTS: dict[str, CodexTransport] = {"stdio": "stdio", "ws": "ws"}
_APPROVAL_MODES: dict[str, CodexApprovalMode] = {
    "manual": "manual",
    "auto_accept": "auto_accept",
    "auto_decline": "auto_decline",
}
_REASONING_EFFORTS: dict[str, CodexReasoningEffort] = {
    "none": "none",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}


def _env_bool(key: str, default: bool = False) -> bool:
    """Read a boolean from an environment variable."""
    return read_bool_env(key, default=default)


def _optional_str(value: Any) -> str | None:
    """Return a stripped string or None for empty/missing values."""
    return optional_text(value)


def _parse_transport(value: str) -> CodexTransport:
    return parse_choice(
        value,
        option_name="CODEX_TRANSPORT",
        allowed_values=_TRANSPORTS,
    )


def _parse_approval_mode(value: str) -> CodexApprovalMode:
    return parse_choice(
        value,
        option_name="CODEX_APPROVAL_MODE",
        allowed_values=_APPROVAL_MODES,
    )


def _parse_reasoning_effort(value: str | None) -> CodexReasoningEffort | None:
    if value is None:
        return None
    return parse_choice(
        value,
        option_name="CODEX_REASONING_EFFORT",
        allowed_values=_REASONING_EFFORTS,
    )


async def main() -> None:
    """Run the Codex agent from YAML configuration."""
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
    ).unwrap(operation="Codex docker runner bootstrap")
    config_path = context.bootstrap.config_path
    config = context.bootstrap.config
    repo_init = context.repo_init

    from thenvoi.adapters import CodexAdapter
    from thenvoi.adapters.codex.adapter import CodexAdapterConfig

    agent_id = config["agent_id"]

    # Codex-specific config from environment (env overrides YAML)
    codex_cwd = (
        _optional_str(os.environ.get("CODEX_CWD"))
        or _optional_str(config.get("workspace"))
        or _optional_str(getattr(repo_init, "repo_path", None))
        or "/workspace/repo"
    )
    codex_transport = _parse_transport(
        _optional_str(os.environ.get("CODEX_TRANSPORT")) or "stdio"
    )
    codex_model = _optional_str(os.environ.get("CODEX_MODEL")) or _optional_str(
        config.get("model")
    )
    codex_role = _optional_str(os.environ.get("CODEX_ROLE")) or _optional_str(
        config.get("role")
    )

    custom_section = compose_runner_prompt(
        config_path,
        role=codex_role,
        extra_sections=[getattr(repo_init, "context_bundle", "")],
        default_prompt="",
        logger=logger,
    )

    codex_sandbox = (
        _optional_str(os.environ.get("CODEX_SANDBOX"))
        or _optional_str(config.get("sandbox"))
        or "external-sandbox"
    )
    codex_reasoning = _parse_reasoning_effort(
        _optional_str(os.environ.get("CODEX_REASONING_EFFORT"))
        or _optional_str(config.get("reasoning_effort"))
    )
    codex_approval = _parse_approval_mode(
        _optional_str(os.environ.get("CODEX_APPROVAL_MODE"))
        or _optional_str(config.get("approval_mode"))
        or "manual"
    )
    codex_turn_markers = _env_bool("CODEX_TURN_TASK_MARKERS", default=False)

    adapter = CodexAdapter(
        config=CodexAdapterConfig(
            transport=codex_transport,
            cwd=codex_cwd,
            model=codex_model,
            personality="pragmatic",
            approval_policy="never",
            approval_mode=codex_approval,
            sandbox=codex_sandbox,
            reasoning_effort=codex_reasoning,
            codex_ws_url=os.environ.get("CODEX_WS_URL", "ws://127.0.0.1:8765"),
            custom_section=custom_section,
            include_base_instructions=True,
            enable_task_events=True,
            emit_turn_task_markers=codex_turn_markers,
            enable_execution_reporting=False,
            emit_thought_events=False,
            fallback_send_agent_text=True,
            experimental_api=True,
        )
    )

    def _log_startup() -> None:
        logger.info("Starting Codex agent: %s", agent_id)
        logger.info(
            "Codex config: transport=%s, model=%s, cwd=%s, sandbox=%s, role=%s",
            codex_transport,
            codex_model or "auto",
            codex_cwd,
            codex_sandbox,
            codex_role or "none",
        )
        log_repo_init_status(logger=logger, repo_init=repo_init)

    await run_runner_with_adapter(
        context,
        adapter=adapter,
        logger=logger,
        on_started=_log_startup,
        create_agent_fn=create_runner_agent,
        run_lifecycle_fn=run_agent_lifecycle,
    )


if __name__ == "__main__":
    asyncio.run(main())
