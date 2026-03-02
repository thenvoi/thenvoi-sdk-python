from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from thenvoi.core.seams import BoundaryResult
from thenvoi.config.defaults import DEFAULT_REST_URL, DEFAULT_WS_URL

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ChoiceT = TypeVar("ChoiceT", bound=str)


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for long-running agent runners."""

    max_retries: int = 5
    initial_delay_s: float = 1.0
    max_delay_s: float = 60.0


@dataclass(frozen=True)
class RunnerSpec:
    """Declarative startup contract for runner entrypoints."""

    required_mounts: tuple[str, ...]
    mount_hint: str | None = None
    agent_key_envs: tuple[str, ...] = ("AGENT_KEY",)
    agent_key_default: str = "agent"


@dataclass(frozen=True)
class RunnerBootstrap:
    """Common runner bootstrap output shared by entrypoints."""

    config_path: str
    agent_key: str
    config: dict[str, Any]
    ws_url: str
    rest_url: str


@dataclass(frozen=True)
class ClaudeSDKRunnerPlan:
    """Normalized Claude SDK runner settings derived from env + config."""

    agent_id: str
    model: str
    role: str | None
    workspace: str | None
    thinking_tokens: int | None
    final_prompt: str
    custom_tools: list[Any]


class RepoInitializer(Protocol):
    """Callable contract for optional repo bootstrap hooks."""

    def __call__(
        self,
        config: dict[str, Any],
        *,
        agent_key: str,
        lock_timeout_s: float,
    ) -> Any:
        """Initialize repository state for runner startup."""


class RunnerAgentFactory(Protocol):
    """Factory contract for creating runner Agent instances."""

    def __call__(
        self,
        *,
        adapter: Any,
        config: dict[str, Any],
        ws_url: str,
        rest_url: str,
    ) -> Any:
        """Create and return a runner Agent instance."""


class RunnerLifecycleFn(Protocol):
    """Async runner lifecycle contract used by entrypoint templates."""

    def __call__(
        self,
        agent: Any,
        shutdown_event: asyncio.Event,
        *,
        logger: logging.Logger,
    ) -> Awaitable[None]:
        """Run the agent lifecycle until completion or shutdown."""


@dataclass(frozen=True)
class RunnerExecutionContext:
    """Shared runtime context for declarative runner entrypoints."""

    shutdown_event: asyncio.Event
    bootstrap: RunnerBootstrap
    repo_init: Any | None = None


@dataclass(frozen=True)
class ClaudeSDKRunnerArtifacts:
    """Bundled Claude SDK plan + adapter produced by shared builder."""

    plan: ClaudeSDKRunnerPlan
    adapter: Any


def configure_runner_logging(module_name: str) -> logging.Logger:
    """Set standard runner logging format and return module logger."""
    logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
    return logging.getLogger(module_name)


def get_runner_config_path(env_var: str = "AGENT_CONFIG") -> str:
    """Read and validate the config path environment variable."""
    config_path = os.environ.get(env_var)
    if not config_path:
        raise ValueError(f"{env_var} environment variable not set")
    return config_path


def get_agent_key(*env_names: str, default: str = "agent") -> str:
    """Read the first present agent key environment variable."""
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    return default


def optional_text(value: Any) -> str | None:
    """Normalize optional config/env strings by trimming empty values to None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_bool_env(key: str, *, default: bool = False) -> bool:
    """Read common boolean env values with a consistent parser."""
    value = optional_text(os.environ.get(key))
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_choice(
    value: str,
    *,
    option_name: str,
    allowed_values: Mapping[str, ChoiceT],
) -> ChoiceT:
    """Parse and normalize constrained string options with shared errors."""
    parsed = allowed_values.get(value.lower())
    if parsed is None:
        raise ValueError(
            f"{option_name} must be one of {', '.join(sorted(allowed_values))}; got: {value}"
        )
    return parsed


def get_platform_urls(
    *,
    ws_default: str = DEFAULT_WS_URL,
    rest_default: str = DEFAULT_REST_URL,
) -> tuple[str, str]:
    """Return validated Thenvoi platform endpoints."""
    ws_url = os.environ.get("THENVOI_WS_URL", ws_default)
    rest_url = os.environ.get("THENVOI_REST_URL", rest_default)
    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is empty")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is empty")
    return ws_url, rest_url


def validate_required_mounts(
    required_mounts: Sequence[str],
    *,
    mount_hint: str | None = None,
) -> None:
    """Validate required mount directories exist."""
    missing = [path for path in required_mounts if not Path(path).is_dir()]
    if missing:
        detail = f"Missing required mount points: {missing}."
        if mount_hint:
            detail = f"{detail} {mount_hint}"
        raise ValueError(detail)


def bootstrap_runner(
    spec: RunnerSpec,
    *,
    logger: logging.Logger,
    load_config: Callable[[str, str], dict[str, Any]],
    get_config_path: Callable[[], str] = get_runner_config_path,
    get_agent_key_fn: Callable[..., str] = get_agent_key,
    get_platform_urls_fn: Callable[[], tuple[str, str]] = get_platform_urls,
    validate_mounts_fn: Callable[..., None] = validate_required_mounts,
) -> RunnerBootstrap:
    """Run shared startup flow and return normalized runner context."""
    config_path = get_config_path()
    agent_key = get_agent_key_fn(*spec.agent_key_envs, default=spec.agent_key_default)
    ws_url, rest_url = get_platform_urls_fn()
    validate_mounts_fn(spec.required_mounts, mount_hint=spec.mount_hint)

    logger.info("Loading config from: %s (key: %s)", config_path, agent_key)
    config = load_config(config_path, agent_key)
    return RunnerBootstrap(
        config_path=config_path,
        agent_key=agent_key,
        config=config,
        ws_url=ws_url,
        rest_url=rest_url,
    )


def build_runner_execution_context(
    spec: RunnerSpec,
    *,
    logger: logging.Logger,
    load_config: Callable[[str, str], dict[str, Any]],
    get_config_path: Callable[[], str] = get_runner_config_path,
    get_agent_key_fn: Callable[..., str] = get_agent_key,
    get_platform_urls_fn: Callable[[], tuple[str, str]] = get_platform_urls,
    validate_mounts_fn: Callable[..., None] = validate_required_mounts,
    create_shutdown_event_fn: Callable[[logging.Logger], asyncio.Event] | None = None,
    bootstrap_runner_fn: Callable[..., RunnerBootstrap] | None = None,
    repo_initializer: RepoInitializer | None = None,
    get_lock_timeout_fn: Callable[[], float] | None = None,
) -> RunnerExecutionContext:
    """Build shared runner execution context from declarative hooks."""
    create_shutdown = create_shutdown_event_fn or create_shutdown_event
    bootstrap_fn = bootstrap_runner_fn or bootstrap_runner
    lock_timeout_fn = get_lock_timeout_fn or get_lock_timeout_seconds

    shutdown_event = create_shutdown(logger)
    bootstrap = bootstrap_fn(
        spec,
        logger=logger,
        load_config=load_config,
        get_config_path=get_config_path,
        get_agent_key_fn=get_agent_key_fn,
        get_platform_urls_fn=get_platform_urls_fn,
        validate_mounts_fn=validate_mounts_fn,
    )

    repo_init: Any | None = None
    if repo_initializer is not None:
        repo_init = repo_initializer(
            bootstrap.config,
            agent_key=bootstrap.agent_key,
            lock_timeout_s=lock_timeout_fn(),
        )

    return RunnerExecutionContext(
        shutdown_event=shutdown_event,
        bootstrap=bootstrap,
        repo_init=repo_init,
    )


def build_runner_execution_contract(
    spec: RunnerSpec,
    *,
    logger: logging.Logger,
    load_config: Callable[[str, str], dict[str, Any]],
    get_config_path: Callable[[], str] = get_runner_config_path,
    get_agent_key_fn: Callable[..., str] = get_agent_key,
    get_platform_urls_fn: Callable[[], tuple[str, str]] = get_platform_urls,
    validate_mounts_fn: Callable[..., None] = validate_required_mounts,
    create_shutdown_event_fn: Callable[[logging.Logger], asyncio.Event] | None = None,
    bootstrap_runner_fn: Callable[..., RunnerBootstrap] | None = None,
    repo_initializer: RepoInitializer | None = None,
    get_lock_timeout_fn: Callable[[], float] | None = None,
) -> BoundaryResult[RunnerExecutionContext]:
    """Build typed boundary result for runner bootstrap orchestration."""
    try:
        context = build_runner_execution_context(
            spec,
            logger=logger,
            load_config=load_config,
            get_config_path=get_config_path,
            get_agent_key_fn=get_agent_key_fn,
            get_platform_urls_fn=get_platform_urls_fn,
            validate_mounts_fn=validate_mounts_fn,
            create_shutdown_event_fn=create_shutdown_event_fn,
            bootstrap_runner_fn=bootstrap_runner_fn,
            repo_initializer=repo_initializer,
            get_lock_timeout_fn=get_lock_timeout_fn,
        )
    except Exception as exc:
        return BoundaryResult.failure(
            code="runner_bootstrap_failed",
            message=str(exc),
            details={"exception_type": type(exc).__name__},
        )
    return BoundaryResult.success(context)


def get_lock_timeout_seconds(
    env_var: str = "REPO_INIT_LOCK_TIMEOUT_S",
    *,
    default: float = 120.0,
) -> float:
    """Read lock timeout env var with shared defaulting."""
    value = optional_text(os.environ.get(env_var))
    if value is None:
        return default
    return float(value)


def compose_runner_prompt(
    config_path: str,
    *,
    role: str | None,
    custom_prompt: str = "",
    extra_sections: Sequence[str | None] = (),
    default_prompt: str = "You are a helpful assistant.",
    logger: logging.Logger,
) -> str:
    """Build final runner prompt from role file, config prompt, and extras."""
    config_dir = Path(config_path).parent
    prompt_dir = config_dir / "prompts"
    prompt_parts: list[str] = []

    if role:
        prompt_file = prompt_dir / f"{role}.md"
        if prompt_file.exists():
            try:
                prompt_parts.append(prompt_file.read_text(encoding="utf-8"))
            except OSError as exc:
                raise ValueError(
                    f"Failed to read prompt file '{prompt_file}': {exc}"
                ) from exc
            logger.info("Using role prompt from: %s", prompt_file)
        else:
            logger.warning(
                "Role '%s' specified but no prompt file at %s",
                role,
                prompt_file,
            )

    if custom_prompt:
        prompt_parts.append(custom_prompt)

    prompt_parts.extend(section for section in extra_sections if section)

    if not prompt_parts and default_prompt:
        prompt_parts.append(default_prompt)

    return "\n\n".join(prompt_parts)


def load_runner_tools(
    config_path: str,
    tool_names: Sequence[str],
    *,
    logger: logging.Logger,
) -> list[Any]:
    """Load configured tools from the runner-local tools registry."""
    if not tool_names:
        return []

    from thenvoi.testing.tool_loading import load_custom_tools

    config_dir = Path(config_path).parent
    custom_tools = load_custom_tools(
        config_dir / "tools",
        config_dir,
        tool_names,
        logger=logger,
    )
    if custom_tools:
        tool_fn_names = [
            getattr(tool, "_tool_name", tool.__name__) for tool in custom_tools
        ]
        logger.info("Loaded custom tools: %s", tool_fn_names)
    return custom_tools


def build_claude_sdk_runner_plan(
    config_path: str,
    config: Mapping[str, Any],
    *,
    logger: logging.Logger,
    workspace_fallback: str | None = None,
    prompt_extra_sections: Sequence[str | None] = (),
    default_model: str = "claude-sonnet-4-5-20250929",
) -> ClaudeSDKRunnerPlan:
    """Build a reusable Claude SDK runner plan from shared config/env rules."""
    model = optional_text(config.get("model")) or default_model
    custom_prompt = optional_text(config.get("prompt")) or ""
    thinking_tokens = config.get("thinking_tokens")
    raw_tool_names = config.get("tools", [])
    if raw_tool_names is None:
        tool_names: list[str] = []
    elif isinstance(raw_tool_names, Sequence) and not isinstance(raw_tool_names, str):
        tool_names = [str(name) for name in raw_tool_names]
    else:
        raise ValueError("Runner config field 'tools' must be a list of tool names")

    workspace = (
        optional_text(os.environ.get("WORKSPACE"))
        or optional_text(config.get("workspace"))
        or workspace_fallback
    )
    role = optional_text(os.environ.get("AGENT_ROLE")) or optional_text(
        config.get("role")
    )
    final_prompt = compose_runner_prompt(
        config_path,
        role=role,
        custom_prompt=custom_prompt,
        extra_sections=prompt_extra_sections,
        logger=logger,
    )
    custom_tools = load_runner_tools(config_path, tool_names, logger=logger)

    agent_id = optional_text(config.get("agent_id"))
    if agent_id is None:
        raise ValueError("Runner config is missing required field 'agent_id'")

    return ClaudeSDKRunnerPlan(
        agent_id=agent_id,
        model=model,
        role=role,
        workspace=workspace,
        thinking_tokens=thinking_tokens,
        final_prompt=final_prompt,
        custom_tools=custom_tools,
    )


def create_claude_sdk_adapter(plan: ClaudeSDKRunnerPlan) -> Any:
    """Create a ClaudeSDKAdapter from a normalized runner plan."""
    from thenvoi.adapters import ClaudeSDKAdapter

    return ClaudeSDKAdapter(
        model=plan.model,
        custom_section=plan.final_prompt,
        max_thinking_tokens=plan.thinking_tokens,
        enable_execution_reporting=True,
        additional_tools=plan.custom_tools if plan.custom_tools else None,
        cwd=plan.workspace,
    )


def build_claude_sdk_runner_artifacts(
    context: RunnerExecutionContext,
    *,
    logger: logging.Logger,
    workspace_fallback: str | None = None,
    prompt_extra_sections: Sequence[str | None] = (),
    default_model: str = "claude-sonnet-4-5-20250929",
) -> ClaudeSDKRunnerArtifacts:
    """Build Claude runner plan and adapter from a shared entrypoint context."""
    plan = build_claude_sdk_runner_plan(
        context.bootstrap.config_path,
        context.bootstrap.config,
        logger=logger,
        workspace_fallback=workspace_fallback,
        prompt_extra_sections=prompt_extra_sections,
        default_model=default_model,
    )
    return ClaudeSDKRunnerArtifacts(
        plan=plan,
        adapter=create_claude_sdk_adapter(plan),
    )


def build_claude_sdk_runner_artifacts_contract(
    context: RunnerExecutionContext,
    *,
    logger: logging.Logger,
    workspace_fallback: str | None = None,
    prompt_extra_sections: Sequence[str | None] = (),
    default_model: str = "claude-sonnet-4-5-20250929",
) -> BoundaryResult[ClaudeSDKRunnerArtifacts]:
    """Build typed boundary result for Claude runner seam handoff."""
    try:
        artifacts = build_claude_sdk_runner_artifacts(
            context,
            logger=logger,
            workspace_fallback=workspace_fallback,
            prompt_extra_sections=prompt_extra_sections,
            default_model=default_model,
        )
    except Exception as exc:
        return BoundaryResult.failure(
            code="claude_runner_artifacts_failed",
            message=str(exc),
            details={"exception_type": type(exc).__name__},
        )
    return BoundaryResult.success(artifacts)


def log_claude_sdk_runner_startup(
    *,
    logger: logging.Logger,
    plan: ClaudeSDKRunnerPlan,
    startup_note: str | None = None,
) -> dict[str, Any]:
    """Emit consistent startup logs for Claude SDK runner entrypoints.

    Returns the startup context payload that was emitted for structured callers.
    """
    startup_context: dict[str, Any] = {
        "agent_id": plan.agent_id,
        "model": plan.model,
        "role": plan.role,
        "workspace": plan.workspace,
        "thinking_tokens": plan.thinking_tokens,
        "startup_note": startup_note,
    }

    logger.info("Starting agent: %s", plan.agent_id)
    logger.info("Model: %s", plan.model)
    if plan.role:
        logger.info("Role: %s", plan.role)
    if plan.workspace:
        logger.info("Workspace: %s", plan.workspace)
    if plan.thinking_tokens:
        logger.info("Extended thinking is enabled")
    if startup_note:
        logger.info(startup_note)

    return startup_context


def log_repo_init_status(*, logger: logging.Logger, repo_init: Any | None) -> None:
    """Emit a consistent repo bootstrap status log when enabled."""
    if repo_init is None or not getattr(repo_init, "enabled", False):
        return

    logger.info(
        "Repo init: cloned=%s indexed=%s path=%s",
        getattr(repo_init, "cloned", False),
        getattr(repo_init, "indexed", False),
        getattr(repo_init, "repo_path", None),
    )


def create_runner_agent(
    *,
    adapter: Any,
    config: dict[str, Any],
    ws_url: str,
    rest_url: str,
) -> Any:
    """Create an Agent from normalized runner config and adapter."""
    from thenvoi import Agent

    return Agent.create(
        adapter=adapter,
        agent_id=config["agent_id"],
        api_key=config["api_key"],
        ws_url=ws_url,
        rest_url=rest_url,
    )


async def run_runner_with_adapter(
    context: RunnerExecutionContext,
    *,
    adapter: Any,
    logger: logging.Logger,
    on_started: Callable[[], None] | None = None,
    create_agent_fn: RunnerAgentFactory | None = None,
    run_lifecycle_fn: RunnerLifecycleFn | None = None,
) -> None:
    """Create agent from context + adapter, then run shared lifecycle."""
    create_agent = create_agent_fn or create_runner_agent
    run_lifecycle = run_lifecycle_fn or run_agent_lifecycle

    agent = create_agent(
        adapter=adapter,
        config=context.bootstrap.config,
        ws_url=context.bootstrap.ws_url,
        rest_url=context.bootstrap.rest_url,
    )
    if on_started is not None:
        on_started()

    await run_lifecycle(agent, context.shutdown_event, logger=logger)


def create_shutdown_event(logger: logging.Logger) -> asyncio.Event:
    """Create shutdown event and install SIGTERM/SIGINT handlers."""
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal(sig: signal.Signals) -> None:
        logger.info("Received %s, initiating graceful shutdown...", sig.name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig)

    return shutdown_event


async def run_agent_with_retry(
    agent: Any,
    shutdown_event: asyncio.Event,
    *,
    logger: logging.Logger,
    retry_config: RetryConfig = RetryConfig(),
) -> None:
    """Run an agent task with shutdown handling and retry backoff."""
    retry_count = 0
    retry_delay = retry_config.initial_delay_s

    while not shutdown_event.is_set():
        try:
            agent_task = asyncio.create_task(agent.run())
            shutdown_task = asyncio.create_task(shutdown_event.wait())

            done, pending = await asyncio.wait(
                [agent_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                result = (await asyncio.gather(task, return_exceptions=True))[0]
                if isinstance(result, asyncio.CancelledError):
                    logger.debug("Cancelled pending runner task during shutdown")
                elif isinstance(result, Exception):
                    logger.warning(
                        "Pending runner task exited with error during shutdown: %s",
                        result,
                    )

            if agent_task in done:
                agent_task.result()

            retry_count = 0
            retry_delay = retry_config.initial_delay_s
            break

        except (ConnectionError, OSError) as exc:
            retry_count += 1
            if retry_count > retry_config.max_retries:
                logger.error(
                    "Max retries (%s) exceeded, giving up",
                    retry_config.max_retries,
                )
                raise

            logger.warning(
                "Connection issue detected (%s). Retrying in %.1fs (attempt %s/%s)",
                exc,
                retry_delay,
                retry_count,
                retry_config.max_retries,
            )
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, retry_config.max_delay_s)

        except asyncio.CancelledError:
            logger.info("Agent task cancelled")
            break


async def close_agent(agent: Any, *, logger: logging.Logger) -> bool:
    """Close an agent if it supports async close()."""
    try:
        if hasattr(agent, "close"):
            await agent.close()
        return True
    except Exception as error:
        logger.exception("Error during agent cleanup: %s", error)
        return False


async def run_agent_lifecycle(
    agent: Any,
    shutdown_event: asyncio.Event,
    *,
    logger: logging.Logger,
) -> None:
    """Run agent with retry policy and graceful close logging."""
    await run_agent_with_retry(agent, shutdown_event, logger=logger)
    logger.info("Shutting down...")
    if not await close_agent(agent, logger=logger):
        logger.warning("Agent cleanup completed with errors")
    logger.info("Agent stopped")
