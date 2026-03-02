"""Tests for shared runner composition helpers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, call

import pytest

from thenvoi.testing.runner_core import (
    RunnerBootstrap,
    RunnerExecutionContext,
    RunnerSpec,
    build_claude_sdk_runner_artifacts,
    build_claude_sdk_runner_artifacts_contract,
    build_claude_sdk_runner_plan,
    build_runner_execution_contract,
    build_runner_execution_context,
    close_agent,
    compose_runner_prompt,
    log_claude_sdk_runner_startup,
    log_repo_init_status,
    load_runner_tools,
    run_runner_with_adapter,
)


def test_compose_runner_prompt_combines_role_prompt_custom_and_extras(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = config_dir / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "planner.md").write_text("role prompt", encoding="utf-8")
    config_path = config_dir / "agent.yaml"
    config_path.write_text("agent_id: x\napi_key: y\n", encoding="utf-8")

    prompt = compose_runner_prompt(
        str(config_path),
        role="planner",
        custom_prompt="custom prompt",
        extra_sections=["context bundle"],
        logger=logging.getLogger(__name__),
    )
    assert prompt == "role prompt\n\ncustom prompt\n\ncontext bundle"


def test_compose_runner_prompt_uses_default_when_no_sections(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "agent.yaml"
    config_path.write_text("agent_id: x\napi_key: y\n", encoding="utf-8")

    prompt = compose_runner_prompt(
        str(config_path),
        role=None,
        custom_prompt="",
        logger=logging.getLogger(__name__),
    )
    assert prompt == "You are a helpful assistant."


def test_compose_runner_prompt_wraps_prompt_read_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = config_dir / "prompts"
    prompts_dir.mkdir(parents=True)
    prompt_file = prompts_dir / "planner.md"
    prompt_file.write_text("role prompt", encoding="utf-8")
    config_path = config_dir / "agent.yaml"
    config_path.write_text("agent_id: x\napi_key: y\n", encoding="utf-8")

    original_read_text = Path.read_text

    def _failing_read_text(self: Path, *, encoding: str = "utf-8") -> str:
        if self == prompt_file:
            raise OSError("permission denied")
        return original_read_text(self, encoding=encoding)

    monkeypatch.setattr(Path, "read_text", _failing_read_text)

    with pytest.raises(ValueError, match="Failed to read prompt file"):
        compose_runner_prompt(
            str(config_path),
            role="planner",
            logger=logging.getLogger(__name__),
        )


def test_load_runner_tools_loads_only_requested_tools(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    tools_dir = config_dir / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "__init__.py").write_text(
        "def ping() -> str:\n"
        "    return 'pong'\n\n"
        "def pong() -> str:\n"
        "    return 'ping'\n\n"
        "TOOL_REGISTRY = {'ping': ping, 'pong': pong}\n",
        encoding="utf-8",
    )
    config_path = config_dir / "agent.yaml"
    config_path.write_text("agent_id: x\napi_key: y\n", encoding="utf-8")

    tools = load_runner_tools(
        str(config_path),
        ["ping"],
        logger=logging.getLogger(__name__),
    )

    assert len(tools) == 1
    assert tools[0].__name__ == "ping"


def test_load_runner_tools_no_tools_returns_empty(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "agent.yaml"
    config_path.write_text("agent_id: x\napi_key: y\n", encoding="utf-8")

    assert load_runner_tools(str(config_path), [], logger=logging.getLogger(__name__)) == []


def test_build_claude_sdk_runner_plan_applies_env_overrides(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "agent.yaml"
    config_path.write_text("agent_id: x\napi_key: y\n", encoding="utf-8")
    config = {
        "agent_id": "agent-1",
        "model": "config-model",
        "prompt": "prompt",
        "thinking_tokens": 123,
        "tools": [],
        "workspace": "/config/workspace",
        "role": "planner",
    }

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("WORKSPACE", "/env/workspace")
        monkeypatch.setenv("AGENT_ROLE", "reviewer")
        plan = build_claude_sdk_runner_plan(
            str(config_path),
            config,
            logger=logging.getLogger(__name__),
            workspace_fallback="/fallback",
        )

    assert plan.agent_id == "agent-1"
    assert plan.model == "config-model"
    assert plan.role == "reviewer"
    assert plan.workspace == "/env/workspace"
    assert plan.thinking_tokens == 123


def test_build_claude_sdk_runner_plan_rejects_non_list_tools(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "agent.yaml"
    config_path.write_text("agent_id: x\napi_key: y\n", encoding="utf-8")
    config = {"agent_id": "agent-1", "tools": "not-a-list"}

    with pytest.raises(ValueError, match="must be a list"):
        build_claude_sdk_runner_plan(
            str(config_path),
            config,
            logger=logging.getLogger(__name__),
        )


def test_build_runner_execution_context_applies_repo_initializer() -> None:
    logger = logging.getLogger(__name__)
    shutdown_event = asyncio.Event()
    bootstrap = RunnerBootstrap(
        config_path="agent.yml",
        agent_key="agent-key",
        config={"agent_id": "agent-1", "api_key": "api-key"},
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    create_shutdown_mock = MagicMock(return_value=shutdown_event)
    bootstrap_mock = MagicMock(return_value=bootstrap)
    load_config_mock = MagicMock()
    repo_init = SimpleNamespace(enabled=True)
    repo_initializer = MagicMock(return_value=repo_init)
    get_lock_timeout_mock = MagicMock(return_value=30.0)

    context = build_runner_execution_context(
        RunnerSpec(required_mounts=()),
        logger=logger,
        load_config=load_config_mock,
        create_shutdown_event_fn=create_shutdown_mock,
        bootstrap_runner_fn=bootstrap_mock,
        repo_initializer=repo_initializer,
        get_lock_timeout_fn=get_lock_timeout_mock,
    )

    create_shutdown_mock.assert_called_once_with(logger)
    bootstrap_mock.assert_called_once_with(
        RunnerSpec(required_mounts=()),
        logger=logger,
        load_config=load_config_mock,
        get_config_path=ANY,
        get_agent_key_fn=ANY,
        get_platform_urls_fn=ANY,
        validate_mounts_fn=ANY,
    )
    repo_initializer.assert_called_once_with(
        bootstrap.config,
        agent_key="agent-key",
        lock_timeout_s=30.0,
    )
    assert context.shutdown_event is shutdown_event
    assert context.bootstrap is bootstrap
    assert context.repo_init is repo_init


def test_build_claude_sdk_runner_artifacts_bundles_plan_and_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import thenvoi.testing.runner_core as runner_core_module

    bootstrap = RunnerBootstrap(
        config_path="agent.yml",
        agent_key="agent-key",
        config={"agent_id": "agent-1", "api_key": "api-key"},
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    context = RunnerExecutionContext(
        shutdown_event=asyncio.Event(),
        bootstrap=bootstrap,
    )
    plan = SimpleNamespace(agent_id="agent-1")
    adapter = object()
    plan_builder = MagicMock(return_value=plan)
    adapter_builder = MagicMock(return_value=adapter)
    monkeypatch.setattr(runner_core_module, "build_claude_sdk_runner_plan", plan_builder)
    monkeypatch.setattr(runner_core_module, "create_claude_sdk_adapter", adapter_builder)

    artifacts = build_claude_sdk_runner_artifacts(
        context,
        logger=logging.getLogger(__name__),
        workspace_fallback="/workspace",
        prompt_extra_sections=["context bundle"],
    )

    plan_builder.assert_called_once_with(
        "agent.yml",
        bootstrap.config,
        logger=ANY,
        workspace_fallback="/workspace",
        prompt_extra_sections=["context bundle"],
        default_model="claude-sonnet-4-5-20250929",
    )
    adapter_builder.assert_called_once_with(plan)
    assert artifacts.plan is plan
    assert artifacts.adapter is adapter


def test_build_runner_execution_contract_wraps_failures() -> None:
    result = build_runner_execution_contract(
        RunnerSpec(required_mounts=()),
        logger=logging.getLogger(__name__),
        load_config=MagicMock(),
        create_shutdown_event_fn=MagicMock(return_value=asyncio.Event()),
        bootstrap_runner_fn=MagicMock(side_effect=ValueError("bad bootstrap")),
    )

    assert result.is_ok is False
    assert result.error is not None
    assert result.error.code == "runner_bootstrap_failed"
    assert "bad bootstrap" in result.error.message


def test_build_claude_sdk_runner_artifacts_contract_wraps_failures() -> None:
    context = RunnerExecutionContext(
        shutdown_event=asyncio.Event(),
        bootstrap=RunnerBootstrap(
            config_path="agent.yml",
            agent_key="agent-key",
            config={},
            ws_url="wss://ws.example",
            rest_url="https://rest.example",
        ),
    )

    result = build_claude_sdk_runner_artifacts_contract(
        context,
        logger=logging.getLogger(__name__),
    )

    assert result.is_ok is False
    assert result.error is not None
    assert result.error.code == "claude_runner_artifacts_failed"
    assert "agent_id" in result.error.message


def test_log_repo_init_status_logs_only_when_enabled() -> None:
    logger = MagicMock()

    log_repo_init_status(logger=logger, repo_init=None)
    log_repo_init_status(logger=logger, repo_init=SimpleNamespace(enabled=False))
    logger.info.assert_not_called()

    repo_init = SimpleNamespace(
        enabled=True,
        cloned=True,
        indexed=False,
        repo_path="/workspace/repo",
    )
    log_repo_init_status(logger=logger, repo_init=repo_init)
    logger.info.assert_called_once_with(
        "Repo init: cloned=%s indexed=%s path=%s",
        True,
        False,
        "/workspace/repo",
    )


def test_log_claude_sdk_runner_startup_emits_config_summary() -> None:
    logger = MagicMock()
    plan = SimpleNamespace(
        agent_id="agent-1",
        model="claude-sonnet",
        role="planner",
        workspace="/tmp/workspace",
        thinking_tokens=2048,
    )

    startup_context = log_claude_sdk_runner_startup(
        logger=logger,
        plan=plan,
        startup_note="Press Ctrl+C to stop",
    )

    logger.info.assert_has_calls(
        [
            call("Starting agent: %s", "agent-1"),
            call("Model: %s", "claude-sonnet"),
            call("Role: %s", "planner"),
            call("Workspace: %s", "/tmp/workspace"),
            call("Extended thinking is enabled"),
            call("Press Ctrl+C to stop"),
        ]
    )
    assert startup_context == {
        "agent_id": "agent-1",
        "model": "claude-sonnet",
        "role": "planner",
        "workspace": "/tmp/workspace",
        "thinking_tokens": 2048,
        "startup_note": "Press Ctrl+C to stop",
    }


@pytest.mark.asyncio
async def test_run_runner_with_adapter_uses_injected_hooks() -> None:
    shutdown_event = asyncio.Event()
    bootstrap = RunnerBootstrap(
        config_path="agent.yml",
        agent_key="agent-key",
        config={"agent_id": "agent-1", "api_key": "api-key"},
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    context = RunnerExecutionContext(
        shutdown_event=shutdown_event,
        bootstrap=bootstrap,
    )
    adapter = object()
    agent = object()
    on_started = MagicMock()
    create_agent_mock = MagicMock(return_value=agent)
    run_lifecycle_mock = AsyncMock()
    logger = logging.getLogger(__name__)

    await run_runner_with_adapter(
        context,
        adapter=adapter,
        logger=logger,
        on_started=on_started,
        create_agent_fn=create_agent_mock,
        run_lifecycle_fn=run_lifecycle_mock,
    )

    create_agent_mock.assert_called_once_with(
        adapter=adapter,
        config=bootstrap.config,
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    on_started.assert_called_once_with()
    run_lifecycle_mock.assert_awaited_once_with(
        agent,
        shutdown_event,
        logger=logger,
    )


@pytest.mark.asyncio
async def test_close_agent_returns_true_without_close_method() -> None:
    agent = object()
    result = await close_agent(agent, logger=logging.getLogger(__name__))
    assert result is True


@pytest.mark.asyncio
async def test_close_agent_returns_false_when_close_raises() -> None:
    agent = MagicMock()
    agent.close = AsyncMock(side_effect=RuntimeError("cleanup failed"))

    result = await close_agent(agent, logger=logging.getLogger(__name__))

    assert result is False
