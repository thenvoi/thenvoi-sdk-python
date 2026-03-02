"""Unit tests for docker/claude_sdk/runner.py."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest


def _load_claude_runner_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    repo_init_stub = ModuleType("repo_init")
    repo_init_stub.initialize_repo = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "repo_init", repo_init_stub)

    module_name = "docker.claude_sdk.runner"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_main_uses_env_overrides_and_custom_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_claude_runner_module(monkeypatch)

    shutdown_event = asyncio.Event()
    logger = MagicMock()
    repo_init = SimpleNamespace(
        repo_path="/repo/path",
        context_bundle="repo-context",
        enabled=True,
        cloned=True,
        indexed=False,
    )
    config = {"agent_id": "agent-123"}
    plan = SimpleNamespace(
        agent_id="agent-123",
        model="claude-test-model",
        role="reviewer",
        workspace="/env/workspace",
        thinking_tokens=4096,
        final_prompt="final-prompt",
        custom_tools=[{"name": "weather_lookup"}],
    )
    adapter_instance = object()
    artifacts_contract = MagicMock()
    artifacts_contract.unwrap.return_value = SimpleNamespace(
        plan=plan,
        adapter=adapter_instance,
    )
    agent_instance = object()
    mock_run_lifecycle = AsyncMock()

    monkeypatch.setenv("WORKSPACE", "/env/workspace")
    monkeypatch.setenv("AGENT_ROLE", "reviewer")
    monkeypatch.setenv("REPO_INIT_LOCK_TIMEOUT_S", "15")

    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module, "create_shutdown_event", MagicMock(return_value=shutdown_event))
    monkeypatch.setattr(module, "get_runner_config_path", MagicMock(return_value="agent.yml"))
    monkeypatch.setattr(module, "get_agent_key", MagicMock(return_value="agent-key"))
    monkeypatch.setattr(
        module,
        "get_platform_urls",
        MagicMock(return_value=("wss://ws.example", "https://rest.example")),
    )
    monkeypatch.setattr(module, "validate_required_mounts", MagicMock())
    monkeypatch.setattr(module, "load_runner_config", MagicMock(return_value=config))
    monkeypatch.setattr(module, "initialize_repo", MagicMock(return_value=repo_init))
    monkeypatch.setattr(
        module,
        "build_claude_sdk_runner_artifacts_contract",
        MagicMock(return_value=artifacts_contract),
    )
    monkeypatch.setattr(module, "log_claude_sdk_runner_startup", MagicMock())
    monkeypatch.setattr(module, "create_runner_agent", MagicMock(return_value=agent_instance))
    monkeypatch.setattr(module, "run_agent_lifecycle", mock_run_lifecycle)

    await module.main()

    module.get_agent_key.assert_called_once_with("AGENT_KEY", default="agent")
    module.validate_required_mounts.assert_called_once()
    module.initialize_repo.assert_called_once_with(
        config,
        agent_key="agent-key",
        lock_timeout_s=15.0,
    )
    module.build_claude_sdk_runner_artifacts_contract.assert_called_once_with(
        ANY,
        logger=logger,
        workspace_fallback="/repo/path",
        prompt_extra_sections=["repo-context"],
    )
    artifacts_contract.unwrap.assert_called_once_with(
        operation="Claude SDK docker runner artifact build"
    )
    module.create_runner_agent.assert_called_once_with(
        adapter=adapter_instance,
        config=config,
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    module.log_claude_sdk_runner_startup.assert_called_once_with(
        logger=logger,
        plan=plan,
    )
    mock_run_lifecycle.assert_awaited_once_with(agent_instance, shutdown_event, logger=logger)


@pytest.mark.asyncio
async def test_main_falls_back_to_defaults_without_optional_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_claude_runner_module(monkeypatch)

    shutdown_event = asyncio.Event()
    logger = MagicMock()
    repo_init = SimpleNamespace(
        repo_path="/repo/default",
        context_bundle="",
        enabled=False,
        cloned=False,
        indexed=False,
    )
    config = {"agent_id": "agent-456"}
    plan = SimpleNamespace(
        agent_id="agent-456",
        model="claude-sonnet-4-5-20250929",
        role=None,
        workspace="/repo/default",
        thinking_tokens=None,
        final_prompt="final-prompt",
        custom_tools=[],
    )
    adapter_instance = object()
    artifacts_contract = MagicMock()
    artifacts_contract.unwrap.return_value = SimpleNamespace(
        plan=plan,
        adapter=adapter_instance,
    )
    agent_instance = object()
    mock_run_lifecycle = AsyncMock()

    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.delenv("AGENT_ROLE", raising=False)
    monkeypatch.delenv("REPO_INIT_LOCK_TIMEOUT_S", raising=False)

    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module, "create_shutdown_event", MagicMock(return_value=shutdown_event))
    monkeypatch.setattr(module, "get_runner_config_path", MagicMock(return_value="agent.yml"))
    monkeypatch.setattr(module, "get_agent_key", MagicMock(return_value="agent-key"))
    monkeypatch.setattr(
        module,
        "get_platform_urls",
        MagicMock(return_value=("wss://ws.example", "https://rest.example")),
    )
    monkeypatch.setattr(module, "validate_required_mounts", MagicMock())
    monkeypatch.setattr(module, "load_runner_config", MagicMock(return_value=config))
    monkeypatch.setattr(module, "initialize_repo", MagicMock(return_value=repo_init))
    monkeypatch.setattr(
        module,
        "build_claude_sdk_runner_artifacts_contract",
        MagicMock(return_value=artifacts_contract),
    )
    monkeypatch.setattr(module, "log_claude_sdk_runner_startup", MagicMock())
    monkeypatch.setattr(module, "create_runner_agent", MagicMock(return_value=agent_instance))
    monkeypatch.setattr(module, "run_agent_lifecycle", mock_run_lifecycle)

    await module.main()

    module.initialize_repo.assert_called_once_with(
        config,
        agent_key="agent-key",
        lock_timeout_s=120.0,
    )
    module.build_claude_sdk_runner_artifacts_contract.assert_called_once_with(
        ANY,
        logger=logger,
        workspace_fallback="/repo/default",
        prompt_extra_sections=[""],
    )
    artifacts_contract.unwrap.assert_called_once_with(
        operation="Claude SDK docker runner artifact build"
    )
    module.log_claude_sdk_runner_startup.assert_called_once_with(
        logger=logger,
        plan=plan,
    )
    mock_run_lifecycle.assert_awaited_once_with(agent_instance, shutdown_event, logger=logger)
