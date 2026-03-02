"""Tests for examples/claude_sdk_docker/runner.py."""

from __future__ import annotations

import asyncio
import importlib
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest


def _load_module() -> ModuleType:
    return importlib.import_module("examples.claude_sdk_docker.runner")


@pytest.mark.asyncio
async def test_main_uses_env_overrides_and_custom_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    shutdown_event = asyncio.Event()
    logger = MagicMock()
    config = {"agent_id": "agent-123"}
    plan = SimpleNamespace(
        agent_id="agent-123",
        model="claude-test",
        role="reviewer",
        workspace="/env/workspace",
        thinking_tokens=2048,
        final_prompt="final-prompt",
        custom_tools=[{"name": "tool_a"}],
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

    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module, "create_shutdown_event", MagicMock(return_value=shutdown_event))
    monkeypatch.setattr(module, "get_runner_config_path", MagicMock(return_value="agent.yml"))
    monkeypatch.setattr(
        module,
        "get_platform_urls",
        MagicMock(return_value=("wss://ws.example", "https://rest.example")),
    )
    monkeypatch.setattr(module, "validate_required_mounts", MagicMock())
    monkeypatch.setattr(module, "load_runner_config", MagicMock(return_value=config))
    monkeypatch.setattr(
        module,
        "build_claude_sdk_runner_artifacts_contract",
        MagicMock(return_value=artifacts_contract),
    )
    monkeypatch.setattr(module, "log_claude_sdk_runner_startup", MagicMock())
    monkeypatch.setattr(module, "create_runner_agent", MagicMock(return_value=agent_instance))
    monkeypatch.setattr(module, "run_agent_lifecycle", mock_run_lifecycle)

    await module.main()

    module.validate_required_mounts.assert_called_once()
    module.build_claude_sdk_runner_artifacts_contract.assert_called_once_with(
        ANY,
        logger=logger,
    )
    artifacts_contract.unwrap.assert_called_once_with(
        operation="Claude SDK example runner artifact build"
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
        startup_note="Press Ctrl+C to stop",
    )
    mock_run_lifecycle.assert_awaited_once_with(agent_instance, shutdown_event, logger=logger)


@pytest.mark.asyncio
async def test_main_falls_back_to_defaults_without_optional_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    shutdown_event = asyncio.Event()
    config = {"agent_id": "agent-456"}
    plan = SimpleNamespace(
        agent_id="agent-456",
        model="claude-sonnet-4-5-20250929",
        role=None,
        workspace=None,
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

    monkeypatch.delenv("WORKSPACE", raising=False)
    monkeypatch.delenv("AGENT_ROLE", raising=False)

    monkeypatch.setattr(module, "create_shutdown_event", MagicMock(return_value=shutdown_event))
    monkeypatch.setattr(module, "get_runner_config_path", MagicMock(return_value="agent.yml"))
    monkeypatch.setattr(module, "get_platform_urls", MagicMock(return_value=("ws", "rest")))
    monkeypatch.setattr(module, "validate_required_mounts", MagicMock())
    monkeypatch.setattr(module, "load_runner_config", MagicMock(return_value=config))
    monkeypatch.setattr(
        module,
        "build_claude_sdk_runner_artifacts_contract",
        MagicMock(return_value=artifacts_contract),
    )
    monkeypatch.setattr(module, "log_claude_sdk_runner_startup", MagicMock())
    monkeypatch.setattr(module, "create_runner_agent", MagicMock(return_value=object()))
    monkeypatch.setattr(module, "run_agent_lifecycle", AsyncMock())

    await module.main()

    module.build_claude_sdk_runner_artifacts_contract.assert_called_once_with(
        ANY,
        logger=module.logger,
    )
    artifacts_contract.unwrap.assert_called_once_with(
        operation="Claude SDK example runner artifact build"
    )
    module.log_claude_sdk_runner_startup.assert_called_once_with(
        logger=module.logger,
        plan=plan,
        startup_note="Press Ctrl+C to stop",
    )
