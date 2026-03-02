"""Unit tests for docker/codex/runner.py."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_codex_runner_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    repo_init_stub = ModuleType("repo_init")
    repo_init_stub.initialize_repo = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "repo_init", repo_init_stub)

    module_name = "docker.codex.runner"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_env_bool_and_optional_str(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_codex_runner_module(monkeypatch)

    monkeypatch.setenv("BOOL_FLAG", "true")
    assert module._env_bool("BOOL_FLAG") is True

    monkeypatch.setenv("BOOL_FLAG", "0")
    assert module._env_bool("BOOL_FLAG") is False
    assert module._env_bool("MISSING_FLAG", default=True) is True

    assert module._optional_str(None) is None
    assert module._optional_str("   ") is None
    assert module._optional_str(" value ") == "value"
    assert module._optional_str(123) == "123"


def test_parse_helpers_validate_and_normalize_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_codex_runner_module(monkeypatch)

    assert module._parse_transport("STDIO") == "stdio"
    assert module._parse_transport("ws") == "ws"
    with pytest.raises(ValueError, match="CODEX_TRANSPORT must be one of"):
        module._parse_transport("http")

    assert module._parse_approval_mode("AUTO_ACCEPT") == "auto_accept"
    with pytest.raises(ValueError, match="CODEX_APPROVAL_MODE must be one of"):
        module._parse_approval_mode("always")

    assert module._parse_reasoning_effort(None) is None
    assert module._parse_reasoning_effort("HIGH") == "high"
    with pytest.raises(ValueError, match="CODEX_REASONING_EFFORT must be one of"):
        module._parse_reasoning_effort("ultra")


@pytest.mark.asyncio
async def test_main_builds_codex_adapter_config_with_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_codex_runner_module(monkeypatch)

    import thenvoi.adapters as adapters_module

    shutdown_event = asyncio.Event()
    logger = MagicMock()
    repo_init = SimpleNamespace(
        repo_path="/repo/path",
        context_bundle="repo-context",
        enabled=True,
        cloned=False,
        indexed=True,
    )
    config = {
        "agent_id": "agent-codex",
        "workspace": "/config/workspace",
        "model": "config-model",
        "role": "planner",
        "sandbox": "config-sandbox",
        "reasoning_effort": "low",
        "approval_mode": "manual",
    }
    adapter_instance = object()
    agent_instance = object()
    mock_codex_adapter_cls = MagicMock(return_value=adapter_instance)
    mock_codex_config_cls = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    mock_run_lifecycle = AsyncMock()

    codex_package_stub = ModuleType("thenvoi.adapters.codex")
    codex_adapter_stub = ModuleType("thenvoi.adapters.codex.adapter")
    codex_adapter_stub.CodexAdapterConfig = mock_codex_config_cls
    monkeypatch.setitem(sys.modules, "thenvoi.adapters.codex", codex_package_stub)
    monkeypatch.setitem(sys.modules, "thenvoi.adapters.codex.adapter", codex_adapter_stub)
    monkeypatch.setitem(adapters_module.__dict__, "CodexAdapter", mock_codex_adapter_cls)

    monkeypatch.setenv("CODEX_CWD", "/env/cwd")
    monkeypatch.setenv("CODEX_TRANSPORT", "WS")
    monkeypatch.setenv("CODEX_MODEL", "env-model")
    monkeypatch.setenv("CODEX_ROLE", "reviewer")
    monkeypatch.setenv("CODEX_SANDBOX", "workspace-write")
    monkeypatch.setenv("CODEX_REASONING_EFFORT", "medium")
    monkeypatch.setenv("CODEX_APPROVAL_MODE", "auto_accept")
    monkeypatch.setenv("CODEX_TURN_TASK_MARKERS", "yes")
    monkeypatch.setenv("CODEX_WS_URL", "ws://codex.example:9000")
    monkeypatch.setenv("REPO_INIT_LOCK_TIMEOUT_S", "30")

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
    monkeypatch.setattr(module, "compose_runner_prompt", MagicMock(return_value="custom-prompt"))
    monkeypatch.setattr(module, "create_runner_agent", MagicMock(return_value=agent_instance))
    monkeypatch.setattr(module, "run_agent_lifecycle", mock_run_lifecycle)

    await module.main()

    module.get_agent_key.assert_called_once_with(
        "AGENT_KEY",
        "CODEX_AGENT_KEY",
        default="agent",
    )
    module.initialize_repo.assert_called_once_with(
        config,
        agent_key="agent-key",
        lock_timeout_s=30.0,
    )
    module.compose_runner_prompt.assert_called_once_with(
        "agent.yml",
        role="reviewer",
        extra_sections=["repo-context"],
        default_prompt="",
        logger=logger,
    )
    codex_config_kwargs = mock_codex_config_cls.call_args.kwargs
    assert codex_config_kwargs["transport"] == "ws"
    assert codex_config_kwargs["cwd"] == "/env/cwd"
    assert codex_config_kwargs["model"] == "env-model"
    assert codex_config_kwargs["approval_mode"] == "auto_accept"
    assert codex_config_kwargs["sandbox"] == "workspace-write"
    assert codex_config_kwargs["reasoning_effort"] == "medium"
    assert codex_config_kwargs["codex_ws_url"] == "ws://codex.example:9000"
    assert codex_config_kwargs["custom_section"] == "custom-prompt"
    assert codex_config_kwargs["emit_turn_task_markers"] is True
    mock_codex_adapter_cls.assert_called_once()
    module.create_runner_agent.assert_called_once_with(
        adapter=adapter_instance,
        config=config,
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    mock_run_lifecycle.assert_awaited_once_with(agent_instance, shutdown_event, logger=logger)


@pytest.mark.asyncio
async def test_main_uses_config_defaults_when_env_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_codex_runner_module(monkeypatch)

    import thenvoi.adapters as adapters_module

    shutdown_event = asyncio.Event()
    repo_init = SimpleNamespace(
        repo_path="/repo/default",
        context_bundle="",
        enabled=False,
        cloned=False,
        indexed=False,
    )
    config = {
        "agent_id": "agent-codex",
        "workspace": "/config/workspace",
        "model": "config-model",
        "role": "planner",
        "sandbox": "config-sandbox",
        "reasoning_effort": "high",
        "approval_mode": "auto_decline",
    }
    mock_codex_adapter_cls = MagicMock(return_value=object())
    mock_codex_config_cls = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))

    codex_package_stub = ModuleType("thenvoi.adapters.codex")
    codex_adapter_stub = ModuleType("thenvoi.adapters.codex.adapter")
    codex_adapter_stub.CodexAdapterConfig = mock_codex_config_cls
    monkeypatch.setitem(sys.modules, "thenvoi.adapters.codex", codex_package_stub)
    monkeypatch.setitem(sys.modules, "thenvoi.adapters.codex.adapter", codex_adapter_stub)
    monkeypatch.setitem(adapters_module.__dict__, "CodexAdapter", mock_codex_adapter_cls)

    for key in (
        "CODEX_CWD",
        "CODEX_TRANSPORT",
        "CODEX_MODEL",
        "CODEX_ROLE",
        "CODEX_SANDBOX",
        "CODEX_REASONING_EFFORT",
        "CODEX_APPROVAL_MODE",
        "CODEX_TURN_TASK_MARKERS",
        "CODEX_WS_URL",
        "REPO_INIT_LOCK_TIMEOUT_S",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(module, "create_shutdown_event", MagicMock(return_value=shutdown_event))
    monkeypatch.setattr(module, "get_runner_config_path", MagicMock(return_value="agent.yml"))
    monkeypatch.setattr(module, "get_agent_key", MagicMock(return_value="agent-key"))
    monkeypatch.setattr(module, "get_platform_urls", MagicMock(return_value=("ws", "rest")))
    monkeypatch.setattr(module, "validate_required_mounts", MagicMock())
    monkeypatch.setattr(module, "load_runner_config", MagicMock(return_value=config))
    monkeypatch.setattr(module, "initialize_repo", MagicMock(return_value=repo_init))
    monkeypatch.setattr(module, "compose_runner_prompt", MagicMock(return_value="prompt"))
    monkeypatch.setattr(module, "create_runner_agent", MagicMock(return_value=object()))
    monkeypatch.setattr(module, "run_agent_lifecycle", AsyncMock())

    await module.main()

    codex_config_kwargs = mock_codex_config_cls.call_args.kwargs
    assert codex_config_kwargs["transport"] == "stdio"
    assert codex_config_kwargs["cwd"] == "/config/workspace"
    assert codex_config_kwargs["model"] == "config-model"
    assert codex_config_kwargs["approval_mode"] == "auto_decline"
    assert codex_config_kwargs["sandbox"] == "config-sandbox"
    assert codex_config_kwargs["reasoning_effort"] == "high"
    assert codex_config_kwargs["codex_ws_url"] == "ws://127.0.0.1:8765"
    assert codex_config_kwargs["emit_turn_task_markers"] is False
