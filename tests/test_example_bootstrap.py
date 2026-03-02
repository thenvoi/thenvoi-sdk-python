"""Tests for shared example bootstrap contract."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import thenvoi.example_support.bootstrap as bootstrap_module
from thenvoi.example_support.bootstrap import BootstrappedAgent, ExampleRuntimeConfig

pytestmark = pytest.mark.contract_gate


def test_load_platform_urls_prefers_environment_over_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment values should win over defaults when both are present."""
    monkeypatch.setenv("THENVOI_WS_URL", "ws://env.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://env.example")
    mock_load_dotenv = MagicMock()
    monkeypatch.setattr(bootstrap_module, "load_dotenv", mock_load_dotenv)

    ws_url, rest_url = bootstrap_module.load_platform_urls(
        ws_default="ws://default.example",
        rest_default="https://default.example",
        load_env=True,
    )

    assert ws_url == "ws://env.example"
    assert rest_url == "https://env.example"
    mock_load_dotenv.assert_called_once_with()


def test_load_platform_urls_uses_defaults_and_skips_dotenv_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defaults should be used when env vars are absent and dotenv is disabled."""
    monkeypatch.delenv("THENVOI_WS_URL", raising=False)
    monkeypatch.delenv("THENVOI_REST_URL", raising=False)
    mock_load_dotenv = MagicMock()
    monkeypatch.setattr(bootstrap_module, "load_dotenv", mock_load_dotenv)

    ws_url, rest_url = bootstrap_module.load_platform_urls(
        ws_default="ws://default.example",
        rest_default="https://default.example",
        load_env=False,
    )

    assert ws_url == "ws://default.example"
    assert rest_url == "https://default.example"
    mock_load_dotenv.assert_not_called()


def test_load_platform_urls_raises_for_missing_rest_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing REST endpoint should raise a clear configuration error."""
    monkeypatch.delenv("THENVOI_WS_URL", raising=False)
    monkeypatch.delenv("THENVOI_REST_URL", raising=False)

    with pytest.raises(
        ValueError, match="THENVOI_REST_URL environment variable is required"
    ):
        bootstrap_module.load_platform_urls(
            ws_default="ws://default.example",
            rest_default=None,
            load_env=False,
        )


def test_load_runtime_config_uses_url_and_credential_resolvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_runtime_config should combine resolved URLs and credentials."""
    monkeypatch.setattr(
        bootstrap_module,
        "load_platform_urls",
        MagicMock(return_value=("ws://resolved", "https://resolved")),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "resolve_agent_credentials",
        MagicMock(return_value=("agent-123", "api-abc")),
    )

    runtime = bootstrap_module.load_runtime_config(
        "sample_agent",
        ws_default="ws://ignored",
        rest_default="https://ignored",
        load_env=False,
    )

    assert runtime == ExampleRuntimeConfig(
        agent_key="sample_agent",
        agent_id="agent-123",
        api_key="api-abc",
        ws_url="ws://resolved",
        rest_url="https://resolved",
    )


def test_load_runtime_config_allows_injected_resolvers() -> None:
    """Injected resolvers should bypass global loader patching."""
    runtime = bootstrap_module.load_runtime_config(
        "sample_agent",
        load_env=False,
        url_resolver=lambda **_: ("ws://injected", "https://injected"),
        credentials_resolver=lambda _agent_key: ("agent-injected", "key-injected"),
    )

    assert runtime == ExampleRuntimeConfig(
        agent_key="sample_agent",
        agent_id="agent-injected",
        api_key="key-injected",
        ws_url="ws://injected",
        rest_url="https://injected",
    )


def test_create_agent_from_runtime_passes_credentials_to_agent_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_agent_from_runtime should forward runtime credentials to Agent.create."""
    agent_instance = object()
    mock_create = MagicMock(return_value=agent_instance)
    monkeypatch.setattr(bootstrap_module.Agent, "create", mock_create)

    adapter = object()
    runtime = ExampleRuntimeConfig(
        agent_key="demo",
        agent_id="agent-1",
        api_key="secret",
        ws_url="ws://thenvoi",
        rest_url="https://thenvoi",
    )

    result = bootstrap_module.create_agent_from_runtime(runtime, adapter)

    assert result == BootstrappedAgent(runtime=runtime, agent=agent_instance)
    mock_create.assert_called_once_with(
        adapter=adapter,
        agent_id="agent-1",
        api_key="secret",
        ws_url="ws://thenvoi",
        rest_url="https://thenvoi",
    )


def test_bootstrap_agent_uses_shared_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """bootstrap_agent should wire load_runtime_config -> create_agent_from_runtime."""
    runtime = ExampleRuntimeConfig(
        agent_key="alpha",
        agent_id="agent-alpha",
        api_key="api-alpha",
        ws_url="ws://alpha",
        rest_url="https://alpha",
    )
    bootstrapped = BootstrappedAgent(runtime=runtime, agent=object())

    mock_load_runtime = MagicMock(return_value=runtime)
    mock_create_agent = MagicMock(return_value=bootstrapped)
    monkeypatch.setattr(bootstrap_module, "load_runtime_config", mock_load_runtime)
    monkeypatch.setattr(
        bootstrap_module, "create_agent_from_runtime", mock_create_agent
    )

    adapter = object()
    result = bootstrap_module.bootstrap_agent("alpha", adapter)

    assert result is bootstrapped
    mock_load_runtime.assert_called_once_with("alpha")
    mock_create_agent.assert_called_once_with(runtime, adapter)


def test_bootstrap_agent_allows_injected_pipeline() -> None:
    runtime = ExampleRuntimeConfig(
        agent_key="alpha",
        agent_id="agent-alpha",
        api_key="api-alpha",
        ws_url="ws://alpha",
        rest_url="https://alpha",
    )
    bootstrapped = BootstrappedAgent(runtime=runtime, agent=object())

    called: dict[str, object] = {}

    def _runtime_loader(agent_key: str) -> ExampleRuntimeConfig:
        called["agent_key"] = agent_key
        return runtime

    def _agent_builder(
        runtime_config: ExampleRuntimeConfig,
        adapter: object,
    ) -> BootstrappedAgent:
        called["runtime"] = runtime_config
        called["adapter"] = adapter
        return bootstrapped

    adapter = object()
    result = bootstrap_module.bootstrap_agent(
        "alpha",
        adapter,
        runtime_loader=_runtime_loader,
        agent_builder=_agent_builder,
    )

    assert result is bootstrapped
    assert called["agent_key"] == "alpha"
    assert called["runtime"] == runtime
    assert called["adapter"] is adapter


def test_bootstrap_agent_resolves_runtime_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bootstrap_agent should run real runtime resolution before agent creation."""
    config_path = tmp_path / "agent_config.yaml"
    config_path.write_text(
        "alpha:\n  agent_id: agent-alpha\n  api_key: key-alpha\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("THENVOI_WS_URL", "wss://env.example/socket")
    monkeypatch.setenv("THENVOI_REST_URL", "https://env.example")
    monkeypatch.setattr(bootstrap_module, "load_dotenv", MagicMock())

    agent_instance = object()
    mock_create = MagicMock(return_value=agent_instance)
    monkeypatch.setattr(bootstrap_module.Agent, "create", mock_create)

    adapter = object()
    result = bootstrap_module.bootstrap_agent("alpha", adapter)

    assert result.runtime == ExampleRuntimeConfig(
        agent_key="alpha",
        agent_id="agent-alpha",
        api_key="key-alpha",
        ws_url="wss://env.example/socket",
        rest_url="https://env.example",
    )
    assert result.agent is agent_instance
    mock_create.assert_called_once_with(
        adapter=adapter,
        agent_id="agent-alpha",
        api_key="key-alpha",
        ws_url="wss://env.example/socket",
        rest_url="https://env.example",
    )
