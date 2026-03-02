"""Tests for examples/a2a_gateway/01_basic_gateway.py."""

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import examples.a2a_gateway.basic_gateway as basic_gateway_module


def _load_example_module() -> ModuleType:
    return importlib.reload(basic_gateway_module)


@pytest.mark.asyncio
async def test_main_prefers_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module()
    monkeypatch.setenv("THENVOI_API_KEY", "api-key")
    monkeypatch.setenv("THENVOI_AGENT_ID", "gateway-123")
    monkeypatch.setenv("GATEWAY_PORT", "12000")
    monkeypatch.delenv("GATEWAY_URL", raising=False)

    mock_setup_logging = MagicMock()
    mock_load_urls = MagicMock(return_value=("ws://example", "https://example"))
    mock_load_runtime = MagicMock()
    mock_adapter_cls = MagicMock(return_value=object())
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_create_agent = MagicMock(return_value=session)

    monkeypatch.setattr(module, "setup_logging_profile", mock_setup_logging)
    monkeypatch.setattr(module, "load_platform_urls", mock_load_urls)
    monkeypatch.setattr(module, "load_runtime_config", mock_load_runtime)
    monkeypatch.setattr(module, "A2AGatewayAdapter", mock_adapter_cls)
    monkeypatch.setattr(module, "create_agent_from_runtime", mock_create_agent)

    await module.main()

    mock_setup_logging.assert_called_once_with("a2a_gateway")
    mock_load_urls.assert_called_once()
    mock_load_runtime.assert_not_called()
    mock_adapter_cls.assert_called_once_with(
        rest_url="https://example",
        api_key="api-key",
        gateway_url="http://localhost:12000",
        port=12000,
    )
    mock_create_agent.assert_called_once()
    runtime_arg, adapter_arg = mock_create_agent.call_args.args
    assert runtime_arg.agent_key == "gateway_agent"
    assert runtime_arg.agent_id == "gateway-123"
    assert runtime_arg.api_key == "api-key"
    assert runtime_arg.ws_url == "ws://example"
    assert runtime_arg.rest_url == "https://example"
    assert adapter_arg is mock_adapter_cls.return_value
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_uses_config_fallback_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()
    monkeypatch.delenv("THENVOI_API_KEY", raising=False)
    monkeypatch.setenv("GATEWAY_PORT", "10000")
    monkeypatch.setenv("GATEWAY_URL", "http://gateway.internal:7777")

    runtime = module.ExampleRuntimeConfig(
        agent_key="gateway_agent",
        agent_id="configured-agent",
        api_key="cfg-key",
        ws_url="ws://cfg",
        rest_url="https://cfg",
    )

    mock_setup_logging = MagicMock()
    mock_load_urls = MagicMock(return_value=("ws://unused", "https://unused"))
    mock_load_runtime = MagicMock(return_value=runtime)
    mock_adapter_cls = MagicMock(return_value=object())
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_create_agent = MagicMock(return_value=session)

    monkeypatch.setattr(module, "setup_logging_profile", mock_setup_logging)
    monkeypatch.setattr(module, "load_platform_urls", mock_load_urls)
    monkeypatch.setattr(module, "load_runtime_config", mock_load_runtime)
    monkeypatch.setattr(module, "A2AGatewayAdapter", mock_adapter_cls)
    monkeypatch.setattr(module, "create_agent_from_runtime", mock_create_agent)

    await module.main()

    mock_load_runtime.assert_called_once_with(
        "gateway_agent",
        ws_default="wss://app.thenvoi.com/api/v1/socket/websocket",
        rest_default="https://app.thenvoi.com",
        load_env=False,
    )
    mock_adapter_cls.assert_called_once_with(
        rest_url="https://cfg",
        api_key="cfg-key",
        gateway_url="http://gateway.internal:7777",
        port=10000,
    )
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_raises_clear_error_when_no_runtime_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()
    monkeypatch.delenv("THENVOI_API_KEY", raising=False)

    monkeypatch.setattr(module, "setup_logging_profile", MagicMock())
    monkeypatch.setattr(
        module,
        "load_platform_urls",
        MagicMock(return_value=("ws://example", "https://example")),
    )
    monkeypatch.setattr(
        module,
        "load_runtime_config",
        MagicMock(side_effect=RuntimeError("missing agent config")),
    )

    with pytest.raises(ValueError, match="THENVOI_API_KEY environment variable is required"):
        await module.main()
