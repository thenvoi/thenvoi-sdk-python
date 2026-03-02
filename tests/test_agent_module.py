"""Direct unit tests for src/thenvoi/agent.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import thenvoi.agent as agent_module


def _make_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        agent_id="agent-123",
        agent_name="Test Agent",
        agent_description="Agent description",
        contact_config=SimpleNamespace(),
        is_contacts_subscribed=False,
        initialize=AsyncMock(),
        start=AsyncMock(),
        stop=AsyncMock(return_value=True),
        run_forever=AsyncMock(),
    )


def _make_adapter() -> SimpleNamespace:
    return SimpleNamespace(
        on_started=AsyncMock(),
        on_cleanup=AsyncMock(),
        on_event=AsyncMock(),
    )


def _make_preprocessor(return_value: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(process=AsyncMock(return_value=return_value))


def test_agent_create_builds_default_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _make_runtime()
    mock_runtime_cls = MagicMock(return_value=runtime)
    monkeypatch.setattr(agent_module, "PlatformRuntime", mock_runtime_cls)
    adapter = _make_adapter()
    preprocessor = _make_preprocessor()

    agent = agent_module.Agent.create(
        adapter=adapter,
        agent_id="agent-123",
        api_key="api-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
        preprocessor=preprocessor,
    )

    assert isinstance(agent, agent_module.Agent)
    mock_runtime_cls.assert_called_once_with(
        agent_id="agent-123",
        api_key="api-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
        config=None,
        session_config=None,
        contact_config=None,
    )


@pytest.mark.asyncio
async def test_start_initializes_runtime_and_adapter() -> None:
    runtime = _make_runtime()
    adapter = _make_adapter()
    preprocessor = _make_preprocessor()
    agent = agent_module.Agent(runtime=runtime, adapter=adapter, preprocessor=preprocessor)

    await agent.start()

    runtime.initialize.assert_awaited_once()
    adapter.on_started.assert_awaited_once_with("Test Agent", "Agent description")
    runtime.start.assert_awaited_once_with(
        on_execute=agent._on_execute,
        on_cleanup=adapter.on_cleanup,
    )
    assert agent.is_running is True


@pytest.mark.asyncio
async def test_stop_returns_graceful_result_and_resets_running_state() -> None:
    runtime = _make_runtime()
    adapter = _make_adapter()
    agent = agent_module.Agent(runtime=runtime, adapter=adapter)
    agent._started = True
    runtime.stop = AsyncMock(return_value=False)

    graceful = await agent.stop(timeout=5.0)

    runtime.stop.assert_awaited_once_with(timeout=5.0)
    assert graceful is False
    assert agent.is_running is False


@pytest.mark.asyncio
async def test_run_stops_in_finally_and_preserves_timeout() -> None:
    runtime = _make_runtime()
    runtime.run_forever = AsyncMock(side_effect=RuntimeError("boom"))
    adapter = _make_adapter()
    agent = agent_module.Agent(runtime=runtime, adapter=adapter)
    agent.stop = AsyncMock(return_value=True)

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run(shutdown_timeout=7.5)

    assert agent._shutdown_timeout == 7.5
    agent.stop.assert_awaited_once_with(timeout=7.5)


@pytest.mark.asyncio
async def test_context_exit_uses_default_timeout_when_run_never_called() -> None:
    runtime = _make_runtime()
    adapter = _make_adapter()
    agent = agent_module.Agent(runtime=runtime, adapter=adapter)
    agent.stop = AsyncMock(return_value=True)

    await agent.__aexit__(None, None, None)

    agent.stop.assert_awaited_once_with(timeout=agent_module.DEFAULT_SHUTDOWN_TIMEOUT)


@pytest.mark.asyncio
async def test_on_execute_forwards_processed_input_to_adapter() -> None:
    runtime = _make_runtime()
    adapter = _make_adapter()
    inp = object()
    preprocessor = _make_preprocessor(return_value=inp)
    agent = agent_module.Agent(runtime=runtime, adapter=adapter, preprocessor=preprocessor)

    ctx = object()
    event = object()
    await agent._on_execute(ctx=ctx, event=event)

    preprocessor.process.assert_awaited_once_with(
        ctx=ctx,
        event=event,
        agent_id="agent-123",
    )
    adapter.on_event.assert_awaited_once_with(inp)


@pytest.mark.asyncio
async def test_on_execute_skips_adapter_when_preprocessor_filters_event() -> None:
    runtime = _make_runtime()
    adapter = _make_adapter()
    preprocessor = _make_preprocessor(return_value=None)
    agent = agent_module.Agent(runtime=runtime, adapter=adapter, preprocessor=preprocessor)

    await agent._on_execute(ctx=object(), event=object())

    adapter.on_event.assert_not_called()
