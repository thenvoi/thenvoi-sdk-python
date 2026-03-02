"""Unit tests for Claude session manager lifecycle behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import thenvoi.integrations.claude_sdk.session_manager as session_manager_module


class _FakeOptions:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        mcp_servers: list[str],
        allowed_tools: list[str],
        permission_mode: str,
        resume: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.mcp_servers = mcp_servers
        self.allowed_tools = allowed_tools
        self.permission_mode = permission_mode
        self.resume = resume


class _FakeClient:
    def __init__(self, options: object) -> None:
        self.options = options
        self.connected = False
        self.disconnected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True


def _base_options(*, max_thinking_tokens: int | None = None) -> SimpleNamespace:
    options = SimpleNamespace(
        model="claude-model",
        system_prompt="prompt",
        mcp_servers=[],
        allowed_tools=["a"],
        permission_mode="default",
    )
    if max_thinking_tokens is not None:
        options.max_thinking_tokens = max_thinking_tokens
    return options


@pytest.fixture
def patched_session_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, list[_FakeClient]]:
    created_clients: list[_FakeClient] = []

    def _client_factory(options: object) -> _FakeClient:
        client = _FakeClient(options)
        created_clients.append(client)
        return client

    monkeypatch.setattr(session_manager_module, "_CLAUDE_SDK_IMPORT_ERROR", None)
    monkeypatch.setattr(session_manager_module, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(session_manager_module, "ClaudeSDKClient", _client_factory)
    return session_manager_module, created_clients


@pytest.mark.asyncio
async def test_get_or_create_session_reuses_client(
    patched_session_manager,
) -> None:
    session_manager, created_clients = patched_session_manager
    manager = session_manager.ClaudeSessionManager(_base_options())

    first = await manager.get_or_create_session("room-1")
    second = await manager.get_or_create_session("room-1")

    assert first is second
    assert first.connected is True
    assert manager.has_session("room-1") is True
    assert manager.get_session_count() == 1
    assert manager.get_active_rooms() == ["room-1"]
    assert len(created_clients) == 1

    await manager.stop()
    assert first.disconnected is True


@pytest.mark.asyncio
async def test_cleanup_session_disconnects_and_removes(
    patched_session_manager,
) -> None:
    session_manager, _ = patched_session_manager
    manager = session_manager.ClaudeSessionManager(_base_options())
    client = await manager.get_or_create_session("room-2")

    await manager.cleanup_session("room-2")

    assert client.disconnected is True
    assert manager.has_session("room-2") is False
    assert manager.get_session_count() == 0

    await manager.stop()


@pytest.mark.asyncio
async def test_cleanup_session_records_nonfatal_when_disconnect_fails(
    patched_session_manager,
) -> None:
    session_manager, _ = patched_session_manager
    manager = session_manager.ClaudeSessionManager(_base_options())
    client = await manager.get_or_create_session("room-bad")

    async def _fail_disconnect() -> None:
        raise RuntimeError("disconnect failed")

    client.disconnect = _fail_disconnect  # type: ignore[method-assign]

    await manager.cleanup_session("room-bad")

    assert manager.has_session("room-bad") is False
    assert manager.nonfatal_errors
    assert manager.nonfatal_errors[0]["operation"] == "disconnect_session"

    await manager.stop()


@pytest.mark.asyncio
async def test_get_or_create_session_with_resume_creates_options_with_resume(
    patched_session_manager,
) -> None:
    session_manager, _ = patched_session_manager
    manager = session_manager.ClaudeSessionManager(
        _base_options(max_thinking_tokens=2048)
    )

    client = await manager.get_or_create_session("room-3", resume_session_id="sess-123")

    assert isinstance(client.options, _FakeOptions)
    assert client.options.resume == "sess-123"
    assert client.options.max_thinking_tokens == 2048

    await manager.stop()


@pytest.mark.asyncio
async def test_cleanup_all_when_not_started_is_noop(
    patched_session_manager,
) -> None:
    session_manager, _ = patched_session_manager
    manager = session_manager.ClaudeSessionManager(_base_options())

    await manager.cleanup_all()

    assert manager.get_session_count() == 0


@pytest.mark.asyncio
async def test_get_or_create_session_propagates_connect_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingClient:
        def __init__(self, options: object) -> None:
            self.options = options

        async def connect(self) -> None:
            raise RuntimeError("connect failed")

        async def disconnect(self) -> None:
            return

    monkeypatch.setattr(session_manager_module, "_CLAUDE_SDK_IMPORT_ERROR", None)
    monkeypatch.setattr(session_manager_module, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(session_manager_module, "ClaudeSDKClient", _FailingClient)
    manager = session_manager_module.ClaudeSessionManager(_base_options())

    with pytest.raises(RuntimeError, match="connect failed"):
        await manager.get_or_create_session("room-err")

    await manager.stop()
