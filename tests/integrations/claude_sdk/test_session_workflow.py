"""Unit tests for Claude session workflow helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from thenvoi.integrations.claude_sdk.session_workflow import (
    do_cleanup_all,
    do_cleanup_session,
    do_create_session,
    fail_pending_commands,
    run_session_loop,
)


@dataclass
class _Command:
    action: str
    room_id: str | None = None
    resume_session_id: str | None = None
    result_future: asyncio.Future[Any] | None = None


class _FakeClient:
    def __init__(self, options: object) -> None:
        self.options = options
        self.connected = False
        self.disconnected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True


class _FakeManager:
    def __init__(self) -> None:
        self._command_queue: asyncio.Queue[_Command] = asyncio.Queue()
        self._sessions: dict[str, _FakeClient] = {}
        self.base_options = SimpleNamespace(
            model="claude",
            system_prompt="prompt",
            mcp_servers=[],
            allowed_tools=[],
            permission_mode="default",
        )
        self.created: list[tuple[str | None, str | None]] = []
        self.cleaned: list[str | None] = []
        self.cleanup_all_count = 0
        self.recorded_errors: list[dict[str, Any]] = []

    async def _do_create_session(
        self, room_id: str | None, resume_session_id: str | None
    ) -> str:
        self.created.append((room_id, resume_session_id))
        return f"created:{room_id}"

    async def _do_cleanup_session(self, room_id: str | None) -> None:
        self.cleaned.append(room_id)

    async def _do_cleanup_all(self) -> None:
        self.cleanup_all_count += 1

    def _record_nonfatal_error(
        self,
        category: str,
        error: Exception,
        **context: Any,
    ) -> None:
        self.recorded_errors.append(
            {
                "category": category,
                "error": str(error),
                "context": context,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )


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
        self.max_thinking_tokens: int | None = None


@pytest.mark.asyncio
async def test_run_session_loop_processes_commands_and_stops() -> None:
    manager = _FakeManager()
    create_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    cleanup_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    stop_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    await manager._command_queue.put(
        _Command(
            action="create",
            room_id="room-1",
            resume_session_id="resume-1",
            result_future=create_future,
        )
    )
    await manager._command_queue.put(
        _Command(action="cleanup", room_id="room-1", result_future=cleanup_future)
    )
    await manager._command_queue.put(_Command(action="stop", result_future=stop_future))

    await run_session_loop(manager)

    assert create_future.result() == "created:room-1"
    assert cleanup_future.done()
    assert stop_future.done()
    assert manager.created == [("room-1", "resume-1")]
    assert manager.cleaned == ["room-1"]
    assert manager.cleanup_all_count == 1


def test_fail_pending_commands_sets_exception_on_futures() -> None:
    manager = _FakeManager()
    loop = asyncio.new_event_loop()
    try:
        future: asyncio.Future[None] = loop.create_future()
        manager._command_queue.put_nowait(_Command(action="cleanup", result_future=future))

        fail_pending_commands(manager, "boom")

        assert future.done()
        with pytest.raises(RuntimeError, match="boom"):
            future.result()
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_do_create_session_creates_and_reuses_clients() -> None:
    base_options = SimpleNamespace(
        model="claude",
        system_prompt="prompt",
        mcp_servers=[],
        allowed_tools=["x"],
        permission_mode="default",
        max_thinking_tokens=2048,
    )
    sessions: dict[str, _FakeClient] = {}

    resumed = await do_create_session(
        room_id="room-1",
        resume_session_id="resume-1",
        base_options=base_options,
        sessions=sessions,
        options_type=_FakeOptions,
        client_type=_FakeClient,
    )
    reused = await do_create_session(
        room_id="room-1",
        resume_session_id=None,
        base_options=base_options,
        sessions=sessions,
        options_type=_FakeOptions,
        client_type=_FakeClient,
    )

    assert resumed is reused
    assert resumed.connected is True
    assert isinstance(resumed.options, _FakeOptions)
    assert resumed.options.resume == "resume-1"
    assert resumed.options.max_thinking_tokens == 2048


@pytest.mark.asyncio
async def test_do_cleanup_session_handles_disconnect_errors() -> None:
    sessions: dict[str, _FakeClient] = {"room": _FakeClient(options={})}
    recorded: list[str] = []

    async def _broken_disconnect() -> None:
        raise RuntimeError("disconnect failed")

    sessions["room"].disconnect = _broken_disconnect  # type: ignore[method-assign]

    await do_cleanup_session(
        room_id="room",
        sessions=sessions,
        record_nonfatal_error=lambda category, error, **_ctx: recorded.append(
            f"{category}:{error}"
        ),
    )

    assert not sessions
    assert recorded == ["disconnect_session:disconnect failed"]


@pytest.mark.asyncio
async def test_do_cleanup_all_invokes_cleanup_for_each_session() -> None:
    sessions = {"a": object(), "b": object()}
    cleaned: list[str] = []

    async def _cleanup(room_id: str) -> None:
        cleaned.append(room_id)
        sessions.pop(room_id, None)

    await do_cleanup_all(sessions=sessions, cleanup_session=_cleanup)

    assert set(cleaned) == {"a", "b"}
    assert sessions == {}
