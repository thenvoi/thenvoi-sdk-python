"""Workflow helpers for ClaudeSessionManager command loop and session lifecycle."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SessionManagerProtocol(Protocol):
    """State surface required by session workflow helpers."""

    base_options: Any
    _sessions: dict[str, Any]
    _command_queue: asyncio.Queue[Any]

    async def _do_create_session(
        self, room_id: str | None, resume_session_id: str | None
    ) -> Any: ...

    async def _do_cleanup_session(self, room_id: str | None) -> None: ...

    async def _do_cleanup_all(self) -> None: ...

    def _record_nonfatal_error(
        self,
        category: str,
        error: Exception,
        **context: Any,
    ) -> None: ...


async def run_session_loop(manager: SessionManagerProtocol) -> None:
    """Process session commands from queue in a single task context."""
    logger.debug("Session loop started")

    while True:
        command: Any | None = None
        try:
            command = await manager._command_queue.get()

            if command.action == "create":
                client = await manager._do_create_session(
                    command.room_id,
                    command.resume_session_id,
                )
                if command.result_future:
                    command.result_future.set_result(client)
            elif command.action == "cleanup":
                await manager._do_cleanup_session(command.room_id)
                if command.result_future:
                    command.result_future.set_result(None)
            elif command.action == "cleanup_all":
                await manager._do_cleanup_all()
                if command.result_future:
                    command.result_future.set_result(None)
            elif command.action == "stop":
                await manager._do_cleanup_all()
                if command.result_future:
                    command.result_future.set_result(None)
                break

            manager._command_queue.task_done()
        except asyncio.CancelledError:
            logger.debug("Session loop cancelled")
            break
        except Exception as error:
            logger.error("Error in session loop: %s", error, exc_info=True)
            if command and command.result_future and not command.result_future.done():
                command.result_future.set_exception(error)

    logger.debug("Session loop exited")


def fail_pending_commands(manager: SessionManagerProtocol, reason: str) -> None:
    """Fail all pending command futures when lifecycle stops unexpectedly."""
    error = RuntimeError(reason)
    while True:
        try:
            command = manager._command_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        manager._command_queue.task_done()
        if command.result_future and not command.result_future.done():
            command.result_future.set_exception(error)


async def do_create_session(
    *,
    room_id: str | None,
    resume_session_id: str | None,
    base_options: Any,
    sessions: dict[str, Any],
    options_type: type[Any],
    client_type: type[Any],
) -> Any:
    """Create or reuse room session client."""
    if not room_id:
        raise ValueError("room_id is required")

    if room_id not in sessions:
        if resume_session_id:
            logger.info("Resuming session %s for room: %s", resume_session_id, room_id)
            options = options_type(
                model=base_options.model,
                system_prompt=base_options.system_prompt,
                mcp_servers=base_options.mcp_servers,
                allowed_tools=base_options.allowed_tools,
                permission_mode=base_options.permission_mode,
                resume=resume_session_id,
            )
            if hasattr(base_options, "max_thinking_tokens"):
                options.max_thinking_tokens = base_options.max_thinking_tokens
        else:
            logger.info("Creating new ClaudeSDKClient session for room: %s", room_id)
            options = base_options

        client = client_type(options=options)
        await client.connect()
        sessions[room_id] = client
        logger.info(
            "Session created for room %s (total sessions: %s)",
            room_id,
            len(sessions),
        )
    else:
        logger.debug("Reusing existing session for room: %s", room_id)

    return sessions[room_id]


async def do_cleanup_session(
    *,
    room_id: str | None,
    sessions: dict[str, Any],
    record_nonfatal_error: Any,
) -> None:
    """Disconnect and remove one room session."""
    if not room_id or room_id not in sessions:
        logger.debug("No session to cleanup for room: %s", room_id)
        return

    logger.info("Cleaning up session for room: %s", room_id)
    try:
        await sessions[room_id].disconnect()
        logger.debug("Disconnected client for room %s", room_id)
    except Exception as error:
        record_nonfatal_error(
            "disconnect_session",
            error,
            room_id=room_id,
        )

    del sessions[room_id]
    logger.info(
        "Session cleaned up for room %s (remaining sessions: %s)",
        room_id,
        len(sessions),
    )


async def do_cleanup_all(*, sessions: dict[str, Any], cleanup_session: Any) -> None:
    """Disconnect all room sessions."""
    logger.info("Cleaning up all sessions (count: %s)", len(sessions))
    for room_id in list(sessions.keys()):
        await cleanup_session(room_id)
    logger.info("All sessions cleaned up")

