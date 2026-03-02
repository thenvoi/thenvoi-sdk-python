"""Shared async lifecycle helpers for integration runtime clients."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

TaskErrorHandler = Callable[[str, Exception], None]
PendingFailureHandler = Callable[[str], None]


class AsyncIntegrationLifecycle:
    """Track background tasks and apply consistent shutdown semantics."""

    def __init__(
        self,
        *,
        owner: str,
        logger: logging.Logger,
        on_task_error: TaskErrorHandler | None = None,
        fail_pending_operations: PendingFailureHandler | None = None,
    ) -> None:
        self._owner = owner
        self._logger = logger
        self._on_task_error = on_task_error
        self._fail_pending_operations = fail_pending_operations
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    def spawn_task(
        self,
        name: str,
        coroutine: Awaitable[Any],
    ) -> asyncio.Task[Any]:
        """Create and register a background task by name."""
        existing = self._tasks.get(name)
        if existing is not None and not existing.done():
            raise ValueError(f"Task '{name}' is already running for {self._owner}")

        task = asyncio.create_task(coroutine)
        self._tasks[name] = task
        return task

    def get_task(self, name: str) -> asyncio.Task[Any] | None:
        """Return a registered task if present."""
        return self._tasks.get(name)

    async def cancel_registered_tasks(self) -> None:
        """Cancel all registered tasks with uniform logging/error handling."""
        for task_name, task in list(self._tasks.items()):
            await self._cancel_task(task_name, task)
        self._tasks.clear()

    def fail_pending_operations(self, reason: str) -> None:
        """Apply the client-specific pending-operation failure policy."""
        if self._fail_pending_operations is None:
            return
        self._fail_pending_operations(reason)

    async def shutdown(self, *, fail_pending_reason: str | None = None) -> None:
        """Cancel all tasks and optionally fail pending operations."""
        await self.cancel_registered_tasks()
        if fail_pending_reason:
            self.fail_pending_operations(fail_pending_reason)

    async def _cancel_task(self, task_name: str, task: asyncio.Task[Any]) -> None:
        if not task.done():
            task.cancel()

        result = (await asyncio.gather(task, return_exceptions=True))[0]
        if isinstance(result, asyncio.CancelledError):
            self._logger.debug(
                "Cancelled %s task '%s' during shutdown",
                self._owner,
                task_name,
            )
            return

        if isinstance(result, Exception):
            if self._on_task_error is not None:
                self._on_task_error(task_name, result)
                return

            self._logger.warning(
                "%s task '%s' exited with error during shutdown: %s",
                self._owner,
                task_name,
                result,
            )
