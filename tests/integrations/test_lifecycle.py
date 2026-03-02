"""Tests for shared integration lifecycle helpers."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from thenvoi.integrations.lifecycle import AsyncIntegrationLifecycle


@pytest.mark.asyncio
async def test_shutdown_cancels_registered_tasks_and_fails_pending() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    fail_pending = MagicMock()
    lifecycle = AsyncIntegrationLifecycle(
        owner="test-client",
        logger=logging.getLogger(__name__),
        fail_pending_operations=fail_pending,
    )

    async def _worker() -> None:
        started.set()
        await release.wait()

    lifecycle.spawn_task("worker", _worker())
    await started.wait()

    await lifecycle.shutdown(fail_pending_reason="transport closed")

    fail_pending.assert_called_once_with("transport closed")


@pytest.mark.asyncio
async def test_shutdown_reports_background_task_errors() -> None:
    errors: list[tuple[str, str]] = []

    def _on_error(task_name: str, error: Exception) -> None:
        errors.append((task_name, str(error)))

    lifecycle = AsyncIntegrationLifecycle(
        owner="test-client",
        logger=logging.getLogger(__name__),
        on_task_error=_on_error,
    )

    async def _fails() -> None:
        raise RuntimeError("boom")

    lifecycle.spawn_task("failing_task", _fails())
    await asyncio.sleep(0)

    await lifecycle.shutdown()

    assert errors == [("failing_task", "boom")]
