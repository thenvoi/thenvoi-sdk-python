"""Tests for shutdown-aware A2A bridge event pump behavior."""

from __future__ import annotations

import asyncio

import pytest

from thenvoi.integrations.a2a_bridge.event_pump import ShutdownAwareEventPump


@pytest.mark.asyncio
async def test_event_pump_exits_immediately_when_shutdown_already_set() -> None:
    shutdown_event = asyncio.Event()
    shutdown_event.set()
    pump = ShutdownAwareEventPump(shutdown_event)
    handled: list[object] = []

    async def _event_stream():
        while True:
            await asyncio.sleep(1)
            yield {"event": "never-consumed"}

    async def _handle_event(event: object) -> None:
        handled.append(event)

    await pump.run(_event_stream(), handle_event=_handle_event)

    assert handled == []


@pytest.mark.asyncio
async def test_event_pump_cancels_inflight_handler_on_shutdown() -> None:
    shutdown_event = asyncio.Event()
    pump = ShutdownAwareEventPump(shutdown_event)
    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()

    async def _event_stream():
        yield {"event": "first"}
        await asyncio.sleep(10)

    async def _handle_event(_event: object) -> None:
        handler_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            handler_cancelled.set()
            raise

    run_task = asyncio.create_task(pump.run(_event_stream(), handle_event=_handle_event))
    await asyncio.wait_for(handler_started.wait(), timeout=1.0)
    shutdown_event.set()
    await asyncio.wait_for(run_task, timeout=1.0)

    assert handler_cancelled.is_set()
