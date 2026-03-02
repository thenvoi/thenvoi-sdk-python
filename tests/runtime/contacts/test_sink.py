"""Tests for contact event sink adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.platform.event import MessageEvent
from thenvoi.runtime.contacts.sink import (
    CallbackContactEventSink,
    RuntimeContactEventSink,
)


def test_runtime_sink_delegates_to_runtime_port() -> None:
    runtime_port = MagicMock()
    runtime_port.hub_contact_events_enabled = True
    runtime_port.queue_contact_broadcast = MagicMock()
    runtime_port.initialize_contact_hub_room = AsyncMock()
    runtime_port.inject_contact_hub_event = AsyncMock()

    sink = RuntimeContactEventSink(runtime_port)
    assert sink.hub_enabled is True
    sink.broadcast("hello")
    runtime_port.queue_contact_broadcast.assert_called_once_with("hello")


@pytest.mark.asyncio
async def test_runtime_sink_async_methods_delegate() -> None:
    runtime_port = MagicMock()
    runtime_port.hub_contact_events_enabled = True
    runtime_port.initialize_contact_hub_room = AsyncMock()
    runtime_port.inject_contact_hub_event = AsyncMock()

    sink = RuntimeContactEventSink(runtime_port)
    event = MessageEvent()

    await sink.initialize_hub_room("room-1", "system prompt")
    await sink.inject_hub_event("room-1", event)

    runtime_port.initialize_contact_hub_room.assert_awaited_once_with(
        "room-1",
        "system prompt",
    )
    runtime_port.inject_contact_hub_event.assert_awaited_once_with("room-1", event)


@pytest.mark.asyncio
async def test_callback_sink_uses_configured_callbacks() -> None:
    on_broadcast = MagicMock()
    on_hub_event = AsyncMock()
    on_hub_init = AsyncMock()
    event = MessageEvent()
    sink = CallbackContactEventSink(
        on_broadcast=on_broadcast,
        on_hub_event=on_hub_event,
        on_hub_init=on_hub_init,
    )

    assert sink.hub_enabled is True
    sink.broadcast("broadcast")
    await sink.initialize_hub_room("room-1", "sys")
    await sink.inject_hub_event("room-1", event)

    on_broadcast.assert_called_once_with("broadcast")
    on_hub_init.assert_awaited_once_with("room-1", "sys")
    on_hub_event.assert_awaited_once_with("room-1", event)


@pytest.mark.asyncio
async def test_callback_sink_without_event_handler_raises() -> None:
    sink = CallbackContactEventSink(
        on_broadcast=None,
        on_hub_event=None,
        on_hub_init=None,
    )

    assert sink.hub_enabled is False
    sink.broadcast("ignored")
    await sink.initialize_hub_room("room-1", "sys")

    with pytest.raises(RuntimeError, match="No hub event sink configured"):
        await sink.inject_hub_event("room-1", MessageEvent())
