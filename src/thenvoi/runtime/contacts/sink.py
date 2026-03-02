"""Contact event sink interface for hub/broadcast integrations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from thenvoi.platform.event import MessageEvent

HubEventCallback = Callable[[str, MessageEvent], Awaitable[None]]
HubInitCallback = Callable[[str, str], Awaitable[None]]


class ContactEventSink(Protocol):
    """Explicit boundary for contact event side effects."""

    @property
    def hub_enabled(self) -> bool:
        """Whether hub-event injection is supported by this sink."""

    def broadcast(self, message: str) -> None:
        """Emit a broadcast message for contact-added/removed events."""

    async def initialize_hub_room(self, room_id: str, system_prompt: str) -> None:
        """Initialize hub-room context before first synthetic message injection."""

    async def inject_hub_event(self, room_id: str, event: MessageEvent) -> None:
        """Inject a synthetic contact event into the hub-room runtime."""


class ContactRuntimePort(Protocol):
    """Runtime-facing port used by contact orchestration."""

    @property
    def hub_contact_events_enabled(self) -> bool:
        """Whether hub-event injection is available in the runtime."""

    def queue_contact_broadcast(self, message: str) -> None:
        """Queue a contact broadcast for active runtime sessions."""

    async def initialize_contact_hub_room(
        self,
        room_id: str,
        system_prompt: str,
    ) -> None:
        """Inject initial system prompt into the runtime hub room."""

    async def inject_contact_hub_event(self, room_id: str, event: MessageEvent) -> None:
        """Inject a synthesized contact event into the runtime hub room."""


class RuntimeContactEventSink:
    """ContactEventSink adapter that targets an explicit runtime port."""

    def __init__(self, runtime_port: ContactRuntimePort) -> None:
        self._runtime_port = runtime_port

    @property
    def hub_enabled(self) -> bool:
        return self._runtime_port.hub_contact_events_enabled

    def broadcast(self, message: str) -> None:
        self._runtime_port.queue_contact_broadcast(message)

    async def initialize_hub_room(self, room_id: str, system_prompt: str) -> None:
        await self._runtime_port.initialize_contact_hub_room(room_id, system_prompt)

    async def inject_hub_event(self, room_id: str, event: MessageEvent) -> None:
        await self._runtime_port.inject_contact_hub_event(room_id, event)


class CallbackContactEventSink:
    """Adapter that preserves existing callback wiring behind ContactEventSink."""

    def __init__(
        self,
        *,
        on_broadcast: Callable[[str], None] | None,
        on_hub_event: HubEventCallback | None,
        on_hub_init: HubInitCallback | None,
    ) -> None:
        self._on_broadcast = on_broadcast
        self._on_hub_event = on_hub_event
        self._on_hub_init = on_hub_init

    @property
    def hub_enabled(self) -> bool:
        return self._on_hub_event is not None

    def broadcast(self, message: str) -> None:
        if self._on_broadcast is not None:
            self._on_broadcast(message)

    async def initialize_hub_room(self, room_id: str, system_prompt: str) -> None:
        if self._on_hub_init is not None:
            await self._on_hub_init(room_id, system_prompt)

    async def inject_hub_event(self, room_id: str, event: MessageEvent) -> None:
        if self._on_hub_event is None:
            raise RuntimeError("No hub event sink configured")
        await self._on_hub_event(room_id, event)


__all__ = [
    "ContactEventSink",
    "ContactRuntimePort",
    "RuntimeContactEventSink",
    "CallbackContactEventSink",
    "HubEventCallback",
    "HubInitCallback",
]
