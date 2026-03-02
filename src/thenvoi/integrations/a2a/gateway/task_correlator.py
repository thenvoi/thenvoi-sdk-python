"""Task correlation and A2A status translation for gateway flows."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from a2a.types import (
    Message as A2AMessage,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from thenvoi.core.types import PlatformMessage
from thenvoi.integrations.a2a.gateway.types import PendingA2ATask


class GatewayTaskCorrelator:
    """Track in-flight A2A tasks and bridge platform responses to SSE events."""

    def __init__(self) -> None:
        self.pending_tasks: dict[str, PendingA2ATask] = {}

    def create_task(self, context_id: str) -> Task:
        """Create a new A2A task with working status."""
        return Task(
            id=str(uuid4()),
            context_id=context_id,
            status=TaskStatus(state=TaskState.working),
        )

    def register_pending(
        self,
        *,
        room_id: str,
        context_id: str,
        peer_id: str,
    ) -> PendingA2ATask:
        """Create and store a pending task for room-scoped correlation."""
        pending = PendingA2ATask(
            task=self.create_task(context_id),
            sse_queue=asyncio.Queue(),
            peer_id=peer_id,
        )
        self.pending_tasks[room_id] = pending
        return pending

    async def ingest_platform_message(
        self,
        *,
        room_id: str,
        message: PlatformMessage,
    ) -> TaskStatusUpdateEvent | None:
        """Translate a platform message and push it to the room's pending SSE queue."""
        pending = self.pending_tasks.get(room_id)
        if pending is None:
            return None

        event = self.translate_to_a2a(message, pending.task)
        await pending.sse_queue.put(event)
        if event.final:
            self.pending_tasks.pop(room_id, None)
        return event

    def pop_room(self, room_id: str) -> None:
        """Drop any pending task for the room."""
        self.pending_tasks.pop(room_id, None)

    def translate_to_a2a(self, msg: PlatformMessage, task: Task) -> TaskStatusUpdateEvent:
        """Convert platform message to A2A task update event."""
        message_type = getattr(msg, "message_type", "text")

        if message_type == "error":
            state = TaskState.failed
            final = True
        elif message_type in ("thought", "tool_call", "tool_result"):
            state = TaskState.working
            final = False
        else:
            state = TaskState.completed
            final = True

        task.status = TaskStatus(
            state=state,
            message=A2AMessage(
                role=Role.agent,
                message_id=str(uuid4()),
                parts=[Part(root=TextPart(text=msg.content))],
            ),
        )

        return TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=task.status,
            final=final,
        )
