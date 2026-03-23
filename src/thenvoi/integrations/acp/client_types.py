"""Types for ACP client adapter."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from acp.interfaces import Client

from thenvoi.integrations.acp.types import CollectedChunk

logger = logging.getLogger(__name__)


@dataclass
class ACPClientSessionState:
    """Session state for ACP client adapter rehydration.

    Attributes:
        room_to_session: Mapping of Thenvoi room_id to ACP session_id.
    """

    room_to_session: dict[str, str] = field(default_factory=dict)


class ThenvoiACPClient(Client):  # type: ignore[misc]  # ACP Client has optional methods treated as abstract by pyrefly
    """ACP Client implementation that collects session_update responses.

    When the external ACP agent calls session_update, we buffer the
    chunks (text, thoughts, tool calls, etc.) per session so the adapter
    can post them back to Thenvoi with full type fidelity.

    Buffers are keyed by session_id to allow concurrent rooms without
    a global lock.
    """

    def __init__(self) -> None:
        self._session_chunks: dict[str, list[CollectedChunk]] = {}
        self._permission_handlers: dict[
            str, Callable[..., Awaitable[dict[str, object]]]
        ] = {}  # session_id -> handler

    async def session_update(
        self, session_id: str, update: object, **kwargs: object
    ) -> None:
        """Buffer chunks from session updates, keyed by session_id.

        Inspects the session_update discriminator to classify the chunk
        type and extract relevant content.

        Args:
            session_id: The ACP session identifier.
            update: The session update from the external ACP agent.
            **kwargs: Additional keyword arguments.
        """
        discriminator = getattr(update, "session_update", None)
        chunk: CollectedChunk | None = None

        match discriminator:
            case "agent_message_chunk":
                text = self._extract_text_from_content(update)
                chunk = CollectedChunk(chunk_type="text", content=text)
            case "agent_thought_chunk":
                text = self._extract_text_from_content(update)
                chunk = CollectedChunk(chunk_type="thought", content=text)
            case "tool_call":
                tool_call_id = getattr(update, "tool_call_id", "")
                title = getattr(update, "title", "")
                raw_input = getattr(update, "raw_input", None)
                chunk = CollectedChunk(
                    chunk_type="tool_call",
                    content=title,
                    metadata={
                        "tool_call_id": tool_call_id,
                        "raw_input": raw_input,
                        "status": getattr(update, "status", "in_progress"),
                    },
                )
            case "tool_call_update":
                tool_call_id = getattr(update, "tool_call_id", "")
                raw_output = getattr(update, "raw_output", "")
                chunk = CollectedChunk(
                    chunk_type="tool_result",
                    content=str(raw_output) if raw_output else "",
                    metadata={
                        "tool_call_id": tool_call_id,
                        "status": getattr(update, "status", "completed"),
                    },
                )
            case "plan":
                entries = getattr(update, "entries", [])
                plan_text = "\n".join(getattr(e, "content", str(e)) for e in entries)
                chunk = CollectedChunk(chunk_type="plan", content=plan_text)
            case _:
                text = self._extract_text_from_content(update)
                if text:
                    chunk = CollectedChunk(chunk_type="text", content=text)

        if chunk is not None:
            self._session_chunks.setdefault(session_id, []).append(chunk)

    async def request_permission(  # type: ignore[override]  # ACP Client uses specific types; we widen to object
        self,
        options: object,
        session_id: str,
        tool_call: object,
        **kwargs: object,
    ) -> dict[str, object]:
        """Handle permission requests from ACP agent.

        If a permission handler is set, delegates to it. Otherwise
        auto-cancels the request.

        Args:
            options: Permission options.
            session_id: The ACP session identifier.
            tool_call: The tool call requesting permission.
            **kwargs: Additional keyword arguments.

        Returns:
            Permission outcome dict.
        """
        handler = self._permission_handlers.get(session_id)
        if handler:
            return await handler(
                options=options,
                session_id=session_id,
                tool_call=tool_call,
                **kwargs,
            )

        logger.debug("Auto-cancelling permission request for session %s", session_id)
        return {"outcome": {"outcome": "cancelled"}}

    def set_permission_handler(
        self,
        session_id: str,
        handler: Callable[..., Awaitable[dict[str, object]]] | None,
    ) -> None:
        """Set a callback for handling permission requests for a session.

        Args:
            session_id: The ACP session to set the handler for.
            handler: Async callable that receives permission request params
                and returns an outcome dict. Set to None to auto-cancel.
        """
        if handler is None:
            self._permission_handlers.pop(session_id, None)
        else:
            self._permission_handlers[session_id] = handler

    def reset_session(self, session_id: str) -> None:
        """Clear the collected chunks buffer and permission handler for a session.

        Args:
            session_id: The ACP session to clear.
        """
        self._session_chunks.pop(session_id, None)
        self._permission_handlers.pop(session_id, None)

    def get_collected_text(self, session_id: str | None = None) -> str:
        """Return collected text chunks as a single string.

        Args:
            session_id: If provided, return text for this session only.
                If None, return text from all sessions (backward-compat).
        """
        if session_id is not None:
            chunks = self._session_chunks.get(session_id, [])
        else:
            chunks = [c for cs in self._session_chunks.values() for c in cs]
        return "".join(c.content for c in chunks if c.chunk_type == "text")

    def get_collected_chunks(
        self, session_id: str | None = None
    ) -> list[CollectedChunk]:
        """Return collected chunks with full type information.

        Args:
            session_id: If provided, return chunks for this session only.
                If None, return chunks from all sessions (backward-compat).
        """
        if session_id is not None:
            return list(self._session_chunks.get(session_id, []))
        return [c for cs in self._session_chunks.values() for c in cs]

    async def ext_method(
        self, method: str, params: dict[str, object]
    ) -> dict[str, object]:
        """Handle extension methods from ACP agents (e.g., Cursor).

        Cursor sends extension methods like cursor/ask_question and
        cursor/create_plan that require a response. We auto-approve
        these since there's no interactive UI in the Thenvoi bridge.

        Args:
            method: Extension method name.
            params: Method parameters.

        Returns:
            Response dict.
        """
        logger.debug("Client ext_method: %s, params=%s", method, params)

        if method == "cursor/ask_question":
            options = params.get("options", [])
            if options:
                first = options[0] if isinstance(options, list) else options
                option_id = (
                    first.get("optionId", "0")
                    if isinstance(first, dict)
                    else getattr(first, "optionId", "0")
                )
                return {"outcome": {"type": "selected", "optionId": option_id}}
            return {"outcome": {"type": "cancelled"}}

        if method == "cursor/create_plan":
            return {"outcome": {"type": "approved"}}

        return {}

    async def ext_notification(self, method: str, params: dict[str, object]) -> None:
        """Handle extension notifications from ACP agents (e.g., Cursor).

        Cursor sends notifications like cursor/update_todos and
        cursor/task. We collect these as chunks so the adapter can
        forward them to the Thenvoi platform.

        Args:
            method: Extension notification name.
            params: Notification parameters.
        """
        logger.debug("Client ext_notification: %s, params=%s", method, params)

        session_id = str(params.get("sessionId") or params.get("session_id") or "")
        if not session_id:
            return

        if method == "cursor/update_todos":
            todos = params.get("todos", [])
            if todos and isinstance(todos, list):
                lines = []
                for t in todos:
                    if isinstance(t, dict):
                        done = t.get("completed", False)
                        text = t.get("content", "")
                        lines.append(f"- [{'x' if done else ' '}] {text}")
                if lines:
                    chunk = CollectedChunk(
                        chunk_type="plan",
                        content="\n".join(lines),
                    )
                    self._session_chunks.setdefault(session_id, []).append(chunk)

        elif method == "cursor/task":
            result = str(params.get("result", ""))
            if result:
                chunk = CollectedChunk(
                    chunk_type="text",
                    content=f"[Task completed] {result}",
                )
                self._session_chunks.setdefault(session_id, []).append(chunk)

    @staticmethod
    def _extract_text_from_content(update: object) -> str:
        """Extract text from a session_update's content field.

        Args:
            update: The session update object.

        Returns:
            Extracted text, or empty string.
        """
        content = getattr(update, "content", None)
        if content is None:
            return ""
        text = getattr(content, "text", None)
        if text is None and isinstance(content, dict):
            text = content.get("text", "")
        return str(text) if text else ""
