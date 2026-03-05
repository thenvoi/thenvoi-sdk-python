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
        self._permission_handler: Callable[..., Awaitable[dict[str, object]]] | None = (
            None
        )

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
        if self._permission_handler:
            return await self._permission_handler(
                options=options,
                session_id=session_id,
                tool_call=tool_call,
                **kwargs,
            )

        logger.debug("Auto-cancelling permission request for session %s", session_id)
        return {"outcome": {"outcome": "cancelled"}}

    def set_permission_handler(
        self, handler: Callable[..., Awaitable[dict[str, object]]] | None
    ) -> None:
        """Set a callback for handling permission requests.

        Args:
            handler: Async callable that receives permission request params
                and returns an outcome dict. Set to None to auto-cancel.
        """
        self._permission_handler = handler

    def reset_session(self, session_id: str) -> None:
        """Clear the collected chunks buffer for a specific session.

        Args:
            session_id: The ACP session to clear.
        """
        self._session_chunks.pop(session_id, None)

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
