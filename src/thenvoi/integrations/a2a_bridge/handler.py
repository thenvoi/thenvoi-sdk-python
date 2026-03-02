"""Protocol for bridge message handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from thenvoi.runtime.types import PlatformMessage
    from thenvoi.runtime.tools import AgentTools


@dataclass(frozen=True)
class HandlerResult:
    """Outcome contract for bridge handlers."""

    status: Literal["handled", "ignored", "error"] = "handled"
    detail: str | None = None

    def __post_init__(self) -> None:
        allowed_statuses = {"handled", "ignored", "error"}
        if self.status not in allowed_statuses:
            raise ValueError(
                f"Invalid handler status '{self.status}'. "
                f"Expected one of: {', '.join(sorted(allowed_statuses))}."
            )

    @classmethod
    def handled(cls, detail: str | None = None) -> HandlerResult:
        """Return a successful handling outcome."""
        return cls(status="handled", detail=detail)

    @classmethod
    def ignored(cls, detail: str | None = None) -> HandlerResult:
        """Return an ignored outcome for intentionally skipped messages."""
        return cls(status="ignored", detail=detail)

    @classmethod
    def error(cls, detail: str | None = None) -> HandlerResult:
        """Return a handled-but-error outcome without raising an exception."""
        return cls(status="error", detail=detail)


class BaseHandler(Protocol):
    """Protocol for bridge message handlers.

    Implement this to create handlers for specific agents.
    Each handler receives parsed @mention messages and can
    respond using the provided AgentTools.
    """

    async def handle(
        self,
        message: PlatformMessage,
        mentioned_agent: str,
        tools: AgentTools,
    ) -> HandlerResult:
        """Handle a routed @mention message.

        Args:
            message: Normalized platform message payload.
            mentioned_agent: The agent name that was @mentioned.
            tools: AgentTools instance bound to the room for sending responses.

        Returns:
            HandlerResult: Explicit outcome metadata for routing behavior.
        """
        ...
