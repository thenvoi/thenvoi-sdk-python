"""CrewAI-specific tool schema overrides."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MessageType = Literal["thought", "error", "task"]


class CrewAISendMessageInput(BaseModel):
    """Backward-compatible send-message schema for CrewAI tools."""

    content: str = Field(..., description="The message content to send")
    mentions: list[str] = Field(
        default_factory=list,
        description=(
            "List of participant handles to @mention (e.g., "
            "['@john', '@john/weather-agent'])."
        ),
    )

    @field_validator("mentions", mode="before")
    @classmethod
    def normalize_mentions(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        raise TypeError("mentions must be a list of handles")


class CrewAISendEventInput(BaseModel):
    """Backward-compatible send-event schema for CrewAI tools."""

    content: str = Field(..., description="Human-readable event content")
    message_type: MessageType = Field(
        default="thought",
        description="Type of event: 'thought', 'error', or 'task'",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured event metadata",
    )


CREWAI_SCHEMA_OVERRIDES: dict[str, type[BaseModel]] = {
    # CrewAI historically normalized missing mentions to [].
    "thenvoi_send_message": CrewAISendMessageInput,
    # Keep backward-compatible default for thought reporting.
    "thenvoi_send_event": CrewAISendEventInput,
}

