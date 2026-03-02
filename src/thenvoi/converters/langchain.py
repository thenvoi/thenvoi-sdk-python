"""LangChain/LangGraph history converter."""

from __future__ import annotations

import logging
import re
from typing import Any

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
except ImportError as e:
    raise ImportError(
        "LangChain dependencies not installed. "
        "Install with: uv add thenvoi-sdk[langgraph]"
    ) from e

from thenvoi.core.protocols import HistoryConverter
from thenvoi.converters.normalized_events import (
    ToolCallHistoryEvent,
    ToolResultHistoryEvent,
    TextHistoryEvent,
    normalize_history_events,
)

logger = logging.getLogger(__name__)

# Type alias for LangChain messages
LangChainMessages = list[AIMessage | HumanMessage | ToolMessage]


class LangChainHistoryConverter(HistoryConverter[LangChainMessages]):
    """
    Converts platform history to LangChain message types.

    Handles:
    - tool_call + tool_result pairing with tool_call_id extraction
    - Skipping this agent's redundant text messages
    - Including other agents' messages as HumanMessage
    - User messages as HumanMessage
    """

    def __init__(self, agent_name: str = ""):
        """
        Initialize converter.

        Args:
            agent_name: Name of this agent. Messages from this agent are skipped
                       (they're redundant with tool calls). Messages from other
                       agents are included as HumanMessage.
        """
        self._agent_name = agent_name

    def set_agent_name(self, name: str) -> None:
        """
        Set agent name so converter knows which messages to skip.

        Args:
            name: Name of this agent
        """
        self._agent_name = name

    def convert(self, raw: list[dict[str, Any]]) -> LangChainMessages:
        """Convert platform history to LangChain messages."""
        messages: LangChainMessages = []
        pending_tool_calls: list[ToolCallHistoryEvent] = []

        for event in normalize_history_events(raw):
            if isinstance(event, ToolCallHistoryEvent):
                pending_tool_calls.append(event)

            elif isinstance(event, ToolResultHistoryEvent):
                tool_name = event.name
                output = event.output
                tool_call_id = event.tool_call_id

                matching_call: ToolCallHistoryEvent | None = None
                for i, pending in enumerate(pending_tool_calls):
                    if pending.name == tool_name:
                        matching_call = pending_tool_calls.pop(i)
                        break

                if matching_call:
                    messages.append(
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": tool_call_id,
                                    "name": tool_name,
                                    "args": matching_call.args,
                                }
                            ],
                        )
                    )

                messages.append(
                    ToolMessage(content=str(output), tool_call_id=tool_call_id)
                )

            elif isinstance(event, TextHistoryEvent):
                if event.role == "assistant" and event.sender_name == self._agent_name:
                    # Skip only THIS agent's text (redundant with tool calls)
                    logger.debug("Skipping own message: %s", event.content[:50])
                else:
                    # Include user messages AND other agents' messages
                    messages.append(
                        HumanMessage(
                            content=f"[{event.sender_name}]: {event.content}"
                        )
                    )

        # Warn about unmatched tool calls
        if pending_tool_calls:
            logger.warning(
                "Found %s tool_calls without matching tool_results",
                len(pending_tool_calls),
            )

        return messages

    @staticmethod
    def _extract_tool_call_id(output: str) -> str | None:
        """Extract tool_call_id from tool output string."""
        match = re.search(r"tool_call_id='([^']+)'", output)
        return match.group(1) if match else None
