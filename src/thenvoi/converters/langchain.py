"""LangChain/LangGraph history converter."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from thenvoi.core.protocols import HistoryConverter

logger = logging.getLogger(__name__)

# Type alias for LangChain messages
LangChainMessages = list[AIMessage | HumanMessage | ToolMessage]


class LangChainHistoryConverter(HistoryConverter[LangChainMessages]):
    """
    Converts platform history to LangChain message types.

    Handles:
    - tool_call + tool_result pairing with tool_call_id extraction
    - Skipping redundant assistant text messages
    - User messages as HumanMessage
    """

    def convert(self, raw: list[dict[str, Any]]) -> LangChainMessages:
        """Convert platform history to LangChain messages."""
        messages: LangChainMessages = []
        pending_tool_calls: list[dict[str, Any]] = []

        for hist in raw:
            message_type = hist.get("message_type", "text")
            content = hist.get("content", "")
            role = hist.get("role")
            sender_name = hist.get("sender_name", "")

            if message_type == "tool_call":
                try:
                    event = json.loads(content)
                    pending_tool_calls.append(event)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool_call: {content[:100]}")

            elif message_type == "tool_result":
                try:
                    event = json.loads(content)
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")

                    tool_call_id = self._extract_tool_call_id(str(output))
                    if not tool_call_id:
                        tool_call_id = event.get("run_id", "unknown")

                    # Match with pending tool call
                    matching_call = None
                    for i, call in enumerate(pending_tool_calls):
                        if call.get("name") == tool_name:
                            matching_call = pending_tool_calls.pop(i)
                            break

                    if matching_call:
                        tool_input = matching_call.get("data", {}).get("input", {})
                        messages.append(
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": tool_call_id,
                                        "name": tool_name,
                                        "args": tool_input,
                                    }
                                ],
                            )
                        )

                    messages.append(
                        ToolMessage(content=str(output), tool_call_id=tool_call_id)
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool_result: {content[:100]}")

            elif message_type == "text":
                if role == "assistant":
                    # Skip redundant assistant text
                    logger.debug(f"Skipping redundant assistant text: {content[:50]}")
                else:
                    messages.append(HumanMessage(content=f"[{sender_name}]: {content}"))

        # Warn about unmatched tool calls
        if pending_tool_calls:
            logger.warning(
                f"Found {len(pending_tool_calls)} tool_calls without matching tool_results"
            )

        return messages

    @staticmethod
    def _extract_tool_call_id(output: str) -> str | None:
        """Extract tool_call_id from tool output string."""
        match = re.search(r"tool_call_id='([^']+)'", output)
        return match.group(1) if match else None
