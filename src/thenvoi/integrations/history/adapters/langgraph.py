"""
LangGraph adapter - converts normalized messages to LangChain message types.

Handles:
- NormalizedUserText → HumanMessage
- NormalizedToolExchange → AIMessage (with tool_calls) + ToolMessage
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..normalized import (
    NormalizedMessage,
    NormalizedToolExchange,
    NormalizedUserText,
)


def to_langgraph_messages(
    normalized: list[NormalizedMessage],
) -> list[AIMessage | HumanMessage | ToolMessage]:
    """
    Convert normalized messages to LangChain message types.

    Args:
        normalized: List of normalized messages from parser

    Returns:
        List of LangChain messages ready for graph input.

    Example:
        >>> from thenvoi.integrations.history import parse_platform_history
        >>> normalized = parse_platform_history(raw_history)
        >>> messages = to_langgraph_messages(normalized)
        >>> # Ready for graph.invoke({"messages": messages})
    """
    messages: list[AIMessage | HumanMessage | ToolMessage] = []

    for msg in normalized:
        if isinstance(msg, NormalizedUserText):
            formatted = f"[{msg.sender_name}]: {msg.content}"
            messages.append(HumanMessage(content=formatted))

        elif isinstance(msg, NormalizedToolExchange):
            # AIMessage with tool_calls
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": msg.tool_id,
                            "name": msg.tool_name,
                            "args": msg.input_args,
                        }
                    ],
                )
            )

            # ToolMessage with result
            messages.append(
                ToolMessage(
                    content=msg.output,
                    tool_call_id=msg.tool_id,
                )
            )

        # Skip NormalizedSystemMessage - LangGraph handles system prompts separately

    return messages
