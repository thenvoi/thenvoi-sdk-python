"""Output adapters for normalizing framework-specific converter results.

Each adapter provides a uniform interface for asserting on converter output,
regardless of the underlying framework's message format.
"""

from __future__ import annotations

from typing import Any, Protocol


class OutputAdapter(Protocol):
    """Protocol for normalizing converter output across frameworks."""

    def result_length(self, result: Any) -> int: ...
    def get_content(self, result: Any, index: int) -> str: ...
    def get_role(self, result: Any, index: int) -> str: ...
    def is_empty(self, result: Any) -> bool: ...
    def content_contains(self, result: Any, substring: str) -> bool: ...


class DictListOutputAdapter:
    """Adapter for Anthropic converter output (list[dict] with tool_use blocks)."""

    def result_length(self, result: list[dict[str, Any]]) -> int:
        return len(result)

    def get_content(self, result: list[dict[str, Any]], index: int) -> str:
        content = result[index]["content"]
        if isinstance(content, list):
            # Tool use blocks -- return first text block or repr
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"]
            return str(content)
        return content

    def get_role(self, result: list[dict[str, Any]], index: int) -> str:
        return result[index]["role"]

    def is_empty(self, result: list[dict[str, Any]]) -> bool:
        return len(result) == 0

    def content_contains(self, result: list[dict[str, Any]], substring: str) -> bool:
        for msg in result:
            content = msg["content"]
            if isinstance(content, str) and substring in content:
                return True
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and substring in str(block):
                        return True
        return False


class LangChainOutputAdapter:
    """Adapter for LangChain converter output (list of Message objects)."""

    def result_length(self, result: list) -> int:
        return len(result)

    def get_content(self, result: list, index: int) -> str:
        return result[index].content

    def get_role(self, result: list, index: int) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        msg = result[index]
        if isinstance(msg, HumanMessage):
            return "user"
        if isinstance(msg, AIMessage):
            return "assistant"
        if isinstance(msg, ToolMessage):
            return "tool"
        return "unknown"

    def is_empty(self, result: list) -> bool:
        return len(result) == 0

    def content_contains(self, result: list, substring: str) -> bool:
        for msg in result:
            if substring in msg.content:
                return True
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    if substring in tc.get("name", ""):
                        return True
        return False


class PydanticAIOutputAdapter:
    """Adapter for PydanticAI converter output (list of ModelRequest/ModelResponse)."""

    def result_length(self, result: list) -> int:
        return len(result)

    def get_content(self, result: list, index: int) -> str:
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            UserPromptPart,
        )

        msg = result[index]
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    return part.content
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    return part.content
        return str(msg)

    def get_role(self, result: list, index: int) -> str:
        from pydantic_ai.messages import ModelRequest, ModelResponse

        msg = result[index]
        if isinstance(msg, ModelRequest):
            return "user"
        if isinstance(msg, ModelResponse):
            return "assistant"
        return "unknown"

    def is_empty(self, result: list) -> bool:
        return len(result) == 0

    def content_contains(self, result: list, substring: str) -> bool:
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )

        for msg in result:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart) and substring in part.content:
                        return True
                    if isinstance(part, ToolReturnPart) and substring in (
                        part.content or ""
                    ):
                        return True
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart) and substring in part.content:
                        return True
                    if isinstance(part, ToolCallPart) and substring in (
                        part.tool_name or ""
                    ):
                        return True
        return False


class StringOutputAdapter:
    """Adapter for ClaudeSDK converter output (newline-joined string).

    Splits on ``"\\n"`` to recover individual ``[sender]: content`` or
    JSON tool-event messages.  Validates that each segment starts with
    ``[`` or ``{`` to catch silent miscounts from embedded newlines.
    """

    @staticmethod
    def _split_messages(result: str) -> list[str]:
        """Split the joined string back into logical messages.

        Raises ``AssertionError`` if any segment doesn't start with
        ``[`` or ``{`` (likely a fragment from an embedded newline).
        """
        if not result:
            return []
        segments = result.split("\n")
        for i, seg in enumerate(segments):
            stripped = seg.lstrip()
            if (
                stripped
                and not stripped.startswith("[")
                and not stripped.startswith("{")
            ):
                raise AssertionError(
                    f"StringOutputAdapter: segment {i} does not look like a "
                    f"complete message (possible embedded newline in content).\n"
                    f"  Segment: {seg!r}\n"
                    f"  Full result: {result!r}"
                )
        return segments

    def result_length(self, result: str) -> int:
        return len(self._split_messages(result))

    def get_content(self, result: str, index: int) -> str:
        return self._split_messages(result)[index]

    def get_role(self, result: str, index: int) -> str:
        raise NotImplementedError(
            "StringOutputAdapter.get_role() is not supported. "
            "ClaudeSDK returns a flat string with no structured role field. "
            "Conformance tests that need roles should skip when "
            "has_role_concept=False."
        )

    def is_empty(self, result: str) -> bool:
        return result == ""

    def content_contains(self, result: str, substring: str) -> bool:
        return substring in result


class SimpleDictListOutputAdapter:
    """Adapter for CrewAI/Parlant converter output (list[dict] with sender/sender_type)."""

    def result_length(self, result: list[dict[str, Any]]) -> int:
        return len(result)

    def get_content(self, result: list[dict[str, Any]], index: int) -> str:
        return result[index]["content"]

    def get_role(self, result: list[dict[str, Any]], index: int) -> str:
        return result[index]["role"]

    def is_empty(self, result: list[dict[str, Any]]) -> bool:
        return len(result) == 0

    def content_contains(self, result: list[dict[str, Any]], substring: str) -> bool:
        return any(substring in msg.get("content", "") for msg in result)
