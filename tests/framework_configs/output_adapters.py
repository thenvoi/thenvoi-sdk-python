"""Output adapters for normalizing framework-specific converter results.

Each adapter provides a uniform interface for asserting on converter output,
regardless of the underlying framework's message format.
"""

from __future__ import annotations

import re
from typing import Any, Protocol


class OutputAdapter(Protocol):
    """Protocol for normalizing converter output across frameworks."""

    def result_length(self, result: Any) -> int: ...
    def get_content(self, result: Any, index: int) -> str: ...
    def get_role(self, result: Any, index: int) -> str: ...
    def is_empty(self, result: Any) -> bool: ...
    def content_contains(self, result: Any, substring: str) -> bool: ...
    def assert_element_type(
        self, result: Any, index: int, expected_role: str
    ) -> None: ...
    def assert_sender_metadata(
        self, result: Any, index: int, sender_name: str, sender_type: str | None = None
    ) -> None: ...
    def assert_result_type(self, result: Any) -> None: ...


class BaseDictListOutputAdapter:
    """Shared logic for adapters whose output is ``list[dict[str, Any]]``."""

    def assert_result_type(self, result: list[dict[str, Any]]) -> None:
        assert isinstance(result, list), f"Expected list, got {type(result).__name__}"

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

    def assert_element_type(
        self, result: list[dict[str, Any]], index: int, expected_role: str
    ) -> None:
        msg = result[index]
        assert isinstance(msg, dict), f"Expected dict, got {type(msg).__name__}"
        assert "role" in msg, "Missing 'role' key"
        assert "content" in msg, "Missing 'content' key"
        assert msg["role"] == expected_role, (
            f"Expected role={expected_role!r}, got {msg['role']!r}"
        )

    def assert_sender_metadata(
        self,
        result: list[dict[str, Any]],
        index: int,
        sender_name: str,
        sender_type: str | None = None,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}.assert_sender_metadata() is not supported. "
            "Ensure has_sender_metadata=False in the ConverterConfig."
        )


class DictListOutputAdapter(BaseDictListOutputAdapter):
    """Adapter for Anthropic converter output (list[dict] with tool_use blocks)."""

    def get_content(self, result: list[dict[str, Any]], index: int) -> str:
        content = result[index]["content"]
        if isinstance(content, list):
            # Tool use blocks -- return first text block or repr
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"]
            return str(content)
        return content

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

    def assert_result_type(self, result: list) -> None:
        assert isinstance(result, list), f"Expected list, got {type(result).__name__}"

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
                    if substring in str(tc.get("args", {})):
                        return True
        return False

    def assert_element_type(self, result: list, index: int, expected_role: str) -> None:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        msg = result[index]
        type_map: dict[str, type] = {
            "user": HumanMessage,
            "assistant": AIMessage,
            "tool": ToolMessage,
        }
        expected_type = type_map.get(expected_role)
        assert expected_type is not None, f"Unknown expected_role: {expected_role!r}"
        assert isinstance(msg, expected_type), (
            f"Expected {expected_type.__name__}, got {type(msg).__name__}"
        )

    def assert_sender_metadata(
        self,
        result: list,
        index: int,
        sender_name: str,
        sender_type: str | None = None,
    ) -> None:
        raise NotImplementedError(
            "LangChainOutputAdapter.assert_sender_metadata() is not supported. "
            "LangChain messages do not include sender metadata. "
            "Ensure has_sender_metadata=False in the ConverterConfig."
        )


class PydanticAIOutputAdapter:
    """Adapter for PydanticAI converter output (list of ModelRequest/ModelResponse)."""

    def assert_result_type(self, result: list) -> None:
        assert isinstance(result, list), f"Expected list, got {type(result).__name__}"

    def result_length(self, result: list) -> int:
        return len(result)

    def get_content(self, result: list, index: int) -> str:
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )

        msg = result[index]
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    return part.content
                if isinstance(part, ToolReturnPart) and part.content:
                    return part.content
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    return part.content
                if isinstance(part, ToolCallPart) and part.tool_name:
                    return part.tool_name
        raise ValueError(
            f"No UserPromptPart, TextPart, or ToolCallPart with "
            f"content found in message at index {index} (type={type(msg).__name__}). "
            "This may indicate a converter bug."
        )

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

    def assert_element_type(self, result: list, index: int, expected_role: str) -> None:
        from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart

        msg = result[index]
        if expected_role == "user":
            assert isinstance(msg, ModelRequest), (
                f"Expected ModelRequest, got {type(msg).__name__}"
            )
            assert len(msg.parts) >= 1, "ModelRequest has no parts"
            assert isinstance(msg.parts[0], UserPromptPart), (
                f"Expected UserPromptPart, got {type(msg.parts[0]).__name__}"
            )
        elif expected_role == "assistant":
            assert isinstance(msg, ModelResponse), (
                f"Expected ModelResponse, got {type(msg).__name__}"
            )
        else:
            raise ValueError(f"Unknown expected_role: {expected_role!r}")

    def assert_sender_metadata(
        self,
        result: list,
        index: int,
        sender_name: str,
        sender_type: str | None = None,
    ) -> None:
        raise NotImplementedError(
            "PydanticAIOutputAdapter.assert_sender_metadata() is not supported. "
            "PydanticAI messages do not include sender metadata. "
            "Ensure has_sender_metadata=False in the ConverterConfig."
        )


class StringOutputAdapter:
    """Adapter for ClaudeSDK converter output (newline-joined string).

    The ClaudeSDK converter joins messages with ``"\\n"``.  Each logical
    message starts with ``[sender]: ...`` (text) or ``{`` (JSON tool event).

    ``_split_messages`` splits the joined string into logical messages by
    checking each line for a valid message-start pattern:
    - Text message: ``[<name>]: <content>`` (validated by ``_SENDER_RE``)
    - Tool event: starts with ``{`` (JSON object from ``json.dumps``)

    Lines that do not match either pattern are treated as continuations of
    the previous message (e.g. multi-line text content).

    ``_split_messages`` is intentionally not cached — test inputs are small
    and caching across test boundaries would share state between independent
    tests.
    """

    # Matches the sender prefix at the start of a text message.
    # Requires at least ``[<anything>]: `` (bracket, colon, space).
    # This avoids false splits on ``[`` inside JSON values.
    _SENDER_RE = re.compile(r"^\[.*?\]: ")

    @classmethod
    def _is_message_start(cls, line: str) -> bool:
        """Return True if *line* looks like the start of a new message."""
        if cls._SENDER_RE.match(line):
            return True
        if line.startswith("{"):
            return True
        return False

    @classmethod
    def _split_messages(cls, result: str) -> list[str]:
        """Split the joined string into logical messages.

        Iterates line-by-line and starts a new message whenever a line
        matches a valid message-start pattern (sender prefix or JSON
        object start).  Other lines are appended to the current message.
        """
        if not result:
            return []
        lines = result.split("\n")
        messages: list[str] = []
        for line in lines:
            if not messages or cls._is_message_start(line):
                messages.append(line)
            else:
                messages[-1] += "\n" + line
        return messages

    def assert_result_type(self, result: str) -> None:
        assert isinstance(result, str), f"Expected str, got {type(result).__name__}"

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

    def assert_element_type(self, result: str, index: int, expected_role: str) -> None:
        assert isinstance(result, str), f"Expected str, got {type(result).__name__}"

    def assert_sender_metadata(
        self,
        result: str,
        index: int,
        sender_name: str,
        sender_type: str | None = None,
    ) -> None:
        raise NotImplementedError(
            "StringOutputAdapter.assert_sender_metadata() is not supported. "
            "ClaudeSDK returns a flat string without sender metadata. "
            "Ensure has_sender_metadata=False in the ConverterConfig."
        )


class SenderDictListAdapter(BaseDictListOutputAdapter):
    """Adapter for CrewAI/Parlant converter output (list[dict] with sender/sender_type).

    Unlike ``DictListOutputAdapter`` (Anthropic), which handles tool_use content
    blocks but has no sender metadata, this adapter supports ``sender`` and
    ``sender_type`` fields on each message dict.
    """

    def assert_sender_metadata(
        self,
        result: list[dict[str, Any]],
        index: int,
        sender_name: str,
        sender_type: str | None = None,
    ) -> None:
        msg = result[index]
        assert msg.get("sender") == sender_name, (
            f"Expected sender={sender_name!r}, got {msg.get('sender')!r}"
        )
        if sender_type is not None:
            assert msg.get("sender_type") == sender_type, (
                f"Expected sender_type={sender_type!r}, got {msg.get('sender_type')!r}"
            )
