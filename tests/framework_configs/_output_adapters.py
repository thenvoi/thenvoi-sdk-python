"""Output type adapters for converter conformance tests."""

from __future__ import annotations

from typing import Any, Protocol


class OutputTypeAdapter(Protocol):
    """Protocol for extracting data from framework-specific output."""

    def get_length(self, result: Any) -> int: ...
    def get_content(self, result: Any, index: int) -> str: ...
    def get_role(self, result: Any, index: int) -> str: ...
    def has_tool_call(
        self, result: Any, index: int, name: str | None, tool_id: str | None
    ) -> bool: ...
    def has_tool_result(
        self, result: Any, index: int, tool_id: str | None, content: str | None
    ) -> bool: ...
    def get_tool_call_count(self, result: Any, index: int) -> int: ...
    def get_tool_result_count(self, result: Any, index: int) -> int: ...
    def has_is_error(self, result: Any, index: int, expected: bool) -> bool: ...


class BaseAdapter:
    """Base adapter with common functionality.

    Subclasses must implement get_content() and get_role().
    """

    def _get(self, result: Any, index: int) -> Any:
        """Safely get item at index, returns None if out of bounds."""
        if not result or index < 0 or index >= len(result):
            return None
        return result[index]

    def get_length(self, result: Any) -> int:
        return len(result) if result else 0

    def get_content(self, result: Any, index: int) -> str:
        raise NotImplementedError("Subclasses must implement get_content")

    def get_role(self, result: Any, index: int) -> str:
        raise NotImplementedError("Subclasses must implement get_role")

    # Default implementations for tool methods (most adapters don't support these)
    def has_tool_call(
        self, result: Any, index: int, name: str | None, tool_id: str | None
    ) -> bool:
        return False

    def has_tool_result(
        self, result: Any, index: int, tool_id: str | None, content: str | None
    ) -> bool:
        return False

    def get_tool_call_count(self, result: Any, index: int) -> int:
        return 0

    def get_tool_result_count(self, result: Any, index: int) -> int:
        return 0

    def has_is_error(self, result: Any, index: int, expected: bool) -> bool:
        return False


class DictListAdapter(BaseAdapter):
    """Adapter for dict_list output (Anthropic, CrewAI, Parlant)."""

    def get_content(self, result: Any, index: int) -> str:
        if (msg := self._get(result, index)) is None:
            return ""
        content = msg.get("content", "")
        return str(content) if isinstance(content, list) else content

    def get_role(self, result: Any, index: int) -> str:
        if (msg := self._get(result, index)) is None:
            return ""
        return msg.get("role", "")

    def _get_tool_block(self, result: Any, index: int) -> dict | None:
        """Get first tool block from message content."""
        if (msg := self._get(result, index)) is None:
            return None
        content = msg.get("content", [])
        if isinstance(content, list) and content:
            return content[0]
        return None

    def has_tool_call(
        self, result: Any, index: int, name: str | None, tool_id: str | None
    ) -> bool:
        if (block := self._get_tool_block(result, index)) is None:
            return False
        if block.get("type") != "tool_use":
            return False
        if name and block.get("name") != name:
            return False
        if tool_id and block.get("id") != tool_id:
            return False
        return True

    def has_tool_result(
        self, result: Any, index: int, tool_id: str | None, content: str | None
    ) -> bool:
        if (block := self._get_tool_block(result, index)) is None:
            return False
        if block.get("type") != "tool_result":
            return False
        if tool_id and block.get("tool_use_id") != tool_id:
            return False
        if content and block.get("content") != content:
            return False
        return True

    def _count_blocks(self, result: Any, index: int, block_type: str) -> int:
        if (msg := self._get(result, index)) is None:
            return 0
        content = msg.get("content", [])
        if not isinstance(content, list):
            return 0
        return sum(1 for b in content if b.get("type") == block_type)

    def get_tool_call_count(self, result: Any, index: int) -> int:
        return self._count_blocks(result, index, "tool_use")

    def get_tool_result_count(self, result: Any, index: int) -> int:
        return self._count_blocks(result, index, "tool_result")

    def has_is_error(self, result: Any, index: int, expected: bool) -> bool:
        if (block := self._get_tool_block(result, index)) is None:
            return False
        return block.get("is_error") is True if expected else "is_error" not in block


class StringAdapter(BaseAdapter):
    """Adapter for string output (ClaudeSDK)."""

    def _lines(self, result: Any) -> list[str]:
        return [line for line in (result or "").split("\n") if line.strip()]

    def get_length(self, result: Any) -> int:
        return len(self._lines(result))

    def get_content(self, result: Any, index: int) -> str:
        lines = self._lines(result)
        return lines[index] if 0 <= index < len(lines) else ""

    def get_role(self, result: Any, index: int) -> str:
        return ""

    def has_tool_call(
        self, result: Any, index: int, name: str | None, tool_id: str | None
    ) -> bool:
        lines = self._lines(result)
        if index < 0 or index >= len(lines):
            return False
        line = lines[index]
        # Use "name": to match JSON key exactly and avoid partial matches
        return (not name or f'"name": "{name}"' in line) and (
            not tool_id or f'"tool_call_id": "{tool_id}"' in line
        )

    def has_tool_result(
        self, result: Any, index: int, tool_id: str | None, content: str | None
    ) -> bool:
        lines = self._lines(result)
        if index < 0 or index >= len(lines):
            return False
        line = lines[index]
        return (not tool_id or f'"tool_call_id": "{tool_id}"' in line) and (
            not content or content in line
        )


class LangChainAdapter(BaseAdapter):
    """Adapter for langchain_messages output."""

    def get_content(self, result: Any, index: int) -> str:
        if (msg := self._get(result, index)) is None:
            return ""
        return msg.content

    def get_role(self, result: Any, index: int) -> str:
        if (msg := self._get(result, index)) is None:
            return ""
        name = type(msg).__name__
        if "Human" in name:
            return "user"
        if "AI" in name or "Tool" in name:
            return "assistant"
        return ""

    def has_tool_call(
        self, result: Any, index: int, name: str | None, tool_id: str | None
    ) -> bool:
        from langchain_core.messages import AIMessage

        if (msg := self._get(result, index)) is None:
            return False
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            return False
        tc = msg.tool_calls[0]
        return (not name or tc.get("name") == name) and (
            not tool_id or tc.get("id") == tool_id
        )

    def has_tool_result(
        self, result: Any, index: int, tool_id: str | None, content: str | None
    ) -> bool:
        from langchain_core.messages import ToolMessage

        if (msg := self._get(result, index)) is None:
            return False
        if not isinstance(msg, ToolMessage):
            return False
        if tool_id and msg.tool_call_id != tool_id:
            return False
        if content and msg.content != content:
            return False
        return True

    def get_tool_call_count(self, result: Any, index: int) -> int:
        from langchain_core.messages import AIMessage

        if (msg := self._get(result, index)) is None:
            return 0
        return (
            len(msg.tool_calls) if isinstance(msg, AIMessage) and msg.tool_calls else 0
        )


class PydanticAIAdapter(BaseAdapter):
    """Adapter for pydantic_ai_messages output."""

    def get_content(self, result: Any, index: int) -> str:
        if (msg := self._get(result, index)) is None or not msg.parts:
            return ""
        part = msg.parts[0]
        return part.content if hasattr(part, "content") else str(part)

    def get_role(self, result: Any, index: int) -> str:
        if (msg := self._get(result, index)) is None:
            return ""
        name = type(msg).__name__
        return (
            "user" if "Request" in name else "assistant" if "Response" in name else ""
        )

    def has_tool_call(
        self, result: Any, index: int, name: str | None, tool_id: str | None
    ) -> bool:
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if (msg := self._get(result, index)) is None:
            return False
        if not isinstance(msg, ModelResponse) or not msg.parts:
            return False
        part = msg.parts[0]
        if not isinstance(part, ToolCallPart):
            return False
        return (not name or part.tool_name == name) and (
            not tool_id or part.tool_call_id == tool_id
        )

    def has_tool_result(
        self, result: Any, index: int, tool_id: str | None, content: str | None
    ) -> bool:
        from pydantic_ai.messages import ModelRequest, ToolReturnPart, RetryPromptPart

        if (msg := self._get(result, index)) is None:
            return False
        if not isinstance(msg, ModelRequest) or not msg.parts:
            return False
        part = msg.parts[0]
        if not isinstance(part, (ToolReturnPart, RetryPromptPart)):
            return False
        return (not tool_id or part.tool_call_id == tool_id) and (
            not content or part.content == content
        )

    def get_tool_call_count(self, result: Any, index: int) -> int:
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if (msg := self._get(result, index)) is None or not isinstance(
            msg, ModelResponse
        ):
            return 0
        return sum(1 for p in msg.parts if isinstance(p, ToolCallPart))

    def get_tool_result_count(self, result: Any, index: int) -> int:
        from pydantic_ai.messages import ModelRequest, ToolReturnPart, RetryPromptPart

        if (msg := self._get(result, index)) is None or not isinstance(
            msg, ModelRequest
        ):
            return 0
        return sum(
            1 for p in msg.parts if isinstance(p, (ToolReturnPart, RetryPromptPart))
        )

    def has_is_error(self, result: Any, index: int, expected: bool) -> bool:
        from pydantic_ai.messages import ModelRequest, ToolReturnPart, RetryPromptPart

        if (msg := self._get(result, index)) is None:
            return False
        if not isinstance(msg, ModelRequest) or not msg.parts:
            return False
        part = msg.parts[0]
        return (
            isinstance(part, RetryPromptPart)
            if expected
            else isinstance(part, ToolReturnPart)
        )
