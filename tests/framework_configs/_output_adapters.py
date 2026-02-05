"""Output type adapters for converter conformance tests.

These adapters provide a unified interface to extract data from
framework-specific converter output formats.
"""

from __future__ import annotations

from typing import Any, Protocol


class OutputTypeAdapter(Protocol):
    """Protocol for adapters that extract data from framework-specific output."""

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


class DictListAdapter:
    """Adapter for dict_list output (Anthropic, CrewAI, Parlant)."""

    def get_length(self, result: Any) -> int:
        return len(result) if result else 0

    def get_content(self, result: Any, index: int) -> str:
        if not result or index >= len(result):
            return ""
        content = result[index].get("content", "")
        return str(content) if isinstance(content, list) else content

    def get_role(self, result: Any, index: int) -> str:
        if not result or index >= len(result):
            return ""
        return result[index].get("role", "")

    def has_tool_call(
        self, result: Any, index: int, name: str | None, tool_id: str | None
    ) -> bool:
        if not result or index >= len(result):
            return False
        content = result[index].get("content", [])
        if not isinstance(content, list) or not content:
            return False
        block = content[0]
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
        if not result or index >= len(result):
            return False
        msg_content = result[index].get("content", [])
        if not isinstance(msg_content, list) or not msg_content:
            return False
        block = msg_content[0]
        if block.get("type") != "tool_result":
            return False
        if tool_id and block.get("tool_use_id") != tool_id:
            return False
        if content and block.get("content") != content:
            return False
        return True

    def get_tool_call_count(self, result: Any, index: int) -> int:
        if not result or index >= len(result):
            return 0
        content = result[index].get("content", [])
        if not isinstance(content, list):
            return 0
        return sum(1 for b in content if b.get("type") == "tool_use")

    def get_tool_result_count(self, result: Any, index: int) -> int:
        if not result or index >= len(result):
            return 0
        content = result[index].get("content", [])
        if not isinstance(content, list):
            return 0
        return sum(1 for b in content if b.get("type") == "tool_result")

    def has_is_error(self, result: Any, index: int, expected: bool) -> bool:
        if not result or index >= len(result):
            return False
        content = result[index].get("content", [])
        if not isinstance(content, list) or not content:
            return False
        block = content[0]
        return block.get("is_error") is True if expected else "is_error" not in block


class StringAdapter:
    """Adapter for string output (ClaudeSDK)."""

    def _lines(self, result: Any) -> list[str]:
        return [line for line in (result or "").split("\n") if line.strip()]

    def get_length(self, result: Any) -> int:
        return len(self._lines(result))

    def get_content(self, result: Any, index: int) -> str:
        lines = self._lines(result)
        return lines[index] if index < len(lines) else ""

    def get_role(self, result: Any, index: int) -> str:
        return ""

    def has_tool_call(
        self, result: Any, index: int, name: str | None, tool_id: str | None
    ) -> bool:
        lines = self._lines(result)
        if index >= len(lines):
            return False
        line = lines[index]
        if name and f'"{name}"' not in line:
            return False
        if tool_id and tool_id not in line:
            return False
        return True

    def has_tool_result(
        self, result: Any, index: int, tool_id: str | None, content: str | None
    ) -> bool:
        lines = self._lines(result)
        if index >= len(lines):
            return False
        line = lines[index]
        if tool_id and tool_id not in line:
            return False
        if content and content not in line:
            return False
        return True

    def get_tool_call_count(self, result: Any, index: int) -> int:
        return 0

    def get_tool_result_count(self, result: Any, index: int) -> int:
        return 0

    def has_is_error(self, result: Any, index: int, expected: bool) -> bool:
        return False


class LangChainAdapter:
    """Adapter for langchain_messages output."""

    def get_length(self, result: Any) -> int:
        return len(result) if result else 0

    def get_content(self, result: Any, index: int) -> str:
        if not result or index >= len(result):
            return ""
        return result[index].content

    def get_role(self, result: Any, index: int) -> str:
        if not result or index >= len(result):
            return ""
        name = type(result[index]).__name__
        if "Human" in name:
            return "user"
        if "AI" in name or "Tool" in name:
            return "assistant"
        return ""

    def has_tool_call(
        self, result: Any, index: int, name: str | None, tool_id: str | None
    ) -> bool:
        from langchain_core.messages import AIMessage

        if not result or index >= len(result):
            return False
        msg = result[index]
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            return False
        tc = msg.tool_calls[0]
        if name and tc.get("name") != name:
            return False
        if tool_id and tc.get("id") != tool_id:
            return False
        return True

    def has_tool_result(
        self, result: Any, index: int, tool_id: str | None, content: str | None
    ) -> bool:
        from langchain_core.messages import ToolMessage

        if not result or index >= len(result):
            return False
        msg = result[index]
        if not isinstance(msg, ToolMessage):
            return False
        if tool_id and msg.tool_call_id != tool_id:
            return False
        return True

    def get_tool_call_count(self, result: Any, index: int) -> int:
        from langchain_core.messages import AIMessage

        if not result or index >= len(result):
            return 0
        msg = result[index]
        return (
            len(msg.tool_calls) if isinstance(msg, AIMessage) and msg.tool_calls else 0
        )

    def get_tool_result_count(self, result: Any, index: int) -> int:
        return 0

    def has_is_error(self, result: Any, index: int, expected: bool) -> bool:
        return False


class PydanticAIAdapter:
    """Adapter for pydantic_ai_messages output."""

    def get_length(self, result: Any) -> int:
        return len(result) if result else 0

    def get_content(self, result: Any, index: int) -> str:
        if not result or index >= len(result):
            return ""
        msg = result[index]
        if not msg.parts:
            return ""
        part = msg.parts[0]
        return part.content if hasattr(part, "content") else str(part)

    def get_role(self, result: Any, index: int) -> str:
        if not result or index >= len(result):
            return ""
        name = type(result[index]).__name__
        if "Request" in name:
            return "user"
        if "Response" in name:
            return "assistant"
        return ""

    def has_tool_call(
        self, result: Any, index: int, name: str | None, tool_id: str | None
    ) -> bool:
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if not result or index >= len(result):
            return False
        msg = result[index]
        if not isinstance(msg, ModelResponse) or not msg.parts:
            return False
        part = msg.parts[0]
        if not isinstance(part, ToolCallPart):
            return False
        if name and part.tool_name != name:
            return False
        if tool_id and part.tool_call_id != tool_id:
            return False
        return True

    def has_tool_result(
        self, result: Any, index: int, tool_id: str | None, content: str | None
    ) -> bool:
        from pydantic_ai.messages import ModelRequest, ToolReturnPart, RetryPromptPart

        if not result or index >= len(result):
            return False
        msg = result[index]
        if not isinstance(msg, ModelRequest) or not msg.parts:
            return False
        part = msg.parts[0]
        if not isinstance(part, (ToolReturnPart, RetryPromptPart)):
            return False
        if tool_id and part.tool_call_id != tool_id:
            return False
        if content and part.content != content:
            return False
        return True

    def get_tool_call_count(self, result: Any, index: int) -> int:
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        if not result or index >= len(result):
            return 0
        msg = result[index]
        if not isinstance(msg, ModelResponse):
            return 0
        return sum(1 for p in msg.parts if isinstance(p, ToolCallPart))

    def get_tool_result_count(self, result: Any, index: int) -> int:
        from pydantic_ai.messages import ModelRequest, ToolReturnPart, RetryPromptPart

        if not result or index >= len(result):
            return 0
        msg = result[index]
        if not isinstance(msg, ModelRequest):
            return 0
        return sum(
            1 for p in msg.parts if isinstance(p, (ToolReturnPart, RetryPromptPart))
        )

    def has_is_error(self, result: Any, index: int, expected: bool) -> bool:
        from pydantic_ai.messages import ModelRequest, ToolReturnPart, RetryPromptPart

        if not result or index >= len(result):
            return False
        msg = result[index]
        if not isinstance(msg, ModelRequest) or not msg.parts:
            return False
        part = msg.parts[0]
        return (
            isinstance(part, RetryPromptPart)
            if expected
            else isinstance(part, ToolReturnPart)
        )
