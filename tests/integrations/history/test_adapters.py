"""Tests for framework adapters."""

from thenvoi.integrations.history import (
    NormalizedToolExchange,
    NormalizedUserText,
)
from thenvoi.integrations.history.adapters.anthropic import to_anthropic_messages
from thenvoi.integrations.history.adapters.langgraph import to_langgraph_messages


class TestAnthropicAdapter:
    """Tests for Anthropic message conversion."""

    def test_converts_user_text(self):
        """NormalizedUserText becomes user message with formatted content."""
        normalized = [NormalizedUserText(sender_name="Alice", content="Hello!")]

        result = to_anthropic_messages(normalized)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[Alice]: Hello!"

    def test_converts_tool_exchange(self):
        """NormalizedToolExchange becomes assistant + user message pair."""
        normalized = [
            NormalizedToolExchange(
                tool_name="get_weather",
                tool_id="run-123",
                input_args={"city": "NYC"},
                output="Sunny",
                is_error=False,
            )
        ]

        result = to_anthropic_messages(normalized)

        assert len(result) == 2
        # Assistant message with tool_use
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "tool_use"
        assert result[0]["content"][0]["id"] == "run-123"
        assert result[0]["content"][0]["name"] == "get_weather"
        assert result[0]["content"][0]["input"] == {"city": "NYC"}
        # User message with tool_result
        assert result[1]["role"] == "user"
        assert result[1]["content"][0]["type"] == "tool_result"
        assert result[1]["content"][0]["tool_use_id"] == "run-123"
        assert result[1]["content"][0]["content"] == "Sunny"
        assert result[1]["content"][0]["is_error"] is False

    def test_preserves_error_flag(self):
        """Tool exchange with is_error=True is preserved."""
        normalized = [
            NormalizedToolExchange(
                tool_name="failing",
                tool_id="run-err",
                input_args={},
                output="Error!",
                is_error=True,
            )
        ]

        result = to_anthropic_messages(normalized)

        assert result[1]["content"][0]["is_error"] is True

    def test_batches_consecutive_user_messages(self):
        """Consecutive user messages are batched into one."""
        normalized = [
            NormalizedUserText(sender_name="Alice", content="Hello"),
            NormalizedUserText(sender_name="Bob", content="Hi there"),
        ]

        result = to_anthropic_messages(normalized)

        # Should be batched into single user message
        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Content should be a list with two text blocks
        assert len(result[0]["content"]) == 2

    def test_batches_consecutive_tool_results(self):
        """Consecutive tool exchanges batch their user messages (tool_results)."""
        normalized = [
            NormalizedToolExchange(
                tool_name="tool_a",
                tool_id="run-a",
                input_args={},
                output="A",
            ),
            NormalizedToolExchange(
                tool_name="tool_b",
                tool_id="run-b",
                input_args={},
                output="B",
            ),
        ]

        result = to_anthropic_messages(normalized)

        # assistant (tool_use a) + user (tool_result a) + assistant (tool_use b) + user (tool_result b)
        # After batching: should alternate properly
        # Actually each exchange is assistant->user, so we get:
        # [assistant(a), user(a), assistant(b), user(b)] -> already alternating, no batching needed
        assert len(result) == 4
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"

    def test_handles_empty_input(self):
        """Empty normalized list returns empty result."""
        result = to_anthropic_messages([])
        assert result == []


class TestLangGraphAdapter:
    """Tests for LangGraph message conversion."""

    def test_converts_user_text(self):
        """NormalizedUserText becomes HumanMessage."""
        from langchain_core.messages import HumanMessage

        normalized = [NormalizedUserText(sender_name="Alice", content="Hello!")]

        result = to_langgraph_messages(normalized)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "[Alice]: Hello!"

    def test_converts_tool_exchange(self):
        """NormalizedToolExchange becomes AIMessage + ToolMessage pair."""
        from langchain_core.messages import AIMessage, ToolMessage

        normalized = [
            NormalizedToolExchange(
                tool_name="get_weather",
                tool_id="run-123",
                input_args={"city": "NYC"},
                output="Sunny",
            )
        ]

        result = to_langgraph_messages(normalized)

        assert len(result) == 2
        # AIMessage with tool_calls
        assert isinstance(result[0], AIMessage)
        assert result[0].tool_calls[0]["id"] == "run-123"
        assert result[0].tool_calls[0]["name"] == "get_weather"
        assert result[0].tool_calls[0]["args"] == {"city": "NYC"}
        # ToolMessage with result
        assert isinstance(result[1], ToolMessage)
        assert result[1].tool_call_id == "run-123"
        assert result[1].content == "Sunny"

    def test_handles_multiple_exchanges(self):
        """Multiple tool exchanges create proper message sequence."""
        from langchain_core.messages import AIMessage, ToolMessage

        normalized = [
            NormalizedToolExchange(
                tool_name="tool_a", tool_id="run-a", input_args={}, output="A"
            ),
            NormalizedToolExchange(
                tool_name="tool_b", tool_id="run-b", input_args={}, output="B"
            ),
        ]

        result = to_langgraph_messages(normalized)

        assert len(result) == 4
        assert isinstance(result[0], AIMessage)
        assert isinstance(result[1], ToolMessage)
        assert isinstance(result[2], AIMessage)
        assert isinstance(result[3], ToolMessage)

    def test_handles_empty_input(self):
        """Empty normalized list returns empty result."""
        result = to_langgraph_messages([])
        assert result == []
