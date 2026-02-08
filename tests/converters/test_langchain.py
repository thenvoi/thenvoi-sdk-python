"""LangChain converter-specific tests.

Tests for LangChain-specific behavior that isn't covered by conformance tests:
- _extract_tool_call_id helper method
- AIMessage/ToolMessage structure
- Tool call/result pairing logic
- run_id fallback when tool_call_id not found
"""

from __future__ import annotations

import json

from thenvoi.converters.langchain import LangChainHistoryConverter


class TestExtractToolCallId:
    """Tests for _extract_tool_call_id helper method."""

    def test_extracts_tool_call_id_from_output(self):
        """Should extract tool_call_id from output string."""
        output = "Result data tool_call_id='toolu_abc123'"
        result = LangChainHistoryConverter._extract_tool_call_id(output)
        assert result == "toolu_abc123"

    def test_returns_none_when_not_found(self):
        """Should return None when tool_call_id pattern not found."""
        output = "Just some regular output without id"
        result = LangChainHistoryConverter._extract_tool_call_id(output)
        assert result is None

    def test_handles_complex_output_string(self):
        """Should extract from complex output with multiple values."""
        output = "weather={'temp': 72} tool_call_id='call_xyz789' status=success"
        result = LangChainHistoryConverter._extract_tool_call_id(output)
        assert result == "call_xyz789"

    def test_extracts_first_match(self):
        """Should extract the first tool_call_id if multiple present."""
        output = "tool_call_id='first' other tool_call_id='second'"
        result = LangChainHistoryConverter._extract_tool_call_id(output)
        assert result == "first"


class TestAIMessageStructure:
    """Tests for AIMessage structure with tool_calls."""

    def test_creates_ai_message_with_tool_calls_array(self):
        """Tool call should create AIMessage with tool_calls array."""
        from langchain_core.messages import AIMessage

        converter = LangChainHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "search", "data": {"input": {"query": "test"}}}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "search",
                        "data": {"output": "results tool_call_id='tc_123'"},
                        "run_id": "tc_123",
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        # Find the AIMessage
        ai_messages = [m for m in result if isinstance(m, AIMessage)]
        assert len(ai_messages) == 1

        ai_msg = ai_messages[0]
        assert ai_msg.tool_calls is not None
        assert len(ai_msg.tool_calls) == 1

        tool_call = ai_msg.tool_calls[0]
        assert tool_call["name"] == "search"
        assert tool_call["args"] == {"query": "test"}
        assert "id" in tool_call


class TestToolMessageStructure:
    """Tests for ToolMessage structure."""

    def test_creates_tool_message_with_tool_call_id(self):
        """Tool result should create ToolMessage with tool_call_id."""
        from langchain_core.messages import ToolMessage

        converter = LangChainHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "search", "data": {"input": {"query": "test"}}}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "search",
                        "data": {"output": "found it tool_call_id='tc_456'"},
                        "run_id": "tc_456",
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        # Find the ToolMessage
        tool_messages = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1

        tool_msg = tool_messages[0]
        assert tool_msg.tool_call_id == "tc_456"
        assert "found it" in tool_msg.content


class TestToolCallPairing:
    """Tests for tool call and result pairing."""

    def test_pairs_tool_call_with_tool_result(self):
        """Should pair tool_call with matching tool_result by name."""
        from langchain_core.messages import AIMessage, ToolMessage

        converter = LangChainHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "weather", "data": {"input": {"city": "NYC"}}}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "weather",
                        "data": {"output": "sunny tool_call_id='w123'"},
                        "run_id": "w123",
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        assert len(result) == 2
        assert isinstance(result[0], AIMessage)
        assert isinstance(result[1], ToolMessage)

        # AIMessage should have the tool call
        assert result[0].tool_calls[0]["name"] == "weather"

    def test_uses_run_id_when_tool_call_id_not_in_output(self):
        """Should use run_id as fallback when tool_call_id not in output."""
        from langchain_core.messages import ToolMessage

        converter = LangChainHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "calc", "data": {"input": {"expr": "2+2"}}}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "calc",
                        "data": {"output": "4"},  # No tool_call_id in output
                        "run_id": "run_fallback_id",
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        tool_msg = [m for m in result if isinstance(m, ToolMessage)][0]
        assert tool_msg.tool_call_id == "run_fallback_id"

    def test_handles_tool_result_without_matching_call(self):
        """Should create ToolMessage even without matching tool_call."""
        from langchain_core.messages import ToolMessage

        converter = LangChainHistoryConverter()
        raw = [
            # No matching tool_call
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "orphan_tool",
                        "data": {"output": "result tool_call_id='orphan_1'"},
                        "run_id": "orphan_1",
                    }
                ),
                "message_type": "tool_result",
            }
        ]

        result = converter.convert(raw)

        # Should still create ToolMessage (no AIMessage since no matching call)
        assert len(result) == 1
        assert isinstance(result[0], ToolMessage)


class TestMultipleToolCalls:
    """Tests for handling multiple tool calls."""

    def test_handles_multiple_sequential_tool_calls(self):
        """Should handle multiple sequential tool call/result pairs."""
        from langchain_core.messages import AIMessage, ToolMessage

        converter = LangChainHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps({"name": "tool1", "data": {"input": {}}}),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "tool1",
                        "data": {"output": "r1 tool_call_id='t1'"},
                        "run_id": "t1",
                    }
                ),
                "message_type": "tool_result",
            },
            {
                "role": "assistant",
                "content": json.dumps({"name": "tool2", "data": {"input": {}}}),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "tool2",
                        "data": {"output": "r2 tool_call_id='t2'"},
                        "run_id": "t2",
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        ai_messages = [m for m in result if isinstance(m, AIMessage)]
        tool_messages = [m for m in result if isinstance(m, ToolMessage)]

        assert len(ai_messages) == 2
        assert len(tool_messages) == 2
