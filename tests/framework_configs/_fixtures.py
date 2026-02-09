"""Shared tool-event payloads for conformance and framework-specific tests.

Payloads include BOTH top-level keys and nested data.* paths because
different frameworks read from different locations:
  - Anthropic, ClaudeSDK, PydanticAI: read top-level args/output
  - LangChain: reads data.input / data.output

Always pair tool_call + tool_result so frameworks that require
pairing (e.g. LangChain) produce valid output.
"""

from __future__ import annotations

TOOL_CALL_SEARCH = {
    "role": "assistant",
    "content": '{"name": "search", "args": {"query": "test"}, "data": {"input": {"query": "test"}}, "tool_call_id": "tc_1"}',
    "message_type": "tool_call",
}
TOOL_RESULT_SEARCH = {
    "role": "assistant",
    "content": '{"name": "search", "output": "result data", "data": {"output": "result data"}, "tool_call_id": "tc_1"}',
    "message_type": "tool_result",
}
TOOL_CALL_LOOKUP = {
    "role": "assistant",
    "content": '{"name": "lookup", "args": {"id": "42"}, "data": {"input": {"id": "42"}}, "tool_call_id": "tc_2"}',
    "message_type": "tool_call",
}
TOOL_RESULT_LOOKUP = {
    "role": "assistant",
    "content": '{"name": "lookup", "output": "found item 42", "data": {"output": "found item 42"}, "tool_call_id": "tc_2"}',
    "message_type": "tool_result",
}
TOOL_CALL_SEARCH_EMPTY = {
    "role": "assistant",
    "content": '{"name": "search", "args": {}, "data": {"input": {}}, "tool_call_id": "tc_1"}',
    "message_type": "tool_call",
}
TOOL_RESULT_SEARCH_FOUND = {
    "role": "assistant",
    "content": '{"name": "search", "output": "found", "data": {"output": "found"}, "tool_call_id": "tc_1"}',
    "message_type": "tool_result",
}
