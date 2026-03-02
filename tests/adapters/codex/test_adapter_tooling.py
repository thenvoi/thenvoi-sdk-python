from __future__ import annotations

import json

from thenvoi.adapters.codex.adapter_tooling import extract_thought_text, extract_tool_item


def test_extract_tool_item_command_execution_formats_output() -> None:
    name, args, output = extract_tool_item(
        "commandExecution",
        {
            "command": "ls -la",
            "cwd": "/tmp",
            "aggregated_output": "file.txt",
            "exitCode": 0,
        },
    )
    assert name == "exec"
    assert args == {"command": "ls -la", "cwd": "/tmp"}
    assert "file.txt" in output
    assert "exit_code=0" in output


def test_extract_tool_item_mcp_tool_call_serializes_result() -> None:
    name, args, output = extract_tool_item(
        "mcpToolCall",
        {
            "server": "linear",
            "tool": "list_issues",
            "arguments": {"limit": 3},
            "result": {"items": [1, 2, 3]},
        },
    )
    assert name == "mcp:linear/list_issues"
    assert args == {"limit": 3}
    assert json.loads(output) == {"items": [1, 2, 3]}


def test_extract_thought_text_handles_reasoning_and_review_modes() -> None:
    assert extract_thought_text("reasoning", {"summary": ["a", "b"]}) == "a\nb"
    assert extract_thought_text("enteredReviewMode", {}) == "Review mode: enteredReviewMode"

