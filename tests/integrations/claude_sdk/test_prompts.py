"""Contract tests for Claude SDK prompt builder."""

from __future__ import annotations

from thenvoi.integrations.claude_sdk.prompts import generate_claude_sdk_agent_prompt


def test_generate_claude_sdk_agent_prompt_builds_expected_preset() -> None:
    """Prompt builder should return Claude Code preset with appended instructions."""
    preset = generate_claude_sdk_agent_prompt(
        agent_name="Planner",
        agent_description="a planning assistant",
    )

    assert preset["type"] == "preset"
    assert preset["preset"] == "claude_code"
    assert "You are **Planner**, a planning assistant" in preset["append"]


def test_generate_claude_sdk_agent_prompt_includes_required_tool_contracts() -> None:
    """Prompt should include critical Thenvoi tool constraints and message rules."""
    preset = generate_claude_sdk_agent_prompt("Planner")
    prompt = preset["append"]

    assert "mcp__thenvoi__thenvoi_send_message" in prompt
    assert "Extract the `room_id`" in prompt
    assert "Always use mcp__thenvoi__thenvoi_send_message" in prompt
    assert "mcp__thenvoi__thenvoi_lookup_peers" in prompt


def test_generate_claude_sdk_agent_prompt_appends_custom_section_when_provided() -> None:
    """Custom instructions should be appended under a dedicated heading."""
    preset = generate_claude_sdk_agent_prompt(
        "Planner",
        custom_section="Respond in bullet points.",
    )

    assert "## Custom Instructions" in preset["append"]
    assert "Respond in bullet points." in preset["append"]


def test_generate_claude_sdk_agent_prompt_omits_custom_heading_when_empty() -> None:
    """Custom instruction heading should be absent when no custom section is passed."""
    preset = generate_claude_sdk_agent_prompt("Planner", custom_section=None)

    assert "## Custom Instructions" not in preset["append"]
