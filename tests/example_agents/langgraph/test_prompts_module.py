"""Tests for examples/scenarios/prompts/langgraph.py."""

from __future__ import annotations

from examples.scenarios.prompts.langgraph import generate_langgraph_agent_prompt


def test_generate_langgraph_agent_prompt_includes_name_and_core_contract() -> None:
    prompt = generate_langgraph_agent_prompt("Navigator")

    assert "You are Navigator" in prompt
    assert "Mandatory Response Contract" in prompt
    assert "thenvoi_send_message" in prompt
    assert "Mention Rules" in prompt
    assert "Collaboration Rules" in prompt
