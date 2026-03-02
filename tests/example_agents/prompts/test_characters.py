"""Tests for examples/prompts/characters.py."""

from __future__ import annotations

import pytest

from thenvoi.example_support.prompts.characters import generate_jerry_prompt, generate_tom_prompt

pytestmark = pytest.mark.contract_gate


def test_generate_tom_prompt_uses_custom_name() -> None:
    prompt = generate_tom_prompt("Thomas")

    assert "You are **Thomas**" in prompt
    assert "Use `thenvoi_lookup_peers(participant_type=\"Agent\")`" in prompt
    assert "Attempts 1-3: friendly invitations." in prompt
    assert "Maximum 10 persuasion messages to Jerry." in prompt


def test_generate_jerry_prompt_uses_custom_name() -> None:
    prompt = generate_jerry_prompt("Gerald")

    assert "You are **Gerald**" in prompt
    assert "You enjoy teasing Tom from safety." in prompt
    assert "If you commit to leaving your hole and Tom pounces" in prompt
    assert "Keep internal analysis in thought events" in prompt


def test_tom_prompt_mentions_jerry_and_not_tom_handle() -> None:
    prompt = generate_tom_prompt()

    assert "Mention `@Jerry` in each response to him." in prompt
    assert "Mention `@Tom` in each response to him." not in prompt


def test_jerry_prompt_mentions_tom_and_not_jerry_handle() -> None:
    prompt = generate_jerry_prompt()

    assert "Mention `@Tom` in each response to him." in prompt
    assert "Mention `@Jerry` in each response to him." not in prompt
