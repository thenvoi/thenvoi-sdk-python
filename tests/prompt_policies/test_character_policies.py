"""Tests for character prompt-policy assets."""

from __future__ import annotations

import pytest

from thenvoi.prompt_policies.characters import (
    generate_character_prompt,
    load_character_prompt_spec,
)


def test_generate_character_prompt_uses_default_name() -> None:
    prompt = generate_character_prompt(character="tom")
    assert "You are **Tom**" in prompt
    assert "Maximum 10 persuasion messages to Jerry." in prompt


def test_load_character_prompt_spec_rejects_unknown_character() -> None:
    with pytest.raises(ValueError, match="Unknown character policy key"):
        load_character_prompt_spec(character="spike")


def test_generate_character_prompt_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="Unknown character policy version"):
        generate_character_prompt(character="jerry", version="v999")
