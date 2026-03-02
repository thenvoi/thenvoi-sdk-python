"""Character prompts for Tom and Jerry example agents."""

from __future__ import annotations

from thenvoi.prompt_policies.characters import generate_character_prompt


def generate_tom_prompt(agent_name: str = "Tom") -> str:
    """Generate Tom's roleplay prompt."""
    return generate_character_prompt(character="tom", agent_name=agent_name)


def generate_jerry_prompt(agent_name: str = "Jerry") -> str:
    """Generate Jerry's roleplay prompt."""
    return generate_character_prompt(character="jerry", agent_name=agent_name)


__all__ = [
    "generate_jerry_prompt",
    "generate_tom_prompt",
]
