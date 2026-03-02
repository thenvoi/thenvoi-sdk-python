"""Tests for contact prompt-policy assets."""

from __future__ import annotations

import pytest

from thenvoi.prompt_policies.contacts import load_hub_room_system_prompt


def test_load_hub_room_system_prompt_default_version() -> None:
    prompt = load_hub_room_system_prompt()
    assert "CONTACTS HUB" in prompt
    assert "thenvoi_respond_contact_request" in prompt


def test_load_hub_room_system_prompt_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="Unknown contacts policy version"):
        load_hub_room_system_prompt("v999")
