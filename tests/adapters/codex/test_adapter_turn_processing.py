from __future__ import annotations

from thenvoi.adapters.codex.adapter_turn_processing import extract_turn_error


def test_extract_turn_error_returns_message_when_present() -> None:
    assert extract_turn_error({"error": {"message": "rate limited"}}) == "rate limited"
    assert extract_turn_error({"error": {"detail": "ignored"}}) == ""
    assert extract_turn_error({"error": "oops"}) == ""

