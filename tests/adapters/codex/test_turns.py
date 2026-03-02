from __future__ import annotations

from thenvoi.adapters.codex.turns import extract_local_command


def test_extract_local_command_handles_mentions_and_arguments() -> None:
    assert extract_local_command("/model gpt-5.3-codex") == ("model", "gpt-5.3-codex")
    assert extract_local_command("@bot /reasoning high") == ("reasoning", "high")


def test_extract_local_command_ignores_unknown_or_deep_tokens() -> None:
    assert extract_local_command("/unknown test") is None
    assert extract_local_command("a b c d e /model gpt-5.3-codex") is None
