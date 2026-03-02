"""Turn-level parser helpers for Codex adapter."""

from __future__ import annotations

_COMMANDS: set[str] = {
    "help",
    "status",
    "model",
    "models",
    "reasoning",
    "approvals",
    "approve",
    "decline",
}


def extract_local_command(content: str) -> tuple[str, str] | None:
    """Parse slash-command and trailing args from a message body."""
    tokens = content.strip().split()
    if not tokens:
        return None

    # Search near start to allow leading @mentions while avoiding accidental
    # slash-words deep in prose.
    search_limit = min(len(tokens), 5)
    for idx in range(search_limit):
        token = tokens[idx]
        if not token.startswith("/") or len(token) == 1:
            continue
        command = token[1:].lower()
        if command not in _COMMANDS:
            continue
        args = " ".join(tokens[idx + 1 :]).strip()
        return command, args
    return None


__all__ = ["extract_local_command"]

