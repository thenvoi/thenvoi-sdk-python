"""Agent mapping parsing for bridge mention routing."""

from __future__ import annotations


def parse_agent_mapping(mapping_str: str) -> dict[str, str]:
    """Parse AGENT_MAPPING in the format 'alice:handler_a,bob:handler_b'."""
    if not mapping_str or not mapping_str.strip():
        raise ValueError("AGENT_MAPPING cannot be empty")

    result: dict[str, str] = {}
    for entry in mapping_str.split(","):
        candidate = entry.strip()
        if not candidate:
            continue

        parts = candidate.split(":")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(
                f"Invalid AGENT_MAPPING entry: '{candidate}'. "
                "Expected format: 'agent_name:handler_name'"
            )
        username = parts[0].strip()
        if username in result:
            raise ValueError(
                f"Duplicate username '{username}' in AGENT_MAPPING. "
                "Each username must map to exactly one handler."
            )
        result[username] = parts[1].strip()

    if not result:
        raise ValueError("AGENT_MAPPING produced no valid entries")

    return result


__all__ = ["parse_agent_mapping"]
