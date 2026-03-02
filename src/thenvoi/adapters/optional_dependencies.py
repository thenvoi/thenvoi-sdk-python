"""Shared optional dependency policy for adapter modules."""

from __future__ import annotations

from collections.abc import Iterable


def ensure_optional_dependency(
    import_error: ImportError | None,
    *,
    package: str,
    integration: str,
    install_commands: Iterable[str],
) -> None:
    """Raise a consistent error when an optional adapter dependency is missing."""
    if import_error is None:
        return

    commands = tuple(dict.fromkeys(install_commands))
    install_lines = "\n".join(f"Install with: {command}" for command in commands)
    message = (
        f"{package} is required for {integration} integrations.\n"
        f"{install_lines}"
    )
    raise ImportError(message) from import_error


__all__ = ["ensure_optional_dependency"]
