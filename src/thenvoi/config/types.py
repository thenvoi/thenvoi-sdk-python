"""Typed configuration models for config-driven agent creation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from thenvoi.core.exceptions import ThenvoiConfigError


@dataclass
class AgentConfig:
    """Typed config loaded from YAML for config-driven agent creation."""

    agent_id: UUID
    api_key: str
    adapter: dict[str, Any] | None = None
    capabilities: list[str] = field(default_factory=list)
    include_categories: list[str] | None = None
    include_tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    emit: list[str] = field(default_factory=list)
    prompt: str | None = None
    prompt_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, UUID):
            try:
                self.agent_id = UUID(str(self.agent_id))
            except (TypeError, ValueError) as exc:
                raise ThenvoiConfigError(
                    "Invalid agent_id value. Use a valid UUID string in your config."
                ) from exc

        if self.prompt_path is not None and not isinstance(self.prompt_path, Path):
            self.prompt_path = Path(self.prompt_path)

    def __iter__(self) -> Iterator[str]:
        warnings.warn(
            "Tuple unpacking from load_agent_config() is deprecated. Use the returned AgentConfig object instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield str(self.agent_id)
        yield self.api_key
