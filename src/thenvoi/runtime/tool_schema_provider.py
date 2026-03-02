"""Tool schema conversion service for runtime tool definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from anthropic.types import ToolParam


class ToolSchemaProvider:
    """Convert tool model registry into provider-specific schema formats."""

    def __init__(
        self,
        *,
        tool_models: dict[str, type[BaseModel]],
        memory_tool_names: frozenset[str],
    ) -> None:
        self._tool_models = tool_models
        self._memory_tool_names = memory_tool_names

    @property
    def tool_models(self) -> dict[str, type[BaseModel]]:
        return self._tool_models

    def get_tool_schemas(
        self,
        format: str,
        *,
        include_memory: bool = False,
    ) -> list[dict[str, Any]] | list["ToolParam"]:
        """Build provider schemas from the canonical tool model registry."""
        if format not in ("openai", "anthropic"):
            raise ValueError(
                f"Invalid format: {format}. Must be 'openai' or 'anthropic'"
            )

        tools: list[Any] = []
        for name, model in self._tool_models.items():
            if not include_memory and name in self._memory_tool_names:
                continue

            schema = model.model_json_schema()
            schema.pop("title", None)

            if format == "openai":
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": model.__doc__ or "",
                            "parameters": schema,
                        },
                    }
                )
            elif format == "anthropic":
                tools.append(
                    {
                        "name": name,
                        "description": model.__doc__ or "",
                        "input_schema": schema,
                    }
                )
        return tools

    def get_anthropic_tool_schemas(
        self,
        *,
        include_memory: bool = False,
    ) -> list["ToolParam"]:
        """Get tool schemas in Anthropic format (strongly typed)."""
        return cast(
            list["ToolParam"],
            self.get_tool_schemas("anthropic", include_memory=include_memory),
        )

    def get_openai_tool_schemas(
        self,
        *,
        include_memory: bool = False,
    ) -> list[dict[str, Any]]:
        """Get tool schemas in OpenAI format (strongly typed)."""
        return cast(
            list[dict[str, Any]],
            self.get_tool_schemas("openai", include_memory=include_memory),
        )
