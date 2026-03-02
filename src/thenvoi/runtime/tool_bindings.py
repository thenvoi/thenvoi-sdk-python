"""Shared tool-binding registry for framework adapters."""

from __future__ import annotations

import inspect
import types
from collections.abc import Mapping
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from thenvoi.runtime.tool_bridge import get_platform_tool_order
from thenvoi.runtime.tool_definitions import TOOL_MODELS


class ToolBindingRegistry:
    """Adapter-agnostic access to tool models and derived bindings."""

    def __init__(self, tool_models: Mapping[str, type[BaseModel]]) -> None:
        self._tool_models: dict[str, type[BaseModel]] = dict(tool_models)

    def tool_names(self, *, include_memory_tools: bool) -> list[str]:
        """Return deterministic platform tool names for an adapter."""
        return get_platform_tool_order(include_memory_tools=include_memory_tools)

    def iter_models(
        self, *, include_memory_tools: bool
    ) -> list[tuple[str, type[BaseModel]]]:
        """Return tool model bindings in deterministic order."""
        return [
            (tool_name, self._tool_models[tool_name])
            for tool_name in self.tool_names(include_memory_tools=include_memory_tools)
        ]

    def crewai_schemas(
        self,
        *,
        include_memory_tools: bool,
        overrides: Mapping[str, type[BaseModel]] | None = None,
    ) -> dict[str, type[BaseModel]]:
        """Return CrewAI args_schema models keyed by tool name."""
        override_models = overrides or {}
        return {
            tool_name: override_models.get(tool_name, model)
            for tool_name, model in self.iter_models(
                include_memory_tools=include_memory_tools
            )
        }

    def claude_mcp_schema(
        self,
        tool_name: str,
        *,
        include_room_id: bool = True,
    ) -> dict[str, type[Any]]:
        """Build Claude SDK MCP schema from the canonical tool model."""
        model = self._tool_models[tool_name]
        schema: dict[str, type[Any]] = {}
        if include_room_id:
            schema["room_id"] = str

        for field_name, field in model.model_fields.items():
            schema[field_name] = self._annotation_to_runtime_type(field.annotation)
        return schema

    def claude_mcp_schemas(
        self,
        *,
        include_memory_tools: bool,
        include_room_id: bool = True,
    ) -> dict[str, dict[str, type[Any]]]:
        """Build Claude SDK MCP schemas for all enabled tools."""
        return {
            tool_name: self.claude_mcp_schema(
                tool_name,
                include_room_id=include_room_id,
            )
            for tool_name, _ in self.iter_models(
                include_memory_tools=include_memory_tools
            )
        }

    def pydantic_parameters(self, tool_name: str) -> list[inspect.Parameter]:
        """Build function parameters for PydanticAI tool registration."""
        model = self._tool_models[tool_name]
        parameters: list[inspect.Parameter] = []
        for field_name, field in model.model_fields.items():
            default = (
                inspect.Parameter.empty if field.is_required() else field.default
            )
            parameters.append(
                inspect.Parameter(
                    field_name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=field.annotation,
                )
            )
        return parameters

    @classmethod
    def _annotation_to_runtime_type(cls, annotation: Any) -> type[Any]:
        """Reduce rich typing annotations to MCP-compatible runtime types."""
        origin = get_origin(annotation)
        if origin is None:
            if annotation is Any:
                return str
            if isinstance(annotation, type):
                if issubclass(annotation, BaseModel):
                    return dict
                return annotation
            return str

        if origin in {list, tuple, set, frozenset}:
            return list
        if origin in {dict, Mapping}:
            return dict
        if origin in {Union, types.UnionType}:
            non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(non_none_args) == 1:
                return cls._annotation_to_runtime_type(non_none_args[0])
            return str
        if origin is Literal:
            literal_args = get_args(annotation)
            if not literal_args:
                return str
            value = literal_args[0]
            if isinstance(value, bool):
                return bool
            if isinstance(value, int):
                return int
            if isinstance(value, float):
                return float
            return str

        return str


TOOL_BINDINGS = ToolBindingRegistry(TOOL_MODELS)
