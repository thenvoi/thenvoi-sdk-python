"""Shared adapter-layer helpers for dynamic platform tool registration."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from thenvoi.runtime.tool_bindings import TOOL_BINDINGS
from thenvoi.runtime.tools import get_tool_description


@dataclass(frozen=True)
class CrewAIToolBinding:
    """Resolved CrewAI tool registration metadata."""

    name: str
    description: str
    args_schema: type[BaseModel]


def platform_tool_names(*, include_memory_tools: bool) -> list[str]:
    """Return canonical platform tool names in registration order."""
    return TOOL_BINDINGS.tool_names(include_memory_tools=include_memory_tools)


def crewai_tool_bindings(
    *,
    include_memory_tools: bool,
    overrides: Mapping[str, type[BaseModel]] | None = None,
) -> list[CrewAIToolBinding]:
    """Resolve CrewAI tool binding metadata from shared registry definitions."""
    schemas = TOOL_BINDINGS.crewai_schemas(
        include_memory_tools=include_memory_tools,
        overrides=overrides,
    )
    bindings: list[CrewAIToolBinding] = []
    for tool_name in platform_tool_names(include_memory_tools=include_memory_tools):
        bindings.append(
            CrewAIToolBinding(
                name=tool_name,
                description=get_tool_description(tool_name),
                args_schema=schemas[tool_name],
            )
        )
    return bindings


def build_pydantic_tool_function(
    tool_name: str,
    *,
    context_annotation: Any,
    invoker: Callable[[Any, str, dict[str, Any]], Awaitable[Any]],
) -> Callable[..., Any]:
    """Create a PydanticAI tool function with shared signature/description rules."""
    parameter_defs = [
        inspect.Parameter(
            "ctx",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=context_annotation,
        ),
        *TOOL_BINDINGS.pydantic_parameters(tool_name),
    ]
    signature = inspect.Signature(
        parameters=parameter_defs,
        return_annotation=Any,
    )

    async def _tool(ctx: Any, **kwargs: Any) -> Any:
        return await invoker(ctx, tool_name, kwargs)

    _tool.__name__ = tool_name
    _tool.__doc__ = get_tool_description(tool_name)
    _tool.__signature__ = signature
    return _tool
