"""Convert AgentTools to LangChain StructuredTool format."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from thenvoi.core.protocols import ToolDispatchProtocol
from thenvoi.runtime.tool_bridge import (
    build_tool_failure,
    get_platform_tool_order,
    invoke_platform_tool,
)
from thenvoi.runtime.tools import TOOL_MODELS, get_tool_description


def _build_structured_tool(
    tools: ToolDispatchProtocol,
    tool_name: str,
) -> StructuredTool:
    args_schema = TOOL_MODELS[tool_name]

    async def _wrapper(**kwargs: Any) -> Any:
        try:
            return await invoke_platform_tool(tools, tool_name, kwargs)
        except Exception as error:
            failure = build_tool_failure(tool_name, kwargs, error)
            return failure.message

    return StructuredTool.from_function(
        coroutine=_wrapper,
        name=tool_name,
        description=get_tool_description(tool_name),
        args_schema=args_schema,
    )


def agent_tools_to_langchain(
    tools: ToolDispatchProtocol, *, include_memory_tools: bool = False
) -> list[Any]:
    """Convert AgentTools to LangChain StructuredTool instances."""
    names = get_platform_tool_order(include_memory_tools=include_memory_tools)
    return [_build_structured_tool(tools, tool_name) for tool_name in names]
