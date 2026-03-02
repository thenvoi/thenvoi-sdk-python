"""Utilities for wrapping LangGraph graphs as tools."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.pregel import Pregel
from pydantic import create_model

from thenvoi.runtime.tool_bridge import format_tool_error

logger = logging.getLogger(__name__)


def graph_as_tool(
    graph: Pregel,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    result_formatter: Callable[[dict[str, Any]], Any] | None = None,
    isolate_thread: bool = True,
) -> BaseTool:
    """Wrap a compiled LangGraph graph as a LangChain tool.

    `input_schema` keys are passed directly to `graph.ainvoke(...)`, so they must
    match the graph state shape. If `isolate_thread` is true, each invocation gets
    a fresh subgraph thread; otherwise, calls share the room thread context.
    """
    if not name:
        raise ValueError("Tool name is required")
    if not description:
        raise ValueError("Tool description is required")
    if not input_schema:
        raise ValueError("Input schema is required")

    from pydantic import Field

    field_definitions = {}
    for param_name, param_desc in input_schema.items():
        field_definitions[param_name] = (Any, Field(description=param_desc))

    InputModel = create_model(f"{name.title()}Input", **field_definitions)  # type: ignore[call-overload]

    schema_desc = "\n".join([f"  - {k}: {v}" for k, v in input_schema.items()])
    full_description = f"{description}\n\nParameters:\n{schema_desc}"

    async def graph_tool_wrapper(
        *, config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        invocation_args = dict(kwargs)
        config_data: RunnableConfig = config or {}
        main_thread_id = config_data.get("configurable", {}).get("thread_id")

        logger.debug("[%s] Invoking subgraph with inputs: %s", name, invocation_args)
        logger.debug("[%s] Main thread_id: %s", name, main_thread_id)

        try:
            if isolate_thread:
                invocation_id = uuid.uuid4().hex[:8]
                subgraph_thread = f"subgraph:{name}:{main_thread_id}:{invocation_id}"
                logger.debug("[%s] Using isolated thread: %s", name, subgraph_thread)
            else:
                subgraph_thread = main_thread_id
                logger.debug(
                    "[%s] Using shared thread_id - subgraph will remember across invocations",
                    name,
                )

            result = await graph.ainvoke(
                invocation_args,
                {"configurable": {"thread_id": subgraph_thread}},
            )

            logger.debug("[%s] Subgraph execution completed", name)
            logger.debug("[%s] Raw result: %s", name, result)

            if result_formatter:
                formatted = result_formatter(result)
                if isinstance(formatted, (dict, list)):
                    formatted = json.dumps(formatted, indent=2)
                return str(formatted)
            return str(result)
        except Exception as error:
            logger.error("[%s] Subgraph execution failed: %s", name, error, exc_info=True)
            return format_tool_error(name, invocation_args, error)

    graph_tool_wrapper.__name__ = name
    graph_tool_wrapper.__doc__ = full_description

    return tool(graph_tool_wrapper, args_schema=InputModel)
