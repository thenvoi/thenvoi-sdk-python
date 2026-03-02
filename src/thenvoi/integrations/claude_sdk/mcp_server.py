"""Composable MCP server builder for Claude SDK adapter bindings."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from thenvoi.core.protocols import ToolDispatchProtocol
from thenvoi.runtime.tooling.custom_tools import (
    CustomToolDef,
    execute_custom_tool,
    get_custom_tool_name,
)
from thenvoi.runtime.tool_binding_factory import ToolBindingFactory
from thenvoi.runtime.tool_bindings import TOOL_BINDINGS
from thenvoi.runtime.tool_sessions import mcp_text_error, mcp_text_result
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)

McpToolDecorator = Callable[
    [str, str, dict[str, type[Any]]],
    Callable[[Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]], Any],
]
McpServerCreator = Callable[..., Any]


class ClaudeMcpServerFactory:
    """Build Claude MCP tool wrappers from shared runtime bindings."""

    def __init__(
        self,
        *,
        tool_decorator: McpToolDecorator,
        create_server: McpServerCreator,
        get_tools: Callable[[str], ToolDispatchProtocol | None],
        include_memory_tools: bool,
        custom_tools: Sequence[CustomToolDef],
        format_success_payload: Callable[[str, dict[str, Any], Any], dict[str, Any]],
    ) -> None:
        self._tool_decorator = tool_decorator
        self._create_server = create_server
        self._get_tools = get_tools
        self._include_memory_tools = include_memory_tools
        self._custom_tools = list(custom_tools)
        self._format_success_payload = format_success_payload
        self._binding_factory = ToolBindingFactory(binding_logger=logger)

    def create(self) -> Any:
        """Create the SDK MCP server for Thenvoi platform tools."""
        platform_tool_names = TOOL_BINDINGS.tool_names(
            include_memory_tools=self._include_memory_tools
        )
        all_tools = [
            self._build_platform_tool(tool_name) for tool_name in platform_tool_names
        ]
        all_tools.extend(self._build_custom_tool(tool_def) for tool_def in self._custom_tools)

        server = self._create_server(
            name="thenvoi",
            version="1.0.0",
            tools=all_tools,
        )
        logger.info(
            "Thenvoi MCP SDK server created with %s tools (%s custom)",
            len(all_tools),
            len(self._custom_tools),
        )
        return server

    def _build_platform_tool(self, tool_name: str) -> Any:
        schema = TOOL_BINDINGS.claude_mcp_schema(tool_name)
        description = get_tool_description(tool_name)

        handler = self._binding_factory.build_mcp_room_handler(
            tool_name=tool_name,
            get_tools=self._get_tools,
            map_args=lambda args: {k: v for k, v in args.items() if k != "room_id"},
            format_success_payload=lambda args, result: self._format_success_payload(
                tool_name,
                args,
                result,
            ),
            success_wrapper=mcp_text_result,
            error_wrapper=mcp_text_error,
        )
        handler.__name__ = tool_name.replace("thenvoi_", "")
        return self._tool_decorator(tool_name, description, schema)(handler)

    def _build_custom_tool(self, tool_def: CustomToolDef) -> Any:
        input_model, _ = tool_def
        tool_name = get_custom_tool_name(input_model)
        tool_description = input_model.__doc__ or f"Custom tool: {tool_name}"
        schema = input_model.model_json_schema()
        properties = schema.get("properties", {})

        mcp_schema: dict[str, type[Any]] = {"room_id": str}
        for prop_name, prop_def in properties.items():
            prop_type = prop_def.get("type", "string")
            if prop_type == "string":
                mcp_schema[prop_name] = str
            elif prop_type == "number":
                mcp_schema[prop_name] = float
            elif prop_type == "integer":
                mcp_schema[prop_name] = int
            elif prop_type == "boolean":
                mcp_schema[prop_name] = bool
            else:
                mcp_schema[prop_name] = str

        async def _handler(args: dict[str, Any]) -> dict[str, Any]:
            try:
                tool_args = {k: v for k, v in args.items() if k != "room_id"}
                result = await execute_custom_tool(tool_def, tool_args)
                return mcp_text_result(result)
            except Exception as error:
                logger.error(
                    "Custom tool %s failed: %s",
                    tool_name,
                    error,
                    exc_info=True,
                )
                return mcp_text_error(str(error))

        _handler.__name__ = tool_name
        return self._tool_decorator(tool_name, tool_description, mcp_schema)(_handler)

