"""Tool call validation and dispatch service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ToolDispatcher:
    """Validate tool arguments and dispatch to platform tool handlers."""

    def __init__(self, *, tool_models: dict[str, type[BaseModel]]) -> None:
        self._tool_models = tool_models

    async def execute_or_raise(
        self,
        tools: Any,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a validated tool call and raise ToolExecutionError on failure."""
        from thenvoi.runtime.tool_bridge import as_tool_execution_error

        raw_arguments = dict(arguments)
        try:
            if tool_name in self._tool_models:
                model = self._tool_models[tool_name]
                validated = model.model_validate(arguments)
                arguments = validated.model_dump(exclude_none=True)
        except Exception as error:
            raise as_tool_execution_error(tool_name, raw_arguments, error) from error

        if tool_name not in self._tool_models:
            raise as_tool_execution_error(
                tool_name,
                raw_arguments,
                ValueError(f"Unknown tool: {tool_name}"),
            )

        try:
            from thenvoi.runtime.tool_bridge import dispatch_platform_tool_call

            return await dispatch_platform_tool_call(tools, tool_name, arguments)
        except Exception as error:
            raise as_tool_execution_error(tool_name, arguments, error) from error

    async def execute(self, tools: Any, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a validated tool call and raise ToolExecutionError on failure."""
        return await self.execute_or_raise(tools, tool_name, arguments)
