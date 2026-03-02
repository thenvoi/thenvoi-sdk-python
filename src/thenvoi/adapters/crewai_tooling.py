"""Composable CrewAI tool bindings used by `CrewAIAdapter`."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from thenvoi.adapters.platform_tool_bindings import crewai_tool_bindings
from thenvoi.core.protocols import MessagingDispatchToolsProtocol
from thenvoi.runtime.tooling.custom_tools import CustomToolDef, get_custom_tool_name
from thenvoi.runtime.tool_bridge import invoke_platform_tool
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CrewAIToolRuntime:
    """Own CrewAI-specific tool execution/reporting and wrapper generation."""

    def __init__(
        self,
        *,
        base_tool_type: type[Any],
        run_async: Callable[[Coroutine[Any, Any, str]], str],
        get_room_context: Callable[
            [], tuple[str, MessagingDispatchToolsProtocol] | None
        ],
        enable_execution_reporting: bool,
        enable_memory_tools: bool,
        custom_tools: list[CustomToolDef],
        schema_overrides: Mapping[str, type[BaseModel]],
    ) -> None:
        self._base_tool_type = base_tool_type
        self._run_async = run_async
        self._get_room_context = get_room_context
        self._enable_execution_reporting = enable_execution_reporting
        self._enable_memory_tools = enable_memory_tools
        self._custom_tools = custom_tools
        self._schema_overrides = dict(schema_overrides)
        self._reporting_errors: list[dict[str, str]] = []

    def execute_tool(
        self,
        tool_name: str,
        coro_factory: Callable[
            [MessagingDispatchToolsProtocol], Coroutine[Any, Any, str]
        ],
    ) -> str:
        """Execute a coroutine-based tool call from CrewAI sync `_run` hooks."""
        context = self._get_room_context()
        if not context:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No room context available - tool called outside message handling",
                }
            )

        room_id, tools = context

        async def _execute() -> str:
            try:
                return await coro_factory(tools)
            except Exception as error:
                error_msg = str(error)
                logger.error("%s failed in room %s: %s", tool_name, room_id, error_msg)
                await self.report_tool_result(
                    tools,
                    tool_name,
                    error_msg,
                    is_error=True,
                )
                return json.dumps({"status": "error", "message": error_msg})

        return self._run_async(_execute())

    async def report_tool_call(
        self,
        tools: MessagingDispatchToolsProtocol,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> None:
        """Emit tool_call event when execution reporting is enabled."""
        if not self._enable_execution_reporting:
            return
        try:
            await tools.send_event(
                content=json.dumps({"tool": tool_name, "input": input_data}),
                message_type="tool_call",
            )
        except Exception as error:
            self._record_reporting_error("tool_call", tool_name, error)
            logger.warning("Failed to send tool_call event: %s", error)

    async def report_tool_result(
        self,
        tools: MessagingDispatchToolsProtocol,
        tool_name: str,
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Emit tool_result event when execution reporting is enabled."""
        if not self._enable_execution_reporting:
            return
        try:
            key = "error" if is_error else "result"
            await tools.send_event(
                content=json.dumps({"tool": tool_name, key: result}),
                message_type="tool_result",
            )
        except Exception as error:
            self._record_reporting_error("tool_result", tool_name, error)
            logger.warning("Failed to send tool_result event: %s", error)

    def _record_reporting_error(
        self,
        operation: str,
        tool_name: str,
        error: Exception,
    ) -> None:
        """Track best-effort reporting failures for debugging and test introspection."""
        self._reporting_errors.append(
            {
                "operation": operation,
                "tool_name": tool_name,
                "error": str(error),
            }
        )

    def build_tools(self) -> list[Any]:
        """Build platform + custom CrewAI tool instances."""
        reportless_tools = {"thenvoi_send_event"}
        bindings = crewai_tool_bindings(
            include_memory_tools=self._enable_memory_tools,
            overrides=self._schema_overrides,
        )

        platform_tools: list[Any] = []
        for binding in bindings:
            tool_class = self._build_platform_tool_class(
                binding.name,
                binding.args_schema,
                description=binding.description,
                report_execution=binding.name not in reportless_tools,
            )
            platform_tools.append(tool_class())

        custom_tools = self._build_custom_tools()
        if custom_tools:
            logger.debug(
                "Added %s custom tools: %s",
                len(custom_tools),
                [tool.name for tool in custom_tools],
            )

        return platform_tools + custom_tools

    def build_custom_tools(self) -> list[Any]:
        """Build custom CrewAI tools only."""
        return self._build_custom_tools()

    def build_platform_tool_class(
        self,
        tool_name: str,
        args_schema: type[BaseModel],
        *,
        description: str | None = None,
        report_execution: bool,
    ) -> type[Any]:
        """Public wrapper for dynamic platform tool class generation."""
        return self._build_platform_tool_class(
            tool_name,
            args_schema,
            description=description,
            report_execution=report_execution,
        )

    def _build_custom_tools(self) -> list[Any]:
        """Convert custom SDK tools into CrewAI BaseTool-compatible instances."""
        tools: list[Any] = []
        for input_model, func in self._custom_tools:
            tool_name = get_custom_tool_name(input_model)
            tool_description = input_model.__doc__ or f"Execute {tool_name}"
            tools.append(
                self._build_single_custom_tool(
                    tool_name=tool_name,
                    tool_description=tool_description,
                    input_model=input_model,
                    handler=func,
                )
            )
        return tools

    def _build_single_custom_tool(
        self,
        *,
        tool_name: str,
        tool_description: str,
        input_model: type[BaseModel],
        handler: Any,
    ) -> Any:
        """Build one custom tool instance with stable closure bindings."""
        runtime = self

        class CustomCrewAITool(self._base_tool_type):
            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                async def _execute(
                    bound_tools: MessagingDispatchToolsProtocol,
                ) -> str:
                    try:
                        validated = input_model.model_validate(kwargs)
                        await runtime.report_tool_call(bound_tools, tool_name, kwargs)

                        if asyncio.iscoroutinefunction(handler):
                            result = await handler(validated)
                        else:
                            result = handler(validated)

                        await runtime.report_tool_result(bound_tools, tool_name, result)
                        return json.dumps(
                            {"status": "success", "result": result},
                            default=str,
                        )
                    except Exception as error:
                        error_msg = str(error)
                        logger.error(
                            "Custom tool %s failed: %s",
                            tool_name,
                            error_msg,
                        )
                        await runtime.report_tool_result(
                            bound_tools,
                            tool_name,
                            error_msg,
                            is_error=True,
                        )
                        return json.dumps({"status": "error", "message": error_msg})

                return runtime.execute_tool(tool_name, _execute)

        CustomCrewAITool.name = tool_name
        CustomCrewAITool.description = tool_description
        CustomCrewAITool.args_schema = input_model

        return CustomCrewAITool()

    def _build_platform_tool_class(
        self,
        tool_name: str,
        args_schema: type[BaseModel],
        *,
        description: str | None = None,
        report_execution: bool,
    ) -> type[Any]:
        """Create a dynamic CrewAI BaseTool class for a platform tool name."""
        runtime = self

        def _run(self: Any, *_args: Any, **kwargs: Any) -> Any:
            raw_args = dict(kwargs)

            async def _execute(
                bound_tools: MessagingDispatchToolsProtocol,
            ) -> str:
                report_input = (
                    {} if tool_name == "thenvoi_get_participants" else dict(raw_args)
                )
                if report_execution:
                    await runtime.report_tool_call(bound_tools, tool_name, report_input)

                result = await invoke_platform_tool(bound_tools, tool_name, raw_args)
                payload = runtime.format_success_payload(tool_name, result)

                if report_execution:
                    await runtime.report_tool_result(
                        bound_tools,
                        tool_name,
                        runtime.report_result_payload(tool_name, payload, result),
                    )
                return json.dumps(payload, default=str)

            return runtime.execute_tool(tool_name, _execute)

        class_name = (
            f"{'_'.join(part.capitalize() for part in tool_name.split('_'))}Tool"
        )
        tool_description = description or get_tool_description(tool_name)
        return type(
            class_name,
            (self._base_tool_type,),
            {
                "name": tool_name,
                "description": tool_description,
                "args_schema": args_schema,
                "_run": _run,
            },
        )

    @staticmethod
    def format_success_payload(tool_name: str, result: Any) -> dict[str, Any]:
        """Return consistent success payload shape across CrewAI wrappers."""
        if tool_name == "thenvoi_send_message":
            return {"status": "success", "message": "Message sent"}
        if tool_name == "thenvoi_send_event":
            return {"status": "success", "message": "Event sent"}
        if tool_name == "thenvoi_get_participants":
            participants = result if isinstance(result, list) else []
            return {
                "status": "success",
                "participants": participants,
                "count": len(participants),
            }
        if tool_name == "thenvoi_create_chatroom":
            return {
                "status": "success",
                "message": "Chat room created",
                "room_id": result,
            }
        if isinstance(result, dict):
            return {"status": "success", **result}
        return {"status": "success", "result": result}

    @staticmethod
    def report_result_payload(
        tool_name: str,
        success_payload: dict[str, Any],
        result: Any,
    ) -> Any:
        """Normalize execution-report payload for tool_result events."""
        if tool_name == "thenvoi_send_message":
            return "success"
        if tool_name in {"thenvoi_get_participants", "thenvoi_create_chatroom"}:
            return success_payload
        return result
