"""Shared factory helpers for framework-specific tool bindings."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

from thenvoi.core.protocols import ToolDispatchProtocol
from thenvoi.runtime.tool_bridge import invoke_platform_tool
from thenvoi.runtime.tool_sessions import ToolSessionRegistry

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _MaybeAsyncOnSuccess(Protocol):
    def __call__(self, room_id: str, result: Any) -> Any: ...


class ToolBindingLookupError(RuntimeError):
    """Raised when framework wrapper cannot resolve bound tools."""


class ToolBindingFactory:
    """Create framework wrappers over the shared `invoke_platform_tool` contract."""

    def __init__(self, *, binding_logger: logging.Logger | None = None) -> None:
        self._logger = binding_logger or logger

    async def invoke_room_tool(
        self,
        *,
        room_id: str,
        get_tools: Callable[[str], ToolDispatchProtocol | None],
        tool_name: str,
        arguments: dict[str, Any],
        missing_tools_message: str | None = None,
    ) -> Any:
        """Run a platform tool for a room-scoped binding."""
        tools = get_tools(room_id)
        if tools is None:
            message = missing_tools_message or f"No tools available for room {room_id}"
            raise ToolBindingLookupError(message)
        return await invoke_platform_tool(tools, tool_name, arguments)

    async def invoke_session_tool(
        self,
        *,
        session_id: str,
        registry: ToolSessionRegistry[ToolDispatchProtocol],
        tool_name: str,
        arguments: dict[str, Any],
        missing_tools_message: str | None = None,
    ) -> Any:
        """Run a platform tool for a session-scoped binding."""
        tools = registry.get_tools(session_id)
        if tools is None:
            message = missing_tools_message or "Error: No tools available in current context"
            raise ToolBindingLookupError(message)
        return await invoke_platform_tool(tools, tool_name, arguments)

    def build_mcp_room_handler(
        self,
        *,
        tool_name: str,
        get_tools: Callable[[str], ToolDispatchProtocol | None],
        map_args: Callable[[dict[str, Any]], dict[str, Any]],
        format_success_payload: Callable[[dict[str, Any], Any], Any],
        success_wrapper: Callable[[Any], dict[str, Any]],
        error_wrapper: Callable[[str], dict[str, Any]],
        room_id_field: str = "room_id",
        missing_tools_message: str | None = None,
        on_success: _MaybeAsyncOnSuccess | None = None,
        format_error_message: Callable[[Exception, str], str] | None = None,
    ) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        """Build a Claude-style MCP handler with shared room tool lookup and dispatch."""

        async def _handler(args: dict[str, Any]) -> dict[str, Any]:
            room_id = str(args.get(room_id_field, ""))
            tool_args = map_args(args)
            try:
                result = await self.invoke_room_tool(
                    room_id=room_id,
                    get_tools=get_tools,
                    tool_name=tool_name,
                    arguments=tool_args,
                    missing_tools_message=missing_tools_message,
                )
                if on_success is not None:
                    await self._await_maybe(on_success(room_id, result))
                payload = format_success_payload(tool_args, result)
                return success_wrapper(payload)
            except ToolBindingLookupError as error:
                return error_wrapper(str(error))
            except Exception as error:
                self._logger.error("%s failed: %s", tool_name, error, exc_info=True)
                message = (
                    format_error_message(error, room_id)
                    if format_error_message is not None
                    else str(error)
                )
                return error_wrapper(message)

        return _handler

    @staticmethod
    async def _await_maybe(value: Any) -> Any:
        if isinstance(value, Awaitable):
            return await value
        return value

