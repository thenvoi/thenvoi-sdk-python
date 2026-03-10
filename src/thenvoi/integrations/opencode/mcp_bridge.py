"""In-process MCP server bridge for exposing custom tools to OpenCode.

Starts a lightweight HTTP SSE server that speaks the MCP protocol.
OpenCode connects to it as a "remote" MCP server and discovers/calls
tools natively.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class McpToolBridge:
    """Runs an in-process MCP server that exposes custom tools over SSE.

    OpenCode registers this server via ``POST /mcp`` and discovers tools
    through the standard MCP ``tools/list`` and ``tools/call`` JSON-RPC
    methods.
    """

    def __init__(
        self,
        *,
        custom_tools: list[tuple[Any, Any]],
        server_name: str = "thenvoi-tools",
    ) -> None:
        from thenvoi.runtime.custom_tools import CustomToolDef

        self._custom_tools: list[CustomToolDef] = list(custom_tools)
        self._server_name = server_name
        self._port: int | None = None
        self._server_task: asyncio.Task[None] | None = None

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def url(self) -> str | None:
        if self._port is None:
            return None
        return f"http://127.0.0.1:{self._port}"

    @property
    def server_name(self) -> str:
        return self._server_name

    async def start(self) -> str:
        """Start the MCP SSE server and return its base URL."""
        from mcp.server import Server
        from mcp.server.sse import SseServerTransport
        from mcp.types import TextContent, Tool
        from starlette.applications import Starlette
        from starlette.routing import Route

        from thenvoi.runtime.custom_tools import (
            execute_custom_tool,
            get_custom_tool_name,
        )

        mcp_server = Server(name=self._server_name)
        sse_transport = SseServerTransport("/messages/")

        tools_snapshot = list(self._custom_tools)

        @mcp_server.list_tools()
        async def _list_tools() -> list[Tool]:
            result: list[Tool] = []
            for input_model, _func in tools_snapshot:
                schema = input_model.model_json_schema()
                schema.pop("title", None)
                result.append(
                    Tool(
                        name=get_custom_tool_name(input_model),
                        description=input_model.__doc__ or "",
                        inputSchema=schema,
                    )
                )
            return result

        @mcp_server.call_tool()
        async def _call_tool(
            name: str, arguments: dict[str, Any] | None = None
        ) -> list[TextContent]:
            from thenvoi.runtime.custom_tools import find_custom_tool

            tool_def = find_custom_tool(tools_snapshot, name)
            if tool_def is None:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            try:
                result = await execute_custom_tool(tool_def, arguments or {})
            except Exception as exc:
                logger.exception("MCP bridge tool %s failed", name)
                return [TextContent(type="text", text=f"Tool error: {exc}")]

            if isinstance(result, str):
                text = result
            else:
                text = json.dumps(result, default=str)
            return [TextContent(type="text", text=text)]

        async def _handle_sse(request: Any) -> Any:
            async with sse_transport.connect_sse(
                request.scope,
                request.receive,
                request._send,  # type: ignore[arg-type]  # ASGI types
            ) as streams:
                await mcp_server.run(
                    streams[0], streams[1], mcp_server.create_initialization_options()
                )

        async def _handle_messages(request: Any) -> Any:
            await sse_transport.handle_post_message(
                request.scope,
                request.receive,
                request._send,  # type: ignore[arg-type]  # ASGI types
            )

        app = Starlette(
            routes=[
                Route("/sse", _handle_sse),
                Route("/messages/", _handle_messages, methods=["POST"]),
            ],
        )

        self._port = _find_free_port()

        import uvicorn

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self._port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        # MCP sessions are created per-SSE-connection in _handle_sse,
        # so we only need to run uvicorn here.
        self._server_task = asyncio.create_task(self._run_server(server))

        # Wait for server to be accepting connections
        for _ in range(50):
            await asyncio.sleep(0.05)
            if server.started:
                break

        url = f"http://127.0.0.1:{self._port}"
        logger.info(
            "MCP bridge started on %s with %d tools",
            url,
            len(tools_snapshot),
        )
        return url

    async def _run_server(
        self,
        uvicorn_server: Any,
    ) -> None:
        """Run uvicorn; MCP sessions are created per-SSE-connection."""
        try:
            await uvicorn_server.serve()
        except asyncio.CancelledError:
            uvicorn_server.should_exit = True
            raise

    async def stop(self) -> None:
        """Stop the MCP SSE server."""
        if self._server_task is not None:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
            self._server_task = None
            logger.info("MCP bridge stopped")
        self._port = None
