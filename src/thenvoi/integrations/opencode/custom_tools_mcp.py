"""MCP server for OpenCode `additional_tools`."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any

from thenvoi.integrations.opencode.mcp_server import OpencodeMcpServerProtocol
from thenvoi.runtime.custom_tools import CustomToolDef

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class CustomToolMcpServer(OpencodeMcpServerProtocol):
    """Expose user-provided `additional_tools` as a local MCP SSE server."""

    def __init__(
        self,
        *,
        custom_tools: list[CustomToolDef],
        server_name: str = "thenvoi-custom-tools",
    ) -> None:
        self._custom_tools = list(custom_tools)
        self._server_name = server_name
        self._url: str | None = None
        self._server_task: asyncio.Task[None] | None = None

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def url(self) -> str | None:
        return self._url

    async def start(self) -> str | None:
        """Start the local MCP SSE server."""
        if self._server_task is not None and self._url is not None:
            return self._url

        from mcp.server import Server
        from mcp.server.sse import SseServerTransport
        from mcp.types import TextContent, Tool
        from starlette.applications import Starlette
        from starlette.routing import Route
        import uvicorn

        from thenvoi.runtime.custom_tools import (
            execute_custom_tool,
            find_custom_tool,
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
            tool_def = find_custom_tool(tools_snapshot, name)
            if tool_def is None:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            try:
                result = await execute_custom_tool(tool_def, arguments or {})
            except Exception as exc:
                logger.exception("Custom tool MCP server failed for %s", name)
                return [TextContent(type="text", text=f"Tool error: {exc}")]

            text = (
                result if isinstance(result, str) else json.dumps(result, default=str)
            )
            return [TextContent(type="text", text=text)]

        async def _handle_sse(request: Any) -> Any:
            async with sse_transport.connect_sse(
                request.scope,
                request.receive,
                request._send,  # type: ignore[arg-type]
            ) as streams:
                await mcp_server.run(
                    streams[0],
                    streams[1],
                    mcp_server.create_initialization_options(),
                )

        async def _handle_messages(request: Any) -> Any:
            await sse_transport.handle_post_message(
                request.scope,
                request.receive,
                request._send,  # type: ignore[arg-type]
            )

        app = Starlette(
            routes=[
                Route("/sse", _handle_sse),
                Route("/messages/", _handle_messages, methods=["POST"]),
            ],
        )

        port = _find_free_port()
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",
            )
        )
        self._server_task = asyncio.create_task(self._run_server(server))

        for _ in range(50):
            await asyncio.sleep(0.05)
            if server.started:
                break

        if not server.started:
            logger.warning(
                "Custom tool MCP server did not start within 2.5s for %s",
                self._server_name,
            )

        self._url = f"http://127.0.0.1:{port}/sse"
        logger.info(
            "Custom tool MCP server ready at %s with %d tools",
            self._url,
            len(tools_snapshot),
        )
        return self._url

    async def stop(self) -> None:
        """Stop the local MCP SSE server."""
        if self._server_task is not None:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
            self._server_task = None
            logger.info("Stopped custom tool MCP server %s", self._server_name)
        self._url = None

    async def _run_server(self, uvicorn_server: Any) -> None:
        try:
            await uvicorn_server.serve()
        except asyncio.CancelledError:
            uvicorn_server.should_exit = True
            raise
