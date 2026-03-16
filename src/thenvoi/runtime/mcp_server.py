from __future__ import annotations

import asyncio
import json
import logging
import socket
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool
from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Mount, Route
import uvicorn

from thenvoi.runtime.custom_tools import (
    CustomToolDef,
    execute_custom_tool,
    get_custom_tool_name,
)
from thenvoi.runtime.tools import AgentTools, ToolDefinition, iter_tool_definitions

logger = logging.getLogger(__name__)

LOCAL_MCP_HOST = "127.0.0.1"
LOCAL_MCP_PORT_MIN = 50000
LOCAL_MCP_PORT_MAX = 60000
LOCAL_MCP_SSE_PATH = "/sse"
LOCAL_MCP_MESSAGE_PATH = "/messages/"
LOCAL_MCP_HEALTH_PATH = "/healthz"
SERVER_START_TIMEOUT_S = 5.0

MCPToolExecutor = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class MCPToolRegistration:
    """A single MCP tool exposed by the local server."""

    name: str
    description: str
    input_model: type[BaseModel]
    execute: MCPToolExecutor

    def to_mcp_tool(self) -> Tool:
        """Convert the registration to an MCP tool definition."""
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=schema,
        )


def build_thenvoi_mcp_tool_registrations(
    agent_tools: AgentTools,
    *,
    include_memory: bool = False,
    additional_tools: list[CustomToolDef] | None = None,
) -> list[MCPToolRegistration]:
    """Build MCP tool registrations for Thenvoi tools and custom tools."""
    registrations = [
        _build_builtin_registration(agent_tools, definition)
        for definition in iter_tool_definitions(include_memory=include_memory)
    ]
    registrations.extend(
        _build_custom_registration(tool_def) for tool_def in additional_tools or []
    )
    _validate_unique_tool_names(registrations)
    return registrations


def _build_builtin_registration(
    agent_tools: AgentTools,
    definition: ToolDefinition,
) -> MCPToolRegistration:
    input_model = definition.input_model
    method = getattr(agent_tools, definition.method_name)

    async def execute(arguments: dict[str, Any]) -> Any:
        try:
            validated = input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ValueError(_format_validation_error(definition.name, exc)) from exc

        call_args = validated.model_dump(exclude_none=True)
        return await method(**call_args)

    return MCPToolRegistration(
        name=definition.name,
        description=input_model.__doc__ or "",
        input_model=input_model,
        execute=execute,
    )


def _build_custom_registration(tool_def: CustomToolDef) -> MCPToolRegistration:
    input_model, _ = tool_def
    tool_name = get_custom_tool_name(input_model)

    async def execute(arguments: dict[str, Any]) -> Any:
        return await execute_custom_tool(tool_def, arguments)

    return MCPToolRegistration(
        name=tool_name,
        description=input_model.__doc__ or "",
        input_model=input_model,
        execute=execute,
    )


def _format_validation_error(tool_name: str, error: ValidationError) -> str:
    errors = [
        f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
        for err in error.errors()
    ]
    return f"Invalid arguments for {tool_name}: {', '.join(errors)}"


def _validate_unique_tool_names(registrations: Sequence[MCPToolRegistration]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for registration in registrations:
        if registration.name in seen:
            duplicates.add(registration.name)
            continue
        seen.add(registration.name)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate MCP tool names: {duplicate_list}")


def _serialize_tool_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        payload = result
    elif isinstance(result, BaseModel):
        payload = result.model_dump(mode="json")
    else:
        payload = {"result": result}

    return json.loads(json.dumps(payload, default=str))


class LocalMCPServer:
    """A local localhost-only SSE MCP server."""

    def __init__(
        self,
        name: str,
        tool_registrations: Sequence[MCPToolRegistration],
        *,
        host: str = LOCAL_MCP_HOST,
        port_min: int = LOCAL_MCP_PORT_MIN,
        port_max: int = LOCAL_MCP_PORT_MAX,
        sse_path: str = LOCAL_MCP_SSE_PATH,
        message_path: str = LOCAL_MCP_MESSAGE_PATH,
    ) -> None:
        if host != LOCAL_MCP_HOST:
            raise ValueError(f"LocalMCPServer only supports host={LOCAL_MCP_HOST}")
        if port_min > port_max:
            raise ValueError("port_min must be less than or equal to port_max")

        registrations = list(tool_registrations)
        _validate_unique_tool_names(registrations)

        self._name = name
        self._host = host
        self._port_min = port_min
        self._port_max = port_max
        self._sse_path = sse_path
        self._message_path = message_path
        self._tool_registrations = {
            registration.name: registration for registration in registrations
        }
        self._mcp_server: Server[Any, Any] | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None
        self._port: int | None = None

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("Local MCP server has not started")
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}{self._sse_path}"

    async def start(self) -> None:
        """Start the local SSE MCP server."""
        if self._serve_task and not self._serve_task.done():
            return

        reserved_socket, port = self._reserve_socket()
        server = self._build_server()
        app = self._build_app(server)
        uvicorn_server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=self._host,
                port=port,
                lifespan="on",
                log_level="warning",
                access_log=False,
            )
        )
        serve_task = asyncio.create_task(
            uvicorn_server.serve(sockets=[reserved_socket])
        )

        self._socket = reserved_socket
        self._port = port
        self._mcp_server = server
        self._uvicorn_server = uvicorn_server
        self._serve_task = serve_task

        try:
            await self._wait_until_started()
        except Exception:
            await self.stop()
            raise

        logger.info(
            "Started local MCP server %s on %s:%s with %s tools",
            self._name,
            self._host,
            self._port,
            len(self._tool_registrations),
        )

    async def stop(self) -> None:
        """Stop the local SSE MCP server."""
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True

        if self._serve_task is not None:
            try:
                await self._serve_task
            except asyncio.CancelledError:
                logger.debug("Local MCP server task cancelled for %s", self._name)

        if self._socket is not None:
            self._socket.close()

        self._mcp_server = None
        self._uvicorn_server = None
        self._serve_task = None
        self._socket = None
        self._port = None

    def _build_server(self) -> Server[Any, Any]:
        server: Server[Any, Any] = Server(self._name)

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                registration.to_mcp_tool()
                for registration in self._tool_registrations.values()
            ]

        @server.call_tool(validate_input=False)
        async def call_tool(
            tool_name: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            registration = self._tool_registrations.get(tool_name)
            if registration is None:
                raise ValueError(f"Unknown tool: {tool_name}")

            result = await registration.execute(arguments)
            return _serialize_tool_result(result)

        return server

    def _build_app(self, server: Server[Any, Any]) -> Starlette:
        transport = SseServerTransport(self._message_path)

        async def sse_endpoint(request: Request) -> Response:
            async with transport.connect_sse(
                request.scope,
                request.receive,
                request._send,  # type: ignore[attr-defined]
            ) as streams:
                await server.run(
                    streams[0],
                    streams[1],
                    server.create_initialization_options(),
                )
            return Response()

        async def healthz(_: Request) -> PlainTextResponse:
            return PlainTextResponse("ok")

        return Starlette(
            routes=[
                Route(self._sse_path, endpoint=sse_endpoint, methods=["GET"]),
                Mount(self._message_path, app=transport.handle_post_message),
                Route(LOCAL_MCP_HEALTH_PATH, endpoint=healthz, methods=["GET"]),
            ]
        )

    def _reserve_socket(self) -> tuple[socket.socket, int]:
        last_error: OSError | None = None
        for port in range(self._port_min, self._port_max + 1):
            reserved_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reserved_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                reserved_socket.bind((self._host, port))
                reserved_socket.listen(2048)
                reserved_socket.setblocking(False)
                return reserved_socket, port
            except OSError as exc:
                last_error = exc
                reserved_socket.close()

        raise RuntimeError(
            "Could not find a free localhost MCP port in range "
            f"{self._port_min}-{self._port_max}"
        ) from last_error

    async def _wait_until_started(self) -> None:
        if self._serve_task is None or self._uvicorn_server is None:
            raise RuntimeError("Local MCP server task not initialized")

        deadline = asyncio.get_running_loop().time() + SERVER_START_TIMEOUT_S
        while not self._uvicorn_server.started:
            if self._serve_task.done():
                await self._serve_task
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Timed out waiting for local MCP server startup")
            await asyncio.sleep(0.05)
