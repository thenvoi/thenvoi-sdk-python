from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel

from thenvoi.runtime.custom_tools import get_custom_tool_name
from thenvoi.runtime.mcp_server import (
    LOCAL_MCP_HOST,
    MCPToolRegistration,
    LocalMCPServer,
    build_thenvoi_mcp_tool_registrations,
)
from thenvoi.runtime.tools import AgentTools


class EchoInput(BaseModel):
    """Echo text back to the caller."""

    message: str


async def echo_tool(input_data: EchoInput) -> dict[str, str]:
    return {"echo": input_data.message}


class TestBuildThenvoiMcpToolRegistrations:
    def test_includes_builtin_and_custom_tools(self) -> None:
        agent_tools = AgentTools("room-123", MagicMock(), [])

        registrations = build_thenvoi_mcp_tool_registrations(
            agent_tools,
            additional_tools=[(EchoInput, echo_tool)],
        )

        tool_names = {registration.name for registration in registrations}
        assert "thenvoi_send_message" in tool_names
        assert get_custom_tool_name(EchoInput) in tool_names

    def test_rejects_duplicate_tool_names(self) -> None:
        agent_tools = AgentTools("room-123", MagicMock(), [])

        with pytest.raises(ValueError, match="Duplicate MCP tool names"):
            build_thenvoi_mcp_tool_registrations(
                agent_tools,
                additional_tools=[
                    (EchoInput, echo_tool),
                    (EchoInput, echo_tool),
                ],
            )


class TestLocalMcpServer:
    @pytest.mark.asyncio
    async def test_serves_sse_tools_on_localhost(self) -> None:
        async def execute(arguments: dict[str, str]) -> dict[str, str]:
            return {"echo": arguments["message"]}

        server = LocalMCPServer(
            name="test-local-mcp",
            tool_registrations=[
                MCPToolRegistration(
                    name="echo",
                    description="Echo a message",
                    input_model=EchoInput,
                    execute=execute,
                )
            ],
            port_min=55050,
            port_max=55059,
        )

        await server.start()
        try:
            assert server.url.startswith(f"http://{LOCAL_MCP_HOST}:55")

            async with sse_client(server.url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    tools_result = await session.list_tools()
                    assert [tool.name for tool in tools_result.tools] == ["echo"]

                    result = await session.call_tool("echo", {"message": "hello"})
                    assert not result.isError
                    assert result.structuredContent == {"echo": "hello"}
        finally:
            await server.stop()
