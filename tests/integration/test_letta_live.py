"""Live integration tests for Letta adapter.

Requires LETTA_API_KEY and LETTA_AGENT_ID environment variables to be set.
Skipped in CI by default.

Run with:
    LETTA_API_KEY="sk-let-..." LETTA_AGENT_ID="agent-..." uv run pytest tests/integration/test_letta_live.py -v -s --no-cov
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.requires_api

LETTA_API_KEY = os.environ.get("LETTA_API_KEY", "")
LETTA_AGENT_ID = os.environ.get("LETTA_AGENT_ID", "")

skip_no_key = pytest.mark.skipif(
    not LETTA_API_KEY,
    reason="LETTA_API_KEY not set",
)

skip_no_agent = pytest.mark.skipif(
    not LETTA_AGENT_ID,
    reason="LETTA_AGENT_ID not set",
)


@skip_no_key
@skip_no_agent
@pytest.mark.asyncio
async def test_existing_agent_message() -> None:
    """Send a message to a pre-existing Letta agent and verify response."""
    from letta_client import AsyncLetta

    client = AsyncLetta(api_key=LETTA_API_KEY)

    # Verify agent exists
    agent = await client.agents.retrieve(LETTA_AGENT_ID)
    assert agent.id == LETTA_AGENT_ID

    # Send message
    response = await client.agents.messages.create(
        agent_id=LETTA_AGENT_ID,
        messages=[{"role": "user", "content": "Say hello in exactly one word."}],
    )
    assert response.messages
    msg_types = [getattr(m, "message_type", None) for m in response.messages]
    assert any(t == "assistant_message" for t in msg_types), (
        f"Expected assistant_message, got: {msg_types}"
    )


@skip_no_key
@skip_no_agent
@pytest.mark.asyncio
async def test_client_tools_approval_flow() -> None:
    """Test the client_tools + approval_request_message flow with existing agent."""
    from letta_client import AsyncLetta

    client = AsyncLetta(api_key=LETTA_API_KEY)

    client_tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        }
    ]

    response = await client.agents.messages.create(
        agent_id=LETTA_AGENT_ID,
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        client_tools=client_tools,
    )

    # Check if we got an approval request
    approval_msgs = [
        m
        for m in response.messages
        if getattr(m, "message_type", None) == "approval_request_message"
    ]

    if approval_msgs:
        # Send approval response
        tool_call = approval_msgs[0].tool_call
        response2 = await client.agents.messages.create(
            agent_id=LETTA_AGENT_ID,
            messages=[
                {
                    "type": "approval",
                    "approvals": [
                        {
                            "type": "tool",
                            "tool_call_id": tool_call.tool_call_id,
                            "tool_return": '{"temperature": "15°C", "condition": "Cloudy"}',
                            "status": "success",
                        }
                    ],
                }
            ],
            client_tools=client_tools,
        )
        assert response2.messages
