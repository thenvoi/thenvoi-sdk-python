"""
Parlant agent with behavioral guidelines using the official Parlant SDK.

This example shows how to use Parlant's guideline system for controlled
agent behavior using the SDK directly.

Run with:
    uv run python examples/parlant/02_with_guidelines.py

See also: https://github.com/emcie-co/parlant/blob/develop/examples/travel_voice_agent.py
"""

import asyncio
import logging
import os

import parlant.sdk as p
from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ParlantAdapter
from thenvoi.config import load_agent_config

setup_logging()
logger = logging.getLogger(__name__)


CUSTOM_DESCRIPTION = """
You are a collaborative assistant in the Thenvoi multi-agent platform.

Your role:
- Help users navigate multi-agent conversations
- Facilitate collaboration between different agents
- Manage participants in chat rooms
- Create new chat rooms when needed for specific topics

When interacting:
1. Be proactive about suggesting relevant agents to add
2. Keep responses focused and actionable
3. Always confirm actions taken with the user
"""


async def setup_agent_with_guidelines(server: p.Server) -> p.Agent:
    """Create and configure a Parlant agent with guidelines."""
    agent = await server.create_agent(
        name="Collaborative Assistant",
        description=CUSTOM_DESCRIPTION,
    )

    # Add behavioral guidelines using the SDK
    await agent.create_guideline(
        condition="User asks for help or assistance",
        action="First acknowledge their request, then ask clarifying questions if needed before providing detailed help",
    )

    await agent.create_guideline(
        condition="User mentions a specific participant or agent name",
        action="Use the lookup_peers tool to find available agents, then add_participant to bring them into the conversation",
    )

    await agent.create_guideline(
        condition="User asks about current participants",
        action="Use get_participants to list all current room members",
    )

    await agent.create_guideline(
        condition="User wants to create a new chat or discussion",
        action="Use create_chatroom to create a dedicated space for the new topic",
    )

    await agent.create_guideline(
        condition="Conversation is ending or user says goodbye",
        action="Summarize what was discussed and offer to help with anything else",
    )

    return agent


async def main() -> None:
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("parlant_agent")

    # Start Parlant server with OpenAI
    async with p.Server(nlp_service=p.NLPServices.openai) as server:
        # Create Parlant agent with guidelines
        parlant_agent = await setup_agent_with_guidelines(server)
        logger.info(f"Parlant agent with guidelines created: {parlant_agent.id}")

        # Create adapter using Parlant SDK directly
        adapter = ParlantAdapter(
            server=server,
            parlant_agent=parlant_agent,
        )

        # Create and start Thenvoi agent
        agent = Agent.create(
            adapter=adapter,
            agent_id=agent_id,
            api_key=api_key,
            ws_url=ws_url,
            rest_url=rest_url,
        )

        logger.info("Starting Thenvoi agent with Parlant SDK and guidelines...")
        await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
