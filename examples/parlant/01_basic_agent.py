"""
Basic Parlant agent example using the official Parlant SDK.

This example shows how to create a Thenvoi agent using the Parlant SDK
directly, without any HTTP communication.

Run with:
    uv run python examples/parlant/01_basic_agent.py

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

    # Start Parlant server with OpenAI (requires OPENAI_API_KEY env var)
    async with p.Server(nlp_service=p.NLPServices.openai) as server:
        # Create Parlant agent with detailed description
        parlant_agent = await server.create_agent(
            name="Thenvoi Assistant",
            description="""You are a helpful, knowledgeable assistant.

When responding:
- Give detailed, specific answers to questions
- Remember information the user shares about themselves
- Reference previous parts of the conversation when relevant
- Ask follow-up questions to better understand the user's needs
- Be friendly but substantive - avoid generic or vague responses

If the user shares personal information (name, job, interests), acknowledge it
and use it to personalize your responses throughout the conversation.""",
        )

        logger.info(f"Parlant agent created: {parlant_agent.id}")

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

        logger.info("Starting Thenvoi agent with Parlant SDK...")
        await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
