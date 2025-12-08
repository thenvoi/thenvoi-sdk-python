"""
Simple CrewAI agent example using the functional API.

This is the simplest way to create a Thenvoi agent with CrewAI - just call
create_crewai_agent() and it handles everything.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from setup_logging import setup_logging
from thenvoi.agent.crewai import create_crewai_agent
from thenvoi.config import load_agent_config

setup_logging()


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    thenvoi_restapi_url = os.getenv("THENVOI_REST_API_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not thenvoi_restapi_url:
        raise ValueError("THENVOI_REST_API_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("simple_crewai_agent")

    # Create and start agent in one call - that's it!
    await create_crewai_agent(
        agent_id=agent_id,
        api_key=api_key,
        llm=ChatOpenAI(model="gpt-4o"),
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
    )

    # Agent is now listening for messages!


if __name__ == "__main__":
    asyncio.run(main())
