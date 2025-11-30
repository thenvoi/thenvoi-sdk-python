"""
Simple LangGraph agent example using the functional API.

This is the simplest way to create a Thenvoi agent - just call
create_langgraph_agent() and it handles everything.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from setup_logging import setup_logging
from thenvoi.agent.langgraph import create_langgraph_agent
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
    agent_id, api_key = load_agent_config("simple_agent")

    # Create and start agent in one call - that's it!
    await create_langgraph_agent(
        agent_id=agent_id,
        api_key=api_key,
        llm=ChatOpenAI(model="gpt-4o"),
        checkpointer=InMemorySaver(),
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
    )

    # Agent is now listening for messages!


if __name__ == "__main__":
    asyncio.run(main())
