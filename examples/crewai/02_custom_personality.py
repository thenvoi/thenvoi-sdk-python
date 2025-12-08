"""
CrewAI agent with custom personality using the functional API.

This example shows how to customize the agent's role, goal, and backstory
to create a unique personality.
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
    agent_id, api_key = load_agent_config("personality_crewai_agent")

    # Create agent with custom personality
    await create_crewai_agent(
        agent_id=agent_id,
        api_key=api_key,
        llm=ChatOpenAI(model="gpt-4o"),
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
        # Custom CrewAI personality settings
        role="Friendly Tech Support Specialist",
        goal="Help users solve their technical problems with patience and expertise",
        backstory="""You are a seasoned tech support specialist with over 10 years 
        of experience helping people with all kinds of technical issues. You're known 
        for your patience, clear explanations, and ability to make complex topics 
        accessible to everyone. You always maintain a friendly and supportive tone, 
        and you never make users feel bad for asking questions.""",
        custom_instructions="""
        - Always greet users warmly
        - Break down complex solutions into simple steps
        - Ask clarifying questions when needed
        - Provide multiple solutions when possible
        - End conversations by asking if there's anything else you can help with
        """,
    )


if __name__ == "__main__":
    asyncio.run(main())
