"""
Connect a custom CrewAI Crew to Thenvoi platform.

This example shows how to create your own Crew with multiple agents
and connect it to the Thenvoi platform for message handling.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Crew, Process

from setup_logging import setup_logging
from thenvoi.agent.crewai import connect_crew_to_platform, get_thenvoi_tools
from thenvoi.agent.core import ThenvoiPlatformClient
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
    agent_id, api_key = load_agent_config("custom_crew_agent")

    # Create platform client (single client for everything)
    platform_client = ThenvoiPlatformClient(
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
    )

    # Fetch agent metadata first to get the API client configured
    await platform_client.fetch_agent_metadata()

    # Get platform tools using the same client
    platform_tools = get_thenvoi_tools(
        client=platform_client.api_client, agent_id=agent_id
    )

    llm = ChatOpenAI(model="gpt-4o")

    # Create a multi-agent crew
    # Agent 1: Researcher - gathers information
    researcher = Agent(
        role="Research Specialist",
        goal="Gather and analyze information to provide comprehensive answers",
        backstory="""You are an expert researcher with a talent for finding 
        and synthesizing information. You're thorough and always verify your findings.""",
        tools=platform_tools,
        llm=llm,
        verbose=True,
    )

    # Agent 2: Communicator - crafts responses
    communicator = Agent(
        role="Communication Expert",
        goal="Craft clear, helpful, and friendly responses to users",
        backstory="""You are a skilled communicator who excels at taking complex 
        information and presenting it in a clear, accessible way. You always 
        ensure users feel heard and supported.""",
        tools=platform_tools,
        llm=llm,
        verbose=True,
    )

    # Create the crew
    # Note: Tasks are created dynamically from incoming messages
    my_crew = Crew(
        agents=[researcher, communicator],
        tasks=[],  # Tasks will be added dynamically
        process=Process.sequential,
        verbose=True,
    )

    # Connect crew to platform
    await connect_crew_to_platform(
        crew=my_crew,
        platform_client=platform_client,
    )

    # Crew is now listening for messages!


if __name__ == "__main__":
    asyncio.run(main())
