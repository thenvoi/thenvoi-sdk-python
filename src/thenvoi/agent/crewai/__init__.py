"""
CrewAI adapter for Thenvoi platform.

Quick start (SDK-managed agent):
    from thenvoi.agent.crewai import create_crewai_agent

    agent = await create_crewai_agent(
        agent_id=agent_id,
        api_key=api_key,
        llm=ChatOpenAI(model="gpt-4o"),
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
    )

Custom crew integration (user-provided crew):
    from thenvoi.agent.crewai import connect_crew_to_platform

    my_crew = Crew(agents=[...], tasks=[...])
    agent = await connect_crew_to_platform(
        crew=my_crew,
        platform_client=platform_client,
    )

Advanced (class-based API):
    from thenvoi.agent.crewai import ThenvoiCrewAIAgent, ConnectedCrewAgent

    agent = ThenvoiCrewAIAgent(...)
    await agent.start()

Utilities (for custom integration):
    from thenvoi.agent.crewai import get_thenvoi_tools
"""

from .agent import (
    create_crewai_agent,
    ThenvoiCrewAIAgent,
    connect_crew_to_platform,
    ConnectedCrewAgent,
)
from .tools import get_thenvoi_tools

__all__ = [
    "create_crewai_agent",
    "ThenvoiCrewAIAgent",
    "connect_crew_to_platform",
    "ConnectedCrewAgent",
    "get_thenvoi_tools",
]
