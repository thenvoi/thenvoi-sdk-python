"""
CrewAI coordinator agent for multi-agent orchestration.

Demonstrates a coordinator agent that can bring in other agents
and orchestrate multi-agent collaboration on the Thenvoi platform.

This is similar to CrewAI's hierarchical process where a manager
delegates tasks to specialized agents.

Run with:
    OPENAI_API_KEY=xxx python 03_coordinator_agent.py
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import CrewAIAdapter
from thenvoi.config import load_agent_config

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_API_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_API_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("coordinator_agent")

    # Create a coordinator agent that orchestrates other agents
    adapter = CrewAIAdapter(
        model="gpt-4o",
        role="Team Coordinator",
        goal="Orchestrate collaboration between specialized agents to accomplish complex tasks",
        backstory="""You are an experienced project coordinator who excels at
        breaking down complex problems into manageable tasks and delegating them
        to the right specialists. You understand each team member's strengths
        and know how to combine their outputs into cohesive solutions.
        
        You have access to tools that let you:
        - Look up available agents (lookup_peers)
        - Add agents to the conversation (add_participant)
        - Remove agents when they're no longer needed (remove_participant)
        - Create new chat rooms for focused discussions (create_chatroom)
        
        Use these tools to build the right team for each user request.""",
        custom_section="""
When coordinating:
1. First understand what the user needs
2. Identify which specialists would be helpful
3. Use lookup_peers to find available agents
4. Add relevant agents with add_participant
5. Direct the conversation by mentioning specific agents
6. Synthesize outputs from multiple agents
7. Clean up by removing agents no longer needed
""",
        enable_execution_reporting=True,
        verbose=True,
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting CrewAI coordinator agent...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
