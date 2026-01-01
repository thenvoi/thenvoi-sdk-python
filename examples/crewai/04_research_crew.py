"""
Complete CrewAI-style crew with multiple specialized agents.

This example demonstrates how to run multiple specialized agents as a crew
on the Thenvoi platform. Each agent has a specific role and they collaborate
to accomplish complex tasks.

The crew consists of:
- Research Analyst: Gathers and analyzes information
- Content Writer: Creates clear, engaging content
- Editor: Reviews and refines content for quality

To run the full crew, start each agent in separate terminals:

Terminal 1:
    OPENAI_API_KEY=xxx python 04_research_crew.py researcher

Terminal 2:
    OPENAI_API_KEY=xxx python 04_research_crew.py writer

Terminal 3:
    OPENAI_API_KEY=xxx python 04_research_crew.py editor

Then in the Thenvoi chat, add all agents to the same room and they'll
collaborate on requests.
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import CrewAIAdapter
from thenvoi.config import load_agent_config

setup_logging()
logger = logging.getLogger(__name__)


# Define crew member configurations
CREW_MEMBERS = {
    "researcher": {
        "config_name": "research_agent",
        "role": "Research Analyst",
        "goal": "Gather comprehensive information and provide well-researched insights",
        "backstory": """You are a meticulous research analyst with expertise in 
        finding reliable sources, analyzing data, and synthesizing complex information
        into actionable insights. You're known for your thoroughness and ability to
        uncover relevant details that others might miss.
        
        When working with the crew:
        - Focus on gathering facts and data
        - Cite sources when possible
        - Present findings clearly for the writer to use
        - Flag any uncertainties or gaps in information""",
        "custom_section": """
Your workflow:
1. When asked to research a topic, gather comprehensive information
2. Organize findings into clear categories
3. Highlight key insights and important details
4. Pass your research to @Content Writer for drafting
""",
    },
    "writer": {
        "config_name": "writer_agent",
        "role": "Content Writer",
        "goal": "Transform research into clear, engaging, and well-structured content",
        "backstory": """You are a skilled content writer who excels at taking
        complex information and turning it into readable, engaging content.
        You have a talent for finding the right tone and structure for any
        audience, and you always aim for clarity without sacrificing depth.
        
        When working with the crew:
        - Wait for the Research Analyst to provide findings
        - Create well-structured drafts based on research
        - Use clear, accessible language
        - Pass drafts to the Editor for review""",
        "custom_section": """
Your workflow:
1. Review research provided by @Research Analyst
2. Create a well-structured draft with clear sections
3. Ensure the content flows logically
4. Send your draft to @Editor for review and refinement
""",
    },
    "editor": {
        "config_name": "editor_agent",
        "role": "Editor",
        "goal": "Ensure content quality through careful review and refinement",
        "backstory": """You are an experienced editor with a keen eye for detail
        and a passion for quality. You excel at polishing content while
        maintaining the writer's voice, catching errors, improving clarity,
        and ensuring the final product meets high standards.
        
        When working with the crew:
        - Review drafts from the Content Writer
        - Check for accuracy, clarity, and consistency
        - Suggest improvements and catch errors
        - Provide the final polished version to the user""",
        "custom_section": """
Your workflow:
1. Review drafts from @Content Writer
2. Check for grammar, clarity, and flow
3. Verify accuracy against research from @Research Analyst
4. Provide the final polished content to the user
5. If major issues found, send back to writer with feedback
""",
    },
}


async def main():
    load_dotenv()

    # Determine which crew member to run
    if len(sys.argv) < 2:
        logger.error("Usage: python 04_research_crew.py <role>")
        logger.info("Available roles: researcher, writer, editor")
        logger.info("To run the full crew, start each in a separate terminal:")
        logger.info("  Terminal 1: python 04_research_crew.py researcher")
        logger.info("  Terminal 2: python 04_research_crew.py writer")
        logger.info("  Terminal 3: python 04_research_crew.py editor")
        sys.exit(1)

    role = sys.argv[1].lower()
    if role not in CREW_MEMBERS:
        logger.error(f"Unknown role: {role}")
        logger.info(f"Available roles: {', '.join(CREW_MEMBERS.keys())}")
        sys.exit(1)

    member = CREW_MEMBERS[role]

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_API_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_API_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config(member["config_name"])

    # Create adapter with crew member configuration
    adapter = CrewAIAdapter(
        model="gpt-4o",
        role=member["role"],
        goal=member["goal"],
        backstory=member["backstory"],
        custom_section=member["custom_section"],
        enable_execution_reporting=True,
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info(f"Starting {member['role']}...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

