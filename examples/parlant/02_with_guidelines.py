"""
Parlant agent with behavioral guidelines.

Shows how to use Parlant's guideline system for controlled agent behavior.
Guidelines are condition/action pairs that ensure consistent responses.

Parlant's guidelines provide:
- Ensured rule-following behavior
- Contextual activation based on conditions
- Predictable, consistent responses

Run with:
    OPENAI_API_KEY=xxx python 02_with_guidelines.py
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ParlantAdapter
from thenvoi.config import load_agent_config

setup_logging()
logger = logging.getLogger(__name__)


# Define behavioral guidelines
GUIDELINES = [
    {
        "condition": "User asks for help or assistance",
        "action": "First acknowledge their request, then ask clarifying questions if needed before providing detailed help",
    },
    {
        "condition": "User mentions a specific participant or agent name",
        "action": "Use the lookup_peers tool to find available agents, then add_participant to bring them into the conversation",
    },
    {
        "condition": "User asks about current participants",
        "action": "Use get_participants to list all current room members",
    },
    {
        "condition": "User wants to create a new chat or discussion",
        "action": "Use create_chatroom to create a dedicated space for the new topic",
    },
    {
        "condition": "Conversation is ending or user says goodbye",
        "action": "Summarize what was discussed and offer to help with anything else",
    },
]


CUSTOM_PROMPT = """
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


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("parlant_agent")

    # Create adapter with guidelines and custom instructions
    adapter = ParlantAdapter(
        model="gpt-4o",
        custom_section=CUSTOM_PROMPT,
        guidelines=GUIDELINES,
        # Enable execution reporting to see tool calls in the chat
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

    logger.info("Starting Parlant agent with guidelines...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
