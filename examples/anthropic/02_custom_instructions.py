"""
Agent with custom system prompt instructions.

Shows how to provide detailed custom instructions to shape agent behavior.
Also demonstrates execution reporting for visibility into tool calls.

Run with:
    ANTHROPIC_API_KEY=xxx python 02_custom_instructions.py
"""

import asyncio
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi.integrations.anthropic import ThenvoiAnthropicAgent
from thenvoi.config import load_agent_config

setup_logging()


CUSTOM_PROMPT = """
You are a technical support agent for a software company.

Guidelines:
- Be patient and thorough
- Ask clarifying questions before providing solutions
- Always verify the user's environment before troubleshooting
- Escalate to a human if you cannot resolve the issue

When helping users:
1. First acknowledge their issue
2. Ask for relevant details (OS, version, error messages)
3. Provide step-by-step solutions
4. Confirm the issue is resolved before closing
"""


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_API_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_API_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("support_agent")

    agent = ThenvoiAnthropicAgent(
        model="claude-sonnet-4-5-20250929",
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        custom_section=CUSTOM_PROMPT,
        # Enable execution reporting to see tool calls in the chat
        enable_execution_reporting=True,
    )

    print("Starting support agent...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
