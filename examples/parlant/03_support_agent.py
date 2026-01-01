"""
Customer support agent using Parlant guidelines.

Demonstrates a realistic use case: a customer support agent with
specific behavioral guidelines for handling support requests.

This example shows Parlant's strength in ensuring consistent
customer-facing behavior through guidelines.

Run with:
    OPENAI_API_KEY=xxx python 03_support_agent.py
"""

import asyncio
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ParlantAdapter
from thenvoi.config import load_agent_config

setup_logging()


# Customer support guidelines
SUPPORT_GUIDELINES = [
    {
        "condition": "Customer asks about refunds or returns",
        "action": "Express empathy first, then ask for order details (order number, item) before providing refund information",
    },
    {
        "condition": "Customer is frustrated or upset",
        "action": "Acknowledge their frustration, apologize for any inconvenience, and focus on finding a solution",
    },
    {
        "condition": "Customer asks a technical question",
        "action": "Ask about their setup (device, OS, version) before troubleshooting",
    },
    {
        "condition": "Issue cannot be resolved by this agent",
        "action": "Explain the limitation clearly and offer to escalate to a specialist by adding them to the conversation",
    },
    {
        "condition": "Customer provides positive feedback",
        "action": "Thank them warmly and ask if there's anything else you can help with",
    },
    {
        "condition": "Customer mentions urgency or deadline",
        "action": "Prioritize their request and provide the fastest path to resolution",
    },
]


SUPPORT_PROMPT = """
You are a customer support agent for TechCo Solutions.

Your responsibilities:
- Handle customer inquiries with professionalism and empathy
- Resolve issues efficiently while maintaining quality
- Escalate complex issues to specialists when needed
- Document interactions for follow-up

Communication style:
- Friendly but professional
- Clear and concise
- Solution-focused
- Proactive about next steps

Remember:
- Customer satisfaction is the top priority
- Never make promises you can't keep
- Always follow up on commitments
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

    # Create support agent with specialized guidelines
    adapter = ParlantAdapter(
        model="gpt-4o",
        custom_section=SUPPORT_PROMPT,
        guidelines=SUPPORT_GUIDELINES,
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

    print("Starting customer support agent...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

