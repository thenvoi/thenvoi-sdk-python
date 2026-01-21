"""
Parlant SDK integration for Thenvoi SDK.

This module provides the integration with the official Parlant SDK
(https://github.com/emcie-co/parlant) for building guideline-based
conversational AI agents.

Usage:
    from thenvoi import Agent
    from thenvoi.adapters import ParlantAdapter

    adapter = ParlantAdapter(
        model="gpt-4o",
        guidelines=[
            {"condition": "User asks about refunds", "action": "Check order status first"}
        ],
    )
    agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
    await agent.run()

Internal modules (session_manager, tools) are used by the adapter.
"""

from .session_manager import ParlantSessionManager
from .tools import create_parlant_tools, ParlantToolContext

__all__ = [
    "ParlantSessionManager",
    "create_parlant_tools",
    "ParlantToolContext",
]
