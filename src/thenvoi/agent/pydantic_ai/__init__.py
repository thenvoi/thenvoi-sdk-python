"""
Pydantic AI adapter for Thenvoi platform.

Usage:
    from thenvoi.agent.pydantic_ai import PydanticAIAdapter

    adapter = PydanticAIAdapter(
        model="openai:gpt-4o",
        agent_id="...",
        api_key="...",
    )
    await adapter.run()
"""

from .adapter import PydanticAIAdapter, with_pydantic_ai

__all__ = ["PydanticAIAdapter", "with_pydantic_ai"]
