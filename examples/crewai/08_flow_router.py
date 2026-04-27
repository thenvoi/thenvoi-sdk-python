# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[crewai]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""CrewAI Flow router example.

Demonstrates ``CrewAIFlowAdapter`` for room-native multi-turn orchestration
with parallel join and sequential composition. Use ``CrewAIAdapter`` (see
``01_basic_agent.py``) for normal single-agent CrewAI turns; this adapter
is for room routers that need to delegate to multiple peers and wait for
all replies before synthesizing.

The example uses a toy Flow factory whose terminal decision rotates
between direct_response, parallel delegate, and synthesize so you can see
each shape land as task events without paying for an LLM.

Run with:
    uv run examples/crewai/08_flow_router.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from setup_logging import setup_logging  # noqa: E402

from thenvoi import Agent  # noqa: E402
from thenvoi.adapters import CrewAIFlowAdapter  # noqa: E402
from thenvoi.config import load_agent_config  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


class ToyRouterFlow:
    """Minimal stand-in for a real ``crewai.flow.flow.Flow``.

    A real Flow defines ``@start`` / ``@listen`` / ``@router`` methods and
    exposes ``kickoff_async``. Here we just return a hard-coded decision
    dict so the example is runnable without an LLM.
    """

    def __init__(self) -> None:
        self._turn = 0

    async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        self._turn += 1
        if self._turn == 1:
            # Parent request → fan out to two peers in parallel.
            return {
                "decision": "delegate",
                "delegations": [
                    {
                        "delegation_id": "data-fetcher",
                        "target": "data-fetcher",
                        "content": "fetch the latest data",
                        "mentions": ["@example/data-fetcher"],
                    },
                    {
                        "delegation_id": "ticket-bot",
                        "target": "ticket-bot",
                        "content": "list any open tickets",
                        "mentions": ["@example/ticket-bot"],
                    },
                ],
            }
        if self._turn == 2:
            # Both peers replied → synthesize. Sequential chain
            # (data-fetcher -> presenter) blocks finalization until the
            # presenter has been delegated to.
            return {
                "decision": "delegate",
                "delegations": [
                    {
                        "delegation_id": "presenter",
                        "target": "presenter",
                        "content": "format the combined result for the user",
                        "mentions": ["@example/presenter"],
                    },
                ],
            }
        # Final turn: presenter replied; produce the final synthesis.
        return {
            "decision": "synthesize",
            "content": "Here is the combined report.",
            "mentions": ["@example/user"],
        }


def flow_factory() -> ToyRouterFlow:
    """Constructor passed to the adapter; called once per inbound message."""
    return ToyRouterFlow()


async def main() -> None:
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")
    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    agent_config = load_agent_config("crewai_flow_router")

    adapter = CrewAIFlowAdapter(
        flow_factory=flow_factory,
        join_policy="all",
        sequential_chains={"data-fetcher": "presenter"},
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_config["agent_id"],
        api_key=agent_config["api_key"],
        ws_url=ws_url,
        rest_url=rest_url,
    )
    logger.info("CrewAI Flow router agent starting")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
