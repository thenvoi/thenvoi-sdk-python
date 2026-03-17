"""Manual test entry point for the AgentCore bridge handler.

Run from the thenvoi-bridge directory:
    uv run python examples/run_agentcore.py

Requires .env.test (or ENV_FILE) with:
    THENVOI_AGENT_ID, THENVOI_API_KEY, AGENT_MAPPING,
    AGENTCORE_RUNTIME_ARN, AWS_DEFAULT_REGION,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

# Add thenvoi-bridge dir to path so bridge_core/handlers are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv(os.environ.get("ENV_FILE", os.path.join(os.path.dirname(__file__), "..", "..", ".env.test")))

from bridge_core.bridge import main  # noqa: E402
from handlers.agentcore import AgentCoreHandler  # noqa: E402

arn = os.environ.get("AGENTCORE_RUNTIME_ARN", "")
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
timeout = float(os.environ.get("AGENTCORE_TIMEOUT", "120"))

if not arn:
    raise ValueError("AGENTCORE_RUNTIME_ARN is required in your .env file")

kwargs: dict[str, Any] = {
    "agent_runtime_arn": arn,
    "region": region,
    "timeout": timeout,
}
if mcp_tool := os.environ.get("AGENTCORE_MCP_TOOL"):
    kwargs["mcp_tool_name"] = mcp_tool

handler = AgentCoreHandler(**kwargs)

asyncio.run(main(handlers={"agentcore": handler}))
