# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[codex]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Basic Codex adapter agent example.

Runs a Thenvoi agent backed by Codex app-server.

Prerequisites:
1. OAuth login:
   codex login
2. For stdio mode (default), no extra process is needed.
3. For ws mode, start app-server separately:
   codex app-server --listen ws://127.0.0.1:8765

Run:
    uv run examples/codex/01_basic_agent.py

Optional env overrides:
    AGENT_KEY=darter
    CODEX_TRANSPORT=stdio|ws
    CODEX_WS_URL=ws://127.0.0.1:8765
    CODEX_ROLE=coding|planner|reviewer
    CODEX_MODEL=gpt-5.3-codex
    CODEX_APPROVAL_MODE=manual|auto_accept|auto_decline
    CODEX_TURN_TASK_MARKERS=true|false
"""

from __future__ import annotations

import asyncio
import logging

from examples.codex.basic_agent import main

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    asyncio.run(main())
