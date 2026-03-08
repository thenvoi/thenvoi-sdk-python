"""Quick script to capture the raw room_added WebSocket payload.

Uses the user API key to register a temporary agent, then connects via
WebSocket and creates a chat to capture the room_added event.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx
from phoenix_channels_python_client import (
    PHXChannelsClient,
    PhoenixChannelsProtocolVersion,
)
from phoenix_channels_python_client.phx_messages import PHXMessage

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_URL = os.environ["THENVOI_BASE_URL"]
WS_URL = os.environ["THENVOI_WS_URL"]
USER_API_KEY = os.environ["THENVOI_API_KEY_USER"]


async def main() -> None:
    # 1. Register a temporary external agent via user key
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {USER_API_KEY}"},
    ) as http:
        # First try listing existing agents
        resp = await http.get("/api/v1/me/agents")
        resp.raise_for_status()
        agents = resp.json()["data"]
        logger.info("Found %d agent(s):", len(agents))
        for a in agents:
            logger.info(
                "  - %s (ID: %s, type: %s)",
                a.get("name"),
                a["id"],
                a.get("type"),
            )

        # Register a new temp agent
        resp = await http.post(
            "/api/v1/me/agents/register",
            json={
                "agent": {
                    "name": "schema-test-temp",
                    "description": "Temporary agent for schema verification",
                }
            },
        )
        resp.raise_for_status()
        reg_data = resp.json()["data"]
        agent_id = reg_data["id"]
        agent_api_key = reg_data.get("api_key") or reg_data.get("token")
        logger.info("\nRegistered temp agent: %s", agent_id)
        logger.info("Response keys: %s", sorted(reg_data.keys()))

        if not agent_api_key:
            logger.info(
                "Full registration response: %s", json.dumps(reg_data, indent=2)
            )
            logger.info("Could not find API key in response. Cleaning up...")
            await http.delete(f"/api/v1/me/agents/{agent_id}")
            return

    # 2. Verify agent key works
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {agent_api_key}"},
    ) as http:
        resp = await http.get("/api/v1/agent/me")
        resp.raise_for_status()
        logger.info("Agent key verified OK")

    # 3. Connect WebSocket and capture room_added
    raw_payloads: list[dict] = []
    got_event = asyncio.Event()

    async def on_message(message: PHXMessage) -> None:
        if message.event == "room_added":
            logger.info("\n=== RAW room_added PAYLOAD ===")
            logger.info(json.dumps(message.payload, indent=2, default=str))
            logger.info("=== END ===\n")
            raw_payloads.append(message.payload)
            got_event.set()

    ws = PHXChannelsClient(
        WS_URL, agent_api_key, protocol_version=PhoenixChannelsProtocolVersion.V2
    )
    ws.channel_socket_url += f"&agent_id={agent_id}"

    async with ws:
        await ws.subscribe_to_topic(f"agent_rooms:{agent_id}", on_message)
        logger.info("Subscribed, creating chat room...")
        await asyncio.sleep(0.3)

        async with httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {agent_api_key}"},
        ) as http:
            resp = await http.post("/api/v1/agent/chats", json={"chat": {}})
            resp.raise_for_status()
            logger.info("Created chat: %s", resp.json()["data"]["id"])

        try:
            await asyncio.wait_for(got_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.info("Timeout — no room_added event received")

    if raw_payloads:
        logger.info("Field summary:")
        for key, val in sorted(raw_payloads[0].items()):
            logger.info("  %-20s = %s", key, repr(val))
    else:
        logger.info("No room_added payloads captured.")

    # 4. Cleanup: delete temp agent
    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {USER_API_KEY}"},
    ) as http:
        await http.delete(f"/api/v1/me/agents/{agent_id}")
        logger.info("\nCleaned up temp agent %s", agent_id)


if __name__ == "__main__":
    asyncio.run(main())
