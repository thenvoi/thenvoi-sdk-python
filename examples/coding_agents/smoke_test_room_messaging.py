#!/usr/bin/env python3
"""Smoke test for bridge/coordinator to specialist room messaging."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import yaml

from thenvoi_rest import AsyncRestClient, ChatMessageRequest, ChatRoomRequest
from thenvoi_rest.types import ChatMessageRequestMentionsItem, ParticipantRequest

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate bridge->reviewer room messaging.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "agent_config.yaml",
        help="Path to keyed agent_config.yaml",
    )
    parser.add_argument(
        "--bridge-key",
        default=os.environ.get("BRIDGE_AGENT_KEY", "bridge"),
        help="Coordinator key in agent config",
    )
    parser.add_argument(
        "--reviewer-key",
        default=os.environ.get("REVIEWER_AGENT_KEY", "reviewer"),
        help="Specialist key to mention and await",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="Max seconds to wait for specialist response",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=2,
        help="Polling interval while waiting for a response",
    )
    return parser.parse_args()


def _load_keyed_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def _agent_credentials(config: dict[str, Any], key: str) -> tuple[str, str]:
    section = config.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Missing agent config section: '{key}'")

    agent_id = str(section.get("agent_id", "")).strip()
    api_key = str(section.get("api_key", "")).strip()
    if not agent_id:
        raise ValueError(f"{key}.agent_id is required")
    if not api_key:
        raise ValueError(f"{key}.api_key is required")
    return agent_id, api_key


def _sender_id(message: Any) -> str:
    return str(getattr(message, "sender_id", ""))


def _sender_name(message: Any) -> str:
    sender_name = getattr(message, "sender_name", None)
    if sender_name:
        return str(sender_name)
    return _sender_id(message) or "unknown"


def _truncate(content: str | None, limit: int = 160) -> str:
    if not content:
        return "(empty)"
    return content[:limit]


async def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    if args.poll_interval_seconds <= 0:
        raise ValueError("--poll-interval-seconds must be > 0")

    config = _load_keyed_config(args.config)
    bridge_agent_id, bridge_api_key = _agent_credentials(config, args.bridge_key)
    reviewer_agent_id, _ = _agent_credentials(config, args.reviewer_key)

    base_url = os.environ.get("THENVOI_REST_URL", "https://app.thenvoi.com")
    client = AsyncRestClient(api_key=bridge_api_key, base_url=base_url)

    logger.info("Creating room using bridge key '%s'...", args.bridge_key)
    room_response = await client.agent_api_chats.create_agent_chat(
        chat=ChatRoomRequest()
    )
    room_id = room_response.data.id
    logger.info("Room created: %s", room_id)

    logger.info("Adding specialist '%s' (%s)...", args.reviewer_key, reviewer_agent_id)
    await client.agent_api_participants.add_agent_chat_participant(
        chat_id=room_id,
        participant=ParticipantRequest(participant_id=reviewer_agent_id),
    )

    await asyncio.sleep(3)

    nonce = str(uuid.uuid4())
    content = (
        f"Hello @{args.reviewer_key}. Compose smoke test token={nonce}. "
        "Please acknowledge in one line."
    )
    logger.info("Sending mention from bridge (%s) to reviewer...", bridge_agent_id)
    sent = await client.agent_api_messages.create_agent_chat_message(
        chat_id=room_id,
        message=ChatMessageRequest(
            content=content,
            mentions=[
                ChatMessageRequestMentionsItem(
                    id=reviewer_agent_id,
                    name=args.reviewer_key,
                )
            ],
        ),
    )
    sent_message_id = str(sent.data.id)
    logger.info("Sent message id: %s", sent_message_id)
    logger.info("Token: %s", nonce)

    deadline = asyncio.get_running_loop().time() + args.timeout_seconds
    reviewer_replied = False

    while asyncio.get_running_loop().time() < deadline:
        messages_response = await client.agent_api_messages.list_agent_messages(
            chat_id=room_id
        )
        for message in messages_response.data:
            message_id = str(getattr(message, "id", ""))
            if message_id == sent_message_id:
                continue
            if _sender_id(message) == reviewer_agent_id:
                reviewer_replied = True
                break
        if reviewer_replied:
            break
        await asyncio.sleep(args.poll_interval_seconds)

    logger.info("")
    logger.info("Messages in room %s:", room_id)
    final_messages = await client.agent_api_messages.list_agent_messages(
        chat_id=room_id
    )
    for message in final_messages.data:
        message_type = str(getattr(message, "message_type", "unknown"))
        logger.info(
            "[%s] %s: %s",
            message_type,
            _sender_name(message),
            _truncate(getattr(message, "content", None)),
        )

    if not reviewer_replied:
        raise RuntimeError(
            "Smoke test failed: specialist did not respond before timeout."
        )

    logger.info("")
    logger.info("Smoke test passed: reviewer response detected.")


if __name__ == "__main__":
    asyncio.run(main())
