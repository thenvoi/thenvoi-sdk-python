"""Game orchestration for the 20 Questions Arena UI.

Manages agent subprocesses, room creation, and message polling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thenvoi.client.streaming.client import WebSocketClient
from thenvoi_rest import AsyncRestClient, ChatMessageRequest, ParticipantRequest
from thenvoi_rest.human_api_chats.types.create_my_chat_room_request_chat import (
    CreateMyChatRoomRequestChat,
)
from thenvoi_rest.types import ChatMessageRequestMentionsItem as Mention

DEFAULT_WS_URL = "wss://app.thenvoi.com/api/v1/socket/websocket"

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

THINKER_AGENT_ID = "3f39b82c-b6ab-4300-bfd6-f3ff5792e348"

GUESSER_REGISTRY: list[dict[str, str]] = [
    {
        "key": "arena_guesser",
        "agent_id": "f00d355b-df9b-4659-aa8e-9f5abad4fb5d",
        "default_model": "gpt-5-nano",
        "label": "Guesser 1",
        "color": "#3B82F6",
    },
    {
        "key": "arena_guesser_2",
        "agent_id": "8b6510bc-b678-4ee4-8f71-d84de19b43b8",
        "default_model": "gpt-5.2-pro",
        "label": "Guesser 2",
        "color": "#10B981",
    },
    {
        "key": "arena_guesser_3",
        "agent_id": "c10a872d-075a-488b-8611-8d9eda3a0b94",
        "default_model": "claude-haiku-4-5-20251001",
        "label": "Guesser 3",
        "color": "#F59E0B",
    },
    {
        "key": "arena_guesser_4",
        "agent_id": "40fb98b6-7a6d-4ed6-988b-1246e6015a8d",
        "default_model": "claude-opus-4-6",
        "label": "Guesser 4",
        "color": "#F43F5E",
    },
]

MODEL_OPTIONS: list[dict[str, str]] = [
    {"id": "gpt-5.2", "name": "GPT 5.2", "provider": "OpenAI"},
    {"id": "gpt-5.2-pro", "name": "GPT 5.2 Pro", "provider": "OpenAI"},
    {"id": "gpt-5-mini", "name": "GPT 5 Mini", "provider": "OpenAI"},
    {"id": "gpt-5-nano", "name": "GPT 5 Nano", "provider": "OpenAI"},
    {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "provider": "Anthropic"},
    {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "provider": "Anthropic"},
    {
        "id": "claude-haiku-4-5-20251001",
        "name": "Claude Haiku 4.5",
        "provider": "Anthropic",
    },
    {
        "id": "claude-sonnet-4-5-20250929",
        "name": "Claude Sonnet 4.5",
        "provider": "Anthropic",
    },
]

AGENT_STARTUP_DELAY = 8  # seconds to wait for agents to connect


@dataclass
class ActiveGame:
    game_id: str
    chat_id: str
    thinker_model: str
    guesser_configs: list[dict[str, str]]
    user_api_key: str
    rest_url: str
    ws_url: str = DEFAULT_WS_URL
    ws_client: WebSocketClient | None = None
    processes: list[asyncio.subprocess.Process] = field(default_factory=list)
    seen_message_ids: set[str] = field(default_factory=set)
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    is_active: bool = True


class GameManager:
    """Manages 20 Questions Arena game lifecycle: agent processes, room creation, message polling."""

    def __init__(self, ws_url: str = DEFAULT_WS_URL) -> None:
        self._games: dict[str, ActiveGame] = {}
        self._counter = 0
        self._ws_url = ws_url

    @property
    def current_game(self) -> ActiveGame | None:
        for g in reversed(list(self._games.values())):
            if g.is_active:
                return g
        return None

    async def create_game(
        self,
        user_api_key: str,
        rest_url: str,
        thinker_model: str,
        guesser_selections: list[dict[str, str]],
    ) -> ActiveGame:
        """Start agent processes, create a room, and kick off a new game."""
        if self.current_game:
            await self.stop_game(self.current_game.game_id)

        self._counter += 1
        game_id = f"game_{self._counter}"

        registry_map = {g["key"]: g for g in GUESSER_REGISTRY}
        guesser_configs = []
        for sel in guesser_selections:
            reg = registry_map[sel["key"]]
            guesser_configs.append(
                {
                    "key": reg["key"],
                    "agent_id": reg["agent_id"],
                    "model": sel.get("model", reg["default_model"]),
                    "label": reg["label"],
                    "color": reg["color"],
                }
            )

        # --- Start agent subprocesses ---
        # Use `uv run python <script>` (not `uv run <script>`) so that the
        # project's full venv is used.  Running `uv run <script>` triggers
        # PEP 723 isolated-env resolution which misses langchain-anthropic
        # and other extras not listed in the script's inline metadata.
        uv_bin = shutil.which("uv") or "uv"
        env = {**os.environ}
        processes: list[asyncio.subprocess.Process] = []

        thinker_cmd = [
            uv_bin,
            "run",
            "python",
            str(REPO_ROOT / "examples" / "20-questions-arena" / "thinker_agent.py"),
        ]
        if thinker_model:
            thinker_cmd.extend(["--model", thinker_model])

        logger.info("Starting thinker: %s", " ".join(thinker_cmd))
        thinker_proc = await asyncio.create_subprocess_exec(
            *thinker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env=env,
        )
        processes.append(thinker_proc)

        for gc in guesser_configs:
            guesser_cmd = [
                uv_bin,
                "run",
                "python",
                str(REPO_ROOT / "examples" / "20-questions-arena" / "guesser_agent.py"),
                "--config",
                gc["key"],
                "--model",
                gc["model"],
            ]
            logger.info("Starting guesser: %s", " ".join(guesser_cmd))
            proc = await asyncio.create_subprocess_exec(
                *guesser_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=env,
            )
            processes.append(proc)

        logger.info("Waiting %ds for agents to connect...", AGENT_STARTUP_DELAY)
        await asyncio.sleep(AGENT_STARTUP_DELAY)

        # Verify processes are still alive
        for i, proc in enumerate(processes):
            if proc.returncode is not None:
                label = "Thinker" if i == 0 else guesser_configs[i - 1]["label"]
                stdout = await proc.stdout.read() if proc.stdout else b""
                raise RuntimeError(
                    f"{label} process died (code {proc.returncode}): "
                    f"{stdout.decode(errors='replace')[-500:]}"
                )

        # --- Create room and start game ---
        client = AsyncRestClient(api_key=user_api_key, base_url=rest_url)

        chat_resp = await client.human_api_chats.create_my_chat_room(
            chat=CreateMyChatRoomRequestChat(),
        )
        chat_id = str(chat_resp.data.id)
        logger.info("Created chat room: %s", chat_id)

        await client.human_api_participants.add_my_chat_participant(
            chat_id,
            participant=ParticipantRequest(
                participant_id=THINKER_AGENT_ID, role="member"
            ),
        )

        for gc in guesser_configs:
            await client.human_api_participants.add_my_chat_participant(
                chat_id,
                participant=ParticipantRequest(
                    participant_id=gc["agent_id"], role="member"
                ),
            )

        parts_resp = await client.human_api_participants.list_my_chat_participants(
            chat_id
        )
        agents_map = {str(p.id): p for p in parts_resp.data}

        thinker_p = agents_map.get(THINKER_AGENT_ID)
        thinker_name = thinker_p.name if thinker_p else "Thinker"

        guesser_names: list[str] = []
        for gc in guesser_configs:
            p = agents_map.get(gc["agent_id"])
            if p:
                gc["name"] = p.name
                guesser_names.append(p.name)

        names_list = ", ".join(guesser_names)
        await client.human_api_messages.send_my_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=(
                    f"@{thinker_name} start a new game of 20 questions "
                    f"with {names_list}. "
                    "These guessers are already in the room — do NOT "
                    "look up or invite any other guessers."
                ),
                mentions=[Mention(id=THINKER_AGENT_ID, name=thinker_name)],
            ),
        )
        logger.info("Game started in room %s", chat_id)

        game = ActiveGame(
            game_id=game_id,
            chat_id=chat_id,
            thinker_model=thinker_model,
            guesser_configs=guesser_configs,
            user_api_key=user_api_key,
            rest_url=rest_url,
            ws_url=self._ws_url,
            processes=processes,
        )
        self._games[game_id] = game

        # --- Connect WebSocket to receive ALL messages in real-time ---
        await self._start_ws_listener(game)

        return game

    async def _start_ws_listener(self, game: ActiveGame) -> None:
        """Open a WebSocket as the user and subscribe to the chat room."""
        try:
            ws = WebSocketClient(
                ws_url=game.ws_url,
                api_key=game.user_api_key,
                # No agent_id — connect as the human user
            )
            await ws.__aenter__()
            game.ws_client = ws

            def _serialize(payload: dict[str, Any]) -> dict[str, Any]:
                """Convert a raw WS payload to our serialized format."""
                mid = str(payload.get("id", ""))
                serialized: dict[str, Any] = {
                    "id": mid,
                    "content": payload.get("content", ""),
                    "sender_name": payload.get("sender_name") or "Unknown",
                    "sender_id": str(payload.get("sender_id", "")),
                    "sender_type": str(payload.get("sender_type", "unknown")),
                    "message_type": str(payload.get("message_type", "text")),
                    "inserted_at": str(payload.get("inserted_at", "")),
                }
                mentions = payload.get("metadata", {})
                if isinstance(mentions, dict):
                    mentions = mentions.get("mentions")
                    if mentions and isinstance(mentions, list):
                        serialized["mentions"] = [
                            {"id": str(m.get("id", "")), "name": m.get("name", "")}
                            for m in mentions
                            if isinstance(m, dict)
                        ]
                return serialized

            def _ingest(payload: dict[str, Any]) -> None:
                """Ingest a new message/event payload into the game."""
                mid = str(payload.get("id", ""))
                if not mid or mid in game.seen_message_ids:
                    return
                game.seen_message_ids.add(mid)
                serialized = _serialize(payload)
                game.all_messages.append(serialized)
                logger.info(
                    "WS %s [%s]: [%s] %s",
                    game.game_id,
                    serialized["message_type"],
                    serialized["sender_name"],
                    serialized["content"][:80],
                )

            def _ingest_update(payload: dict[str, Any]) -> None:
                """Update an already-seen message (e.g. text→thought type change)."""
                mid = str(payload.get("id", ""))
                if not mid:
                    return
                serialized = _serialize(payload)
                # Find and replace the existing entry
                for i, msg in enumerate(game.all_messages):
                    if msg["id"] == mid:
                        game.all_messages[i] = serialized
                        logger.info(
                            "WS %s [%s] UPDATED: [%s] %s",
                            game.game_id,
                            serialized["message_type"],
                            serialized["sender_name"],
                            serialized["content"][:80],
                        )
                        return
                # Not found in list (shouldn't happen), append it
                game.all_messages.append(serialized)

            # Subscribe directly to the topic so we handle ALL event types,
            # not just message_created (thoughts may arrive as message_updated
            # or event_created).
            topic = f"chat_room:{game.chat_id}"

            async def raw_handler(message: Any) -> None:
                event = getattr(message, "event", "")
                payload = getattr(message, "payload", {}) or {}
                if event == "message_created":
                    _ingest(payload)
                elif event == "message_updated":
                    mtype = payload.get("message_type", "?")
                    mid = str(payload.get("id", "?"))
                    is_new = mid not in game.seen_message_ids
                    logger.info(
                        "WS %s message_updated: id=%s type=%s new=%s content=%s",
                        game.game_id,
                        mid[:8],
                        mtype,
                        is_new,
                        str(payload.get("content", ""))[:60],
                    )
                    if is_new:
                        _ingest(payload)
                    else:
                        # Update in place — catches type changes (text→thought)
                        _ingest_update(payload)
                elif event == "event_created":
                    logger.info(
                        "WS %s event_created: type=%s keys=%s content=%s",
                        game.game_id,
                        payload.get("message_type", "?"),
                        list(payload.keys()),
                        str(payload.get("content", ""))[:80],
                    )
                    _ingest(payload)
                elif event in ("phx_reply", "phx_error"):
                    pass
                else:
                    logger.info(
                        "WS %s: event=%s keys=%s",
                        game.game_id,
                        event,
                        list(payload.keys()),
                    )

            await ws.client.subscribe_to_topic(topic, raw_handler)
            logger.info(
                "WebSocket listener connected for game %s (room %s)",
                game.game_id,
                game.chat_id,
            )
        except Exception:
            logger.exception(
                "Failed to start WebSocket listener for game %s", game.game_id
            )

    async def poll_messages(self, game_id: str) -> list[dict[str, Any]]:
        """Return all messages accumulated via the WebSocket listener.

        The WebSocket callback appends messages to game.all_messages
        in real-time, so this just returns the full list.
        """
        game = self._games.get(game_id)
        if not game:
            return []
        return list(game.all_messages)

    async def fetch_messages(self, game_id: str) -> list[dict[str, Any]]:
        """Return all messages accumulated via the WebSocket listener.

        Same as poll_messages — both read from the in-memory list
        populated by the WebSocket `message_created` callback.
        """
        game = self._games.get(game_id)
        if not game:
            return []
        return list(game.all_messages)

    async def get_all_messages(self, game_id: str) -> list[dict[str, Any]]:
        """Return all messages seen so far for a game."""
        game = self._games.get(game_id)
        if not game:
            return []
        return list(game.all_messages)

    async def stop_game(self, game_id: str) -> None:
        """Terminate all agent processes and disconnect WebSocket for a game."""
        game = self._games.get(game_id)
        if not game:
            return
        game.is_active = False

        # Close WebSocket listener
        if game.ws_client is not None:
            try:
                await game.ws_client.__aexit__(None, None, None)
            except Exception:
                logger.warning("Error closing WebSocket for game %s", game_id)
            game.ws_client = None

        for proc in game.processes:
            if proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except (ProcessLookupError, asyncio.TimeoutError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
        logger.info("Game %s stopped", game_id)

    async def stop_all(self) -> None:
        """Stop every active game."""
        for gid in list(self._games):
            await self.stop_game(gid)
