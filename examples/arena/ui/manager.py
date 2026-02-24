"""Game orchestration for the Arena UI.

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

import httpx
from thenvoi_rest import AsyncRestClient, ChatMessageRequest, ParticipantRequest
from thenvoi_rest.human_api_chats.types.create_my_chat_room_request_chat import (
    CreateMyChatRoomRequestChat,
)
from thenvoi_rest.types import ChatMessageRequestMentionsItem as Mention

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
    {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "provider": "Anthropic"},
    {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "provider": "Anthropic"},
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
    processes: list[asyncio.subprocess.Process] = field(default_factory=list)
    seen_message_ids: set[str] = field(default_factory=set)
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    is_active: bool = True


def _get_field(msg: dict[str, Any], snake: str, *alternatives: str) -> Any:
    """Get a field by snake_case name, falling back to camelCase or alternatives."""
    val = msg.get(snake)
    if val is not None:
        return val
    # Try camelCase variant (e.g. sender_name -> senderName)
    parts = snake.split("_")
    camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
    val = msg.get(camel)
    if val is not None:
        return val
    # Try explicit alternatives
    for alt in alternatives:
        val = msg.get(alt)
        if val is not None:
            return val
    return None


def _serialize_message_dict(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw API message dict to a JSON-serializable dict.

    Handles both snake_case and camelCase field names from the API.
    """
    content = _get_field(msg, "content", "body", "text") or ""
    sender_name = _get_field(msg, "sender_name", "senderName", "sender") or "Unknown"
    msg_type = str(
        _get_field(msg, "message_type", "messageType", "type") or "message"
    )

    result: dict[str, Any] = {
        "id": str(_get_field(msg, "id") or ""),
        "content": content,
        "sender_name": sender_name,
        "sender_id": str(_get_field(msg, "sender_id", "senderId") or ""),
        "sender_type": str(
            _get_field(msg, "sender_type", "senderType") or "unknown"
        ),
        "message_type": msg_type,
        "inserted_at": str(_get_field(msg, "inserted_at", "insertedAt") or ""),
    }
    mentions = _get_field(msg, "mentions")
    if mentions and isinstance(mentions, list):
        result["mentions"] = [
            {"id": str(m.get("id", "")), "name": m.get("name", "")}
            for m in mentions
            if isinstance(m, dict)
        ]
    return result


class GameManager:
    """Manages arena game lifecycle: agent processes, room creation, message polling."""

    def __init__(self) -> None:
        self._games: dict[str, ActiveGame] = {}
        self._counter = 0
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        """Reusable httpx client for raw polling."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=15)
        return self._http

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
            guesser_configs.append({
                "key": reg["key"],
                "agent_id": reg["agent_id"],
                "model": sel.get("model", reg["default_model"]),
                "label": reg["label"],
                "color": reg["color"],
            })

        # --- Start agent subprocesses ---
        # Use `uv run python <script>` (not `uv run <script>`) so that the
        # project's full venv is used.  Running `uv run <script>` triggers
        # PEP 723 isolated-env resolution which misses langchain-anthropic
        # and other extras not listed in the script's inline metadata.
        uv_bin = shutil.which("uv") or "uv"
        env = {**os.environ}
        processes: list[asyncio.subprocess.Process] = []

        thinker_cmd = [
            uv_bin, "run", "python",
            str(REPO_ROOT / "examples" / "arena" / "thinker_agent.py"),
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
                uv_bin, "run", "python",
                str(REPO_ROOT / "examples" / "arena" / "guesser_agent.py"),
                "--config", gc["key"],
                "--model", gc["model"],
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

        mentions = [Mention(id=THINKER_AGENT_ID, name=thinker_name)]
        guesser_names: list[str] = []
        for gc in guesser_configs:
            p = agents_map.get(gc["agent_id"])
            if p:
                mentions.append(Mention(id=str(p.id), name=p.name))
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
                mentions=mentions,
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
            processes=processes,
        )
        self._games[game_id] = game
        return game

    async def poll_messages(self, game_id: str) -> list[dict[str, Any]]:
        """Return newly seen messages for a game using raw HTTP.

        Uses httpx directly instead of the Fern client so we get full
        visibility into the HTTP response (status, body) for debugging.
        """
        game = self._games.get(game_id)
        if not game or not game.is_active:
            return []

        base = game.rest_url.rstrip("/")
        url = f"{base}/api/v1/me/chats/{game.chat_id}/messages"
        headers = {"X-API-Key": game.user_api_key}

        try:
            http = await self._get_http()
            resp = await http.get(
                url, headers=headers, params={"page_size": 200}
            )
        except Exception:
            logger.exception("Poll HTTP error for game %s", game_id)
            return []

        if resp.status_code != 200:
            logger.error(
                "Poll %s: HTTP %d — %s",
                game_id, resp.status_code, resp.text[:500],
            )
            return []

        try:
            body = resp.json()
        except Exception:
            logger.error("Poll %s: invalid JSON — %s", game_id, resp.text[:300])
            return []

        # The API may return messages under "data" or at the top level
        messages_raw = body.get("data")
        if messages_raw is None:
            # Fallback: body itself might be a list
            if isinstance(body, list):
                messages_raw = body
            else:
                # Try other common keys
                messages_raw = body.get("messages", body.get("items", []))

        if not isinstance(messages_raw, list):
            logger.error(
                "Poll %s: expected list of messages, got %s. body keys=%s",
                game_id, type(messages_raw).__name__, list(body.keys()) if isinstance(body, dict) else "N/A",
            )
            return []

        messages: list[dict[str, Any]] = messages_raw
        total = len(messages)

        # Detailed debug log on first poll to inspect API response structure
        if len(game.seen_message_ids) == 0:
            sample = messages[0] if messages else {}
            logger.info(
                "Poll %s [FIRST]: %d messages, body keys=%s, "
                "sample msg keys=%s, sample=%s",
                game_id,
                total,
                list(body.keys()) if isinstance(body, dict) else "N/A",
                list(sample.keys()) if isinstance(sample, dict) else "N/A",
                str(sample)[:300],
            )

        # Log on empty or normal
        if total == 0 and len(game.seen_message_ids) < 3:
            logger.warning(
                "Poll %s: 0 messages from API (seen=%d). "
                "HTTP %d, body keys=%s, body[:300]=%s",
                game_id,
                len(game.seen_message_ids),
                resp.status_code,
                list(body.keys()) if isinstance(body, dict) else "N/A",
                resp.text[:300],
            )
        else:
            logger.info(
                "Poll %s: %d total messages, %d already seen",
                game_id, total, len(game.seen_message_ids),
            )

        new_msgs: list[dict[str, Any]] = []
        for msg in messages:
            mid = str(msg.get("id", ""))
            if not mid or mid in game.seen_message_ids:
                continue
            game.seen_message_ids.add(mid)
            serialized = _serialize_message_dict(msg)
            new_msgs.append(serialized)
            game.all_messages.append(serialized)

        if new_msgs:
            logger.info("Poll %s: %d NEW messages", game_id, len(new_msgs))
        new_msgs.sort(key=lambda m: m["inserted_at"])
        return new_msgs

    async def fetch_messages(self, game_id: str) -> list[dict[str, Any]]:
        """Fetch all messages directly from the REST API (fresh call).

        Unlike poll_messages (used by SSE), this makes an independent HTTP
        request each time -- reliable for frontend REST polling.
        """
        game = self._games.get(game_id)
        if not game or not game.is_active:
            return []

        base = game.rest_url.rstrip("/")
        url = f"{base}/api/v1/me/chats/{game.chat_id}/messages"
        headers = {"X-API-Key": game.user_api_key}

        try:
            http = await self._get_http()
            resp = await http.get(url, headers=headers, params={"page_size": 200})
            if resp.status_code != 200:
                logger.error("fetch_messages %s: HTTP %d", game_id, resp.status_code)
                return []
            body = resp.json()

            # Handle multiple possible response shapes
            messages_raw = body.get("data") if isinstance(body, dict) else None
            if messages_raw is None:
                if isinstance(body, list):
                    messages_raw = body
                elif isinstance(body, dict):
                    messages_raw = body.get("messages", body.get("items", []))
                else:
                    messages_raw = []

            if not isinstance(messages_raw, list):
                logger.error(
                    "fetch_messages %s: unexpected type %s",
                    game_id, type(messages_raw).__name__,
                )
                return []

            serialized = [_serialize_message_dict(m) for m in messages_raw]
            serialized.sort(key=lambda m: m["inserted_at"])
            return serialized
        except Exception:
            logger.exception("fetch_messages error for %s", game_id)
            return []

    async def get_all_messages(self, game_id: str) -> list[dict[str, Any]]:
        """Return all messages seen so far for a game."""
        game = self._games.get(game_id)
        if not game:
            return []
        return list(game.all_messages)

    async def stop_game(self, game_id: str) -> None:
        """Terminate all agent processes for a game."""
        game = self._games.get(game_id)
        if not game:
            return
        game.is_active = False
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
        """Stop every active game and close shared resources."""
        for gid in list(self._games):
            await self.stop_game(gid)
        if self._http and not self._http.is_closed:
            await self._http.aclose()
