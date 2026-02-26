"""Starlette server for the 20 Questions Arena UI.

Run from the repo root:
    uv run python examples/20-questions-arena/ui/server.py

Requires dev dependencies (uv sync --extra dev).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from thenvoi_rest import AsyncRestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

# Allow importing manager.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from manager import (  # noqa: E402
    DEFAULT_WS_URL,
    GUESSER_REGISTRY,
    MODEL_OPTIONS,
    THINKER_AGENT_ID,
    GameManager,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("20-questions-arena.ui")

STATIC_DIR = Path(__file__).parent / "static"

DEFAULT_REST_URL = os.getenv("THENVOI_REST_URL", "https://app.thenvoi.com")
DEFAULT_WS = os.getenv("THENVOI_WS_URL", DEFAULT_WS_URL)
manager = GameManager(ws_url=DEFAULT_WS)


# ── Route handlers ──────────────────────────────────────────────────────────


async def index(request: Request) -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


async def get_config(request: Request) -> JSONResponse:
    return JSONResponse({
        "thinker_agent_id": THINKER_AGENT_ID,
        "thinker_default_model": "claude-sonnet-4-5-20250929",
        "guessers": GUESSER_REGISTRY,
        "models": MODEL_OPTIONS,
        "rest_url": DEFAULT_REST_URL,
    })


async def start_game(request: Request) -> JSONResponse:
    body = await request.json()
    user_api_key = body.get("user_api_key", "")
    rest_url = body.get("rest_url", DEFAULT_REST_URL)
    thinker_model = body.get("thinker_model", "")
    guesser_selections = body.get("guessers", [])

    if not user_api_key:
        return JSONResponse({"error": "API key is required"}, status_code=400)
    if not guesser_selections:
        return JSONResponse(
            {"error": "Select at least one guesser"}, status_code=400
        )

    try:
        game = await manager.create_game(
            user_api_key=user_api_key,
            rest_url=rest_url,
            thinker_model=thinker_model,
            guesser_selections=guesser_selections,
        )
    except Exception as exc:
        logger.exception("Failed to start game")
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse({
        "game_id": game.game_id,
        "chat_id": game.chat_id,
        "guessers": game.guesser_configs,
        "thinker_model": game.thinker_model,
    })


async def game_events(request: Request) -> StreamingResponse:
    """SSE endpoint — streams new messages for a running game."""
    game_id = request.path_params["game_id"]
    logger.info("SSE connected for game %s", game_id)

    async def generate():
        # Send initial heartbeat so the browser knows the connection is alive
        yield f"event: heartbeat\ndata: {json.dumps({'game_id': game_id})}\n\n"
        while True:
            if await request.is_disconnected():
                logger.info("SSE client disconnected for %s", game_id)
                break
            game = manager._games.get(game_id)
            if not game or not game.is_active:
                yield f"event: game_ended\ndata: {json.dumps({'reason': 'stopped'})}\n\n"
                break
            try:
                messages = await manager.poll_messages(game_id)
                for msg in messages:
                    logger.info(
                        "SSE >> %s: [%s] %s",
                        game_id, msg.get("sender_name"), msg.get("content", "")[:60],
                    )
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"
            except Exception:
                logger.exception("SSE poll error")
            # Heartbeat every cycle to keep connection alive
            yield ": heartbeat\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def get_messages(request: Request) -> JSONResponse:
    """Fresh API call each time -- used by frontend REST polling."""
    game_id = request.path_params["game_id"]
    game = manager._games.get(game_id)
    if not game:
        return JSONResponse(
            {"messages": [], "error": "Game not found"}, status_code=404
        )
    msgs = await manager.fetch_messages(game_id)
    logger.debug("GET messages %s: %d messages returned", game_id, len(msgs))
    return JSONResponse({"messages": msgs, "game_active": game.is_active})


async def debug_poll(request: Request) -> JSONResponse:
    """One-shot debug — hit /api/debug/poll to inspect raw REST response."""
    import httpx

    game = manager.current_game
    if not game:
        return JSONResponse({"error": "no active game"}, status_code=404)

    base = game.rest_url.rstrip("/")
    url = f"{base}/api/v1/me/chats/{game.chat_id}/messages"
    headers = {"X-API-Key": game.user_api_key}

    async with httpx.AsyncClient() as http:
        raw = await http.get(url, headers=headers, params={"page_size": 10})

    # Also try Fern client for comparison
    fern_client = AsyncRestClient(
        api_key=game.user_api_key, base_url=game.rest_url
    )
    try:
        fern_resp = await fern_client.human_api_messages.list_my_chat_messages(
            game.chat_id, page_size=10
        )
        fern_count = len(fern_resp.data) if fern_resp.data else 0
        fern_error = None
    except Exception as exc:
        fern_count = 0
        fern_error = f"{type(exc).__name__}: {exc}"

    # Parse raw body for message count
    raw_data_count = 0
    try:
        raw_json = raw.json()
        raw_data_count = len(raw_json.get("data", []))
    except Exception:
        pass

    return JSONResponse({
        "game_id": game.game_id,
        "chat_id": game.chat_id,
        "rest_url": game.rest_url,
        "api_url": url,
        "api_key_prefix": game.user_api_key[:8] + "..." if game.user_api_key else "EMPTY",
        "seen_count": len(game.seen_message_ids),
        "all_messages_count": len(game.all_messages),
        "raw_status": raw.status_code,
        "raw_data_count": raw_data_count,
        "raw_body": raw.text[:3000],
        "fern_count": fern_count,
        "fern_error": fern_error,
    })


async def analyze_game(request: Request) -> JSONResponse:
    """Call an LLM to analyze the completed game transcript."""
    game_id = request.path_params["game_id"]
    game = manager._games.get(game_id)
    if not game:
        return JSONResponse({"error": "Game not found"}, status_code=404)

    messages = await manager.get_all_messages(game_id)
    if not messages:
        return JSONResponse({"error": "No messages to analyze"}, status_code=400)

    # Build a lookup from agent_id -> display name for mention resolution
    agent_names: dict[str, str] = {THINKER_AGENT_ID: "Thinker"}
    for gc in game.guesser_configs:
        agent_names[gc["agent_id"]] = gc.get("name") or gc["label"]

    def resolve_mentions(text: str) -> str:
        """Replace @[[uuid]] mention tags with actual agent names."""
        def _replace(m: re.Match) -> str:
            uuid = m.group(1)
            name = agent_names.get(uuid)
            return f"@{name}" if name else "@agent"
        return re.sub(r"@\[\[([\w-]+)\]\]", _replace, text)

    # Build transcript from visible messages only
    transcript_lines: list[str] = []
    for msg in messages:
        msg_type = (msg.get("message_type") or "").lower()
        if msg_type in ("tool_call", "tool_result", "thought"):
            continue
        sender = msg.get("sender_name") or "Unknown"
        content = resolve_mentions((msg.get("content") or "").strip())
        if content:
            transcript_lines.append(f"[{sender}]: {content}")

    if not transcript_lines:
        return JSONResponse({"error": "No visible messages to analyze"}, status_code=400)

    guesser_info = "\n".join(
        f"  - {gc.get('name') or gc['label']} (model: {gc['model']})"
        for gc in game.guesser_configs
    )
    prompt = (
        'You are analyzing a completed game of "20 Questions" between AI agents.\n\n'
        f"Game Setup:\n"
        f"- Game Master (Thinker): model {game.thinker_model}\n"
        f"- Guessers:\n{guesser_info}\n\n"
        f"Full Game Transcript:\n"
        + "\n".join(transcript_lines)
        + "\n\n"
        "Provide a concise, insightful analysis covering:\n"
        "1. The secret word and whether the Thinker played fairly "
        "(gave accurate yes/no answers)\n"
        "2. Each guesser's strategy — what approach did they take, "
        "how effective was it, did they guess correctly?\n"
        "3. Overall verdict — who played best and why\n\n"
        "Keep it conversational and brief (3-5 short paragraphs). "
        "Use agent names directly."
    )

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    analysis: str | None = None

    if anthropic_key:
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=anthropic_key)
            resp = await client.messages.create(
                model="claude-opus-4-6",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            analysis = resp.content[0].text
        except Exception as exc:
            logger.warning("Anthropic analysis failed: %s", exc)

    if analysis is None and openai_key:
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=openai_key)
            resp = await client.chat.completions.create(
                model="gpt-5.2",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            analysis = resp.choices[0].message.content
        except Exception as exc:
            logger.warning("OpenAI analysis failed: %s", exc)

    if analysis is None:
        return JSONResponse(
            {"error": "No LLM API key available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."},
            status_code=500,
        )

    return JSONResponse({"analysis": analysis})


async def stop_game(request: Request) -> JSONResponse:
    game_id = request.path_params["game_id"]
    await manager.stop_game(game_id)
    return JSONResponse({"status": "stopped"})


async def current_game(request: Request) -> JSONResponse:
    game = manager.current_game
    if not game:
        return JSONResponse({"active": False})
    return JSONResponse({
        "active": True,
        "game_id": game.game_id,
        "guessers": game.guesser_configs,
        "thinker_model": game.thinker_model,
    })


# ── App setup ───────────────────────────────────────────────────────────────


async def on_shutdown() -> None:
    await manager.stop_all()


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/config", get_config),
        Route("/api/game/start", start_game, methods=["POST"]),
        Route("/api/game/current", current_game),
        Route("/api/game/{game_id}/events", game_events),
        Route("/api/game/{game_id}/messages", get_messages),
        Route("/api/game/{game_id}/analyze", analyze_game, methods=["POST"]),
        Route("/api/game/{game_id}/stop", stop_game, methods=["POST"]),
        Route("/api/debug/poll", debug_poll),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ],
    on_shutdown=[on_shutdown],
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8420"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
