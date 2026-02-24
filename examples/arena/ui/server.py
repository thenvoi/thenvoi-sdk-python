"""Starlette server for the 20 Questions Arena UI.

Run from the repo root:
    uv run python examples/arena/ui/server.py

Requires dev dependencies (uv sync --extra dev).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

# Allow importing manager.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from manager import (  # noqa: E402
    GUESSER_REGISTRY,
    MODEL_OPTIONS,
    THINKER_AGENT_ID,
    GameManager,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("arena.ui")

STATIC_DIR = Path(__file__).parent / "static"
manager = GameManager()

DEFAULT_REST_URL = os.getenv("THENVOI_REST_URL", "https://app.thenvoi.com")


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

    async def generate():
        while True:
            if await request.is_disconnected():
                break
            game = manager._games.get(game_id)
            if not game or not game.is_active:
                yield f"event: game_ended\ndata: {json.dumps({'reason': 'stopped'})}\n\n"
                break
            try:
                messages = await manager.poll_messages(game_id)
                for msg in messages:
                    yield f"event: message\ndata: {json.dumps(msg)}\n\n"
            except Exception:
                logger.exception("SSE poll error")
            await asyncio.sleep(2)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def get_messages(request: Request) -> JSONResponse:
    game_id = request.path_params["game_id"]
    msgs = await manager.get_all_messages(game_id)
    return JSONResponse({"messages": msgs})


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
        Route("/api/game/{game_id}/stop", stop_game, methods=["POST"]),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ],
    on_shutdown=[on_shutdown],
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8420"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
