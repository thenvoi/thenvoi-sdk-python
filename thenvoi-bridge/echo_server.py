"""Minimal echo server that mimics a LangServe /invoke endpoint.

Run with:
    pip install fastapi uvicorn
    python echo_server.py
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/invoke")
async def invoke(body: dict) -> dict:
    """Echo the input back, matching LangServe response format."""
    prompt = body.get("input", "")
    metadata = body.get("metadata", {})
    config = body.get("config", {})

    logger.info("Received prompt: %s", prompt)
    logger.info("Metadata: %s", metadata)
    logger.info("Config: %s", config)

    return {"output": f"Echo: {prompt}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
