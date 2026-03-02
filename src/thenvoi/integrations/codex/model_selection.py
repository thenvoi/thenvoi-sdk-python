"""Codex model-selection and fallback policy helpers."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from thenvoi.integrations.codex import CodexJsonRpcError


async def start_turn_with_model_fallback(
    *,
    request: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    params: dict[str, Any],
    model_explicitly_set: bool,
    fallback_models: tuple[str, ...],
    update_selected_model: Callable[[str], None],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run `turn/start`, retrying with a visible fallback model when needed."""
    try:
        return await request("turn/start", params)
    except CodexJsonRpcError as error:
        if model_explicitly_set or not is_model_unavailable_error(error):
            raise

        original_model = params.get("model")
        logger.warning(
            "Model %s unavailable (code=%s): %s. Querying available models...",
            original_model,
            error.code,
            error.message,
        )
        fallback = await find_fallback_model(
            request=request,
            exclude=original_model,
            fallback_models=fallback_models,
            logger=logger,
        )
        if fallback is None:
            raise

        logger.warning("Falling back from %s to %s", original_model, fallback)
        update_selected_model(fallback)
        params["model"] = fallback
        return await request("turn/start", params)


async def find_fallback_model(
    *,
    request: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    exclude: Any,
    fallback_models: tuple[str, ...],
    logger: logging.Logger,
) -> str | None:
    """Query `model/list` and return the first viable model candidate."""
    try:
        result = await request("model/list", {})
    except Exception:
        logger.warning("model/list failed during fallback lookup", exc_info=True)
        for model_id in fallback_models:
            if model_id != exclude:
                return model_id
        return None

    models = visible_model_ids(result)
    for model_id in models:
        if model_id != exclude and model_id in fallback_models:
            return model_id
    for model_id in models:
        if model_id != exclude:
            return model_id
    for model_id in fallback_models:
        if model_id != exclude:
            return model_id
    return None


def is_model_unavailable_error(error: CodexJsonRpcError) -> bool:
    """Check whether an RPC error indicates model availability/access failure."""
    message = error.message.lower()
    return any(
        phrase in message
        for phrase in (
            "model not found",
            "model not available",
            "is not available",
            "model_not_found",
            "model unavailable",
            "does not have access",
            "no access to model",
        )
    )


def visible_model_ids(result: dict[str, Any]) -> list[str]:
    """Extract non-hidden model IDs from `model/list` response payload."""
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, list):
        return []

    models: list[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        if bool(entry.get("hidden", False)):
            continue
        models.append(model_id)
    return models

