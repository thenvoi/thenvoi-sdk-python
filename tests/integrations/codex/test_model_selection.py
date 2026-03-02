"""Tests for Codex model selection fallback helpers."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from thenvoi.integrations.codex import CodexJsonRpcError
from thenvoi.integrations.codex.model_selection import (
    find_fallback_model,
    is_model_unavailable_error,
    start_turn_with_model_fallback,
    visible_model_ids,
)


def test_visible_model_ids_filters_hidden_and_invalid_entries() -> None:
    payload = {
        "data": [
            {"id": "gpt-5.3-codex", "hidden": False},
            {"id": "gpt-5.2", "hidden": True},
            {"id": "", "hidden": False},
            {"name": "invalid"},
        ]
    }
    assert visible_model_ids(payload) == ["gpt-5.3-codex"]


def test_is_model_unavailable_error_detects_known_phrases() -> None:
    err = CodexJsonRpcError(code=-32000, message="Model not found", data=None)
    assert is_model_unavailable_error(err) is True
    generic = CodexJsonRpcError(code=-32000, message="transport issue", data=None)
    assert is_model_unavailable_error(generic) is False


@pytest.mark.asyncio
async def test_find_fallback_model_uses_fallback_when_model_list_fails() -> None:
    async def _request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        del method, params
        raise RuntimeError("boom")

    fallback = await find_fallback_model(
        request=_request,
        exclude="gpt-5.3-codex",
        fallback_models=("gpt-5.3-codex", "gpt-5.2"),
        logger=logging.getLogger(__name__),
    )
    assert fallback == "gpt-5.2"


@pytest.mark.asyncio
async def test_start_turn_with_model_fallback_retries_with_selected_model() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        calls.append((method, dict(params)))
        if len(calls) == 1:
            raise CodexJsonRpcError(
                code=-32000,
                message="Model not available",
                data=None,
            )
        if method == "model/list":
            return {"data": [{"id": "gpt-5.2", "hidden": False}]}
        return {"turn": {"id": "turn-1"}}

    selected: list[str] = []
    result = await start_turn_with_model_fallback(
        request=_request,
        params={"threadId": "t1", "model": "gpt-5.3-codex"},
        model_explicitly_set=False,
        fallback_models=("gpt-5.2",),
        update_selected_model=selected.append,
        logger=logging.getLogger(__name__),
    )

    assert result["turn"]["id"] == "turn-1"
    assert selected == ["gpt-5.2"]
    assert any(method == "model/list" for method, _ in calls)

