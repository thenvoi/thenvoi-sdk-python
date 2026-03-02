"""Direct module tests for thenvoi.adapters.langgraph."""

from __future__ import annotations

import pytest

from thenvoi.adapters.langgraph import LangGraphAdapter


def test_langgraph_adapter_requires_graph_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="Must provide either llm \\(simple pattern\\) or graph_factory/graph \\(advanced pattern\\)",
    ):
        LangGraphAdapter()


@pytest.mark.asyncio
async def test_langgraph_adapter_cleanup_discards_bootstrap_room() -> None:
    adapter = LangGraphAdapter(graph=object())
    adapter._bootstrapped_rooms.add("room-1")

    await adapter.on_cleanup("room-1")

    assert "room-1" not in adapter._bootstrapped_rooms
