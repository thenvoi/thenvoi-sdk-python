"""Tests for examples/coding_agents/create_agents.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "examples_coding_agents_create_agents_test",
        Path("examples/coding_agents/create_agents.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load examples/coding_agents/create_agents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_main_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.delenv("THENVOI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="THENVOI_API_KEY environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_main_registers_agents_and_writes_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    monkeypatch.setenv("THENVOI_API_KEY", "user-key")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "create_agents.py"))

    register_mock = AsyncMock(
        side_effect=[
            SimpleNamespace(
                data=SimpleNamespace(
                    agent=SimpleNamespace(name="Planner", id="agent-planner"),
                    credentials=SimpleNamespace(api_key="planner-key"),
                )
            ),
            SimpleNamespace(
                data=SimpleNamespace(
                    agent=SimpleNamespace(name="Reviewer", id="agent-reviewer"),
                    credentials=SimpleNamespace(api_key="reviewer-key"),
                )
            ),
        ]
    )

    class _FakeAsyncRestClient:
        def __init__(self, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.human_api_agents = SimpleNamespace(register_my_agent=register_mock)

    class _FakeAgentRegisterRequest:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    thenvoi_rest_stub = ModuleType("thenvoi_rest")
    thenvoi_rest_stub.AsyncRestClient = _FakeAsyncRestClient
    thenvoi_rest_types_stub = ModuleType("thenvoi_rest.types")
    thenvoi_rest_types_stub.AgentRegisterRequest = _FakeAgentRegisterRequest

    monkeypatch.setitem(sys.modules, "thenvoi_rest", thenvoi_rest_stub)
    monkeypatch.setitem(sys.modules, "thenvoi_rest.types", thenvoi_rest_types_stub)

    await module.main()

    assert register_mock.await_count == 2
    planner_request = register_mock.await_args_list[0].kwargs["agent"]
    reviewer_request = register_mock.await_args_list[1].kwargs["agent"]
    assert planner_request.name == "Planner"
    assert reviewer_request.name == "Reviewer"

    planner_config = yaml.safe_load((tmp_path / "planner.yaml").read_text(encoding="utf-8"))
    reviewer_config = yaml.safe_load((tmp_path / "reviewer.yaml").read_text(encoding="utf-8"))
    assert planner_config["agent_id"] == "agent-planner"
    assert planner_config["api_key"] == "planner-key"
    assert planner_config["role"] == "planner"
    assert reviewer_config["agent_id"] == "agent-reviewer"
    assert reviewer_config["api_key"] == "reviewer-key"
    assert reviewer_config["role"] == "reviewer"

    cleanup_ids = (tmp_path / ".agent_ids.txt").read_text(encoding="utf-8").splitlines()
    assert cleanup_ids == ["agent-planner", "agent-reviewer"]
