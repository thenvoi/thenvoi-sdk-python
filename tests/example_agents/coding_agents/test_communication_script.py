"""Tests for examples/coding_agents/test_communication.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "examples_coding_agents_test_communication_test",
        Path("examples/coding_agents/test_communication.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load examples/coding_agents/test_communication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_agent_runtime_uses_runner_config(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    mock_load_runner_config = MagicMock(
        return_value={"agent_id": "agent-123", "api_key": "key-456"}
    )
    monkeypatch.setattr(module, "load_runner_config", mock_load_runner_config)

    runtime = module.load_agent_runtime("planner.yaml")

    mock_load_runner_config.assert_called_once_with(module.SCRIPT_DIR / "planner.yaml", agent_key="agent")
    assert runtime == {"agent_id": "agent-123", "api_key": "key-456"}


@pytest.mark.asyncio
async def test_main_creates_room_adds_participant_and_sends_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    client_instances: list[object] = []

    class _FakeClient:
        def __init__(self, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.agent_api_chats = SimpleNamespace(
                create_agent_chat=AsyncMock(
                    return_value=SimpleNamespace(data=SimpleNamespace(id="room-1"))
                )
            )
            self.agent_api_participants = SimpleNamespace(
                add_agent_chat_participant=AsyncMock(return_value=None)
            )
            self.agent_api_messages = SimpleNamespace(
                create_agent_chat_message=AsyncMock(
                    return_value=SimpleNamespace(data=SimpleNamespace(id="msg-1"))
                ),
                list_agent_messages=AsyncMock(
                    return_value=SimpleNamespace(
                        data=[
                            SimpleNamespace(
                                sender_id="agent-reviewer",
                                sender_name="Reviewer",
                                content="acknowledged",
                                message_type="text",
                            )
                        ]
                    )
                ),
            )
            client_instances.append(self)

    class _ChatRoomRequest:
        pass

    class _ParticipantRequest:
        def __init__(self, participant_id: str) -> None:
            self.participant_id = participant_id

    class _ChatMessageRequestMentionsItem:
        def __init__(self, id: str, name: str) -> None:
            self.id = id
            self.name = name

    class _ChatMessageRequest:
        def __init__(
            self,
            content: str,
            mentions: list[_ChatMessageRequestMentionsItem],
        ) -> None:
            self.content = content
            self.mentions = mentions

    thenvoi_rest_stub = ModuleType("thenvoi_rest")
    thenvoi_rest_stub.AsyncRestClient = _FakeClient

    thenvoi_rest_types_stub = ModuleType("thenvoi_rest.types")
    thenvoi_rest_types_stub.ChatMessageRequest = _ChatMessageRequest
    thenvoi_rest_types_stub.ChatMessageRequestMentionsItem = _ChatMessageRequestMentionsItem
    thenvoi_rest_types_stub.ChatRoomRequest = _ChatRoomRequest
    thenvoi_rest_types_stub.ParticipantRequest = _ParticipantRequest

    monkeypatch.setitem(sys.modules, "thenvoi_rest", thenvoi_rest_stub)
    monkeypatch.setitem(sys.modules, "thenvoi_rest.types", thenvoi_rest_types_stub)

    monkeypatch.setattr(module, "configure_logging", MagicMock())
    monkeypatch.setattr(
        module,
        "load_agent_runtime",
        MagicMock(
            side_effect=[
                {"agent_id": "agent-planner", "api_key": "planner-key"},
                {"agent_id": "agent-reviewer", "api_key": "reviewer-key"},
            ]
        ),
    )
    mock_sleep = AsyncMock()
    monkeypatch.setattr(module.asyncio, "sleep", mock_sleep)
    monkeypatch.setattr(module, "logger", MagicMock())
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    await module.main()

    assert len(client_instances) == 1
    client = client_instances[0]
    assert client.api_key == "planner-key"
    assert client.base_url == "https://rest.example"

    client.agent_api_chats.create_agent_chat.assert_awaited_once()
    client.agent_api_participants.add_agent_chat_participant.assert_awaited_once()
    add_participant_call = client.agent_api_participants.add_agent_chat_participant.await_args
    assert add_participant_call.kwargs["chat_id"] == "room-1"
    assert add_participant_call.kwargs["participant"].participant_id == "agent-reviewer"

    client.agent_api_messages.create_agent_chat_message.assert_awaited_once()
    create_message_call = client.agent_api_messages.create_agent_chat_message.await_args
    assert create_message_call.kwargs["chat_id"] == "room-1"
    assert create_message_call.kwargs["message"].mentions[0].id == "agent-reviewer"
    client.agent_api_messages.list_agent_messages.assert_awaited_once_with(chat_id="room-1")

    assert mock_sleep.await_count == 2
    assert mock_sleep.await_args_list[0].args == (3,)
    assert mock_sleep.await_args_list[1].args == (30,)
