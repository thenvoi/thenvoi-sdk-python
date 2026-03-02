from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from thenvoi.adapters.codex.adapter import CodexAdapter, CodexAdapterConfig


@dataclass
class _FakeTools:
    def get_openai_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo input",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]


def test_adapter_build_dynamic_tools_accepts_openai_function_shape() -> None:
    adapter = CodexAdapter(config=CodexAdapterConfig())
    dynamic = adapter._build_dynamic_tools(_FakeTools())
    assert dynamic == [
        {
            "name": "echo",
            "description": "Echo input",
            "inputSchema": {"type": "object", "properties": {}},
        }
    ]


def test_adapter_static_extractors_delegate_to_helpers() -> None:
    assert (
        CodexAdapter._extract_tool_item("imageView", {"path": "/tmp/image.png"})
        == ("view_image", {"path": "/tmp/image.png"}, "viewed")
    )
    assert CodexAdapter._extract_thought_text("plan", {"text": "Do thing"}) == "Do thing"
    assert CodexAdapter._extract_turn_error({"error": {"message": "boom"}}) == "boom"

