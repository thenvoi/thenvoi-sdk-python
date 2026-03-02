"""Focused gateway converter normalization tests."""

from __future__ import annotations


def test_gateway_converter_normalizes_sender_type_case() -> None:
    """Gateway converter should treat sender type casing consistently."""
    from thenvoi.converters.a2a_gateway import GatewayHistoryConverter

    converter = GatewayHistoryConverter()
    result = converter.convert(
        [
            {
                "message_type": "text",
                "sender_id": "agent-lower",
                "sender_type": "agent",
                "room_id": "room-1",
            },
            {
                "message_type": "text",
                "sender_id": "agent-title",
                "sender_type": "Agent",
                "room_id": "room-1",
            },
            {
                "message_type": "text",
                "sender_id": "agent-upper",
                "type": "AGENT",
                "room_id": "room-1",
            },
        ]
    )

    assert result.room_participants == {
        "room-1": {"agent-lower", "agent-title", "agent-upper"}
    }
