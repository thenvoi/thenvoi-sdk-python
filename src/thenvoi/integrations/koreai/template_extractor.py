"""Extract readable text from Kore.ai rich message templates."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_text(text_field: str | dict[str, Any] | list[Any]) -> str:
    """Extract displayable text from a Kore.ai callback ``text`` field.

    Kore.ai bots can return plain strings, structured templates (buttons,
    carousels, quick replies), or arrays of messages. This function
    normalizes all formats to a readable plain-text string.

    Args:
        text_field: The ``text`` value from a Kore.ai callback payload.

    Returns:
        A plain-text representation of the message content.
    """
    if isinstance(text_field, str):
        return text_field

    if isinstance(text_field, list):
        return _extract_from_list(text_field)

    if isinstance(text_field, dict):
        return _extract_from_template(text_field)

    # Fallback: stringify
    return str(text_field)


def _extract_from_list(items: list[Any]) -> str:
    """Extract text from a list of callback elements."""
    parts: list[str] = []
    for item in items:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("val") or ""
            if isinstance(text, str) and text:
                parts.append(text)
            elif isinstance(text, dict):
                parts.append(_extract_from_template(text))
            else:
                parts.append(json.dumps(item))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _extract_from_template(template: dict[str, Any]) -> str:
    """Extract readable text from a structured template."""
    # Check for template with payload
    template_type = template.get("type")
    payload = template.get("payload", {})

    if template_type == "template" and isinstance(payload, dict):
        return _extract_payload(payload)

    # Might be a direct payload without the wrapper
    if "template_type" in template:
        return _extract_payload(template)

    # Simple text object
    if "text" in template and isinstance(template["text"], str):
        return template["text"]

    # Last resort
    return json.dumps(template)


def _extract_payload(payload: dict[str, Any]) -> str:
    """Extract text from a template payload by type."""
    ttype = payload.get("template_type", "")

    if ttype == "button":
        return _extract_buttons(payload)
    elif ttype == "quick_reply":
        return _extract_quick_replies(payload)
    elif ttype in ("generic", "carousel"):
        return _extract_carousel(payload)
    elif ttype == "list":
        return _extract_list(payload)
    elif ttype == "text":
        return payload.get("text", "")

    # Unknown template type: include the text field if present, otherwise JSON
    text = payload.get("text", "")
    if isinstance(text, str) and text:
        return text
    return json.dumps(payload)


def _extract_buttons(payload: dict[str, Any]) -> str:
    """Extract text from a button template."""
    lines: list[str] = []
    main_text = payload.get("text", "")
    if main_text:
        lines.append(str(main_text))

    buttons = payload.get("buttons", [])
    for btn in buttons:
        title = btn.get("title", "")
        if title:
            lines.append("- %s" % title)

    return "\n".join(lines)


def _extract_quick_replies(payload: dict[str, Any]) -> str:
    """Extract text from a quick reply template."""
    lines: list[str] = []
    main_text = payload.get("text", "")
    if main_text:
        lines.append(str(main_text))

    replies = payload.get("quick_replies", [])
    for reply in replies:
        title = reply.get("title", "")
        if title:
            lines.append("- %s" % title)

    return "\n".join(lines)


def _extract_carousel(payload: dict[str, Any]) -> str:
    """Extract text from a carousel/generic template."""
    lines: list[str] = []
    elements = payload.get("elements", [])

    for i, card in enumerate(elements, 1):
        title = card.get("title", "")
        subtitle = card.get("subtitle", "")
        if title:
            lines.append("%d. %s" % (i, title))
            if subtitle:
                lines.append("   %s" % subtitle)

    return "\n".join(lines)


def _extract_list(payload: dict[str, Any]) -> str:
    """Extract text from a list template."""
    lines: list[str] = []
    list_title = payload.get("title") or payload.get("text", "")
    if list_title:
        lines.append(str(list_title))

    elements = payload.get("elements", [])
    for item in elements:
        title = item.get("title", "")
        if title:
            lines.append("- %s" % title)

    return "\n".join(lines)
