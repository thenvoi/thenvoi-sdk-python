"""Kore.ai XO Platform adapter.

Bridges Thenvoi's WebSocket-based agent model and Kore.ai's HTTP webhook
model.  See :mod:`thenvoi.integrations.koreai` for the full implementation.

Install the optional dependency::

    pip install thenvoi-sdk[koreai]
    # or
    uv add thenvoi-sdk[koreai]
"""

from __future__ import annotations

from thenvoi.integrations.koreai.adapter import KoreAIAdapter as KoreAIAdapter
from thenvoi.integrations.koreai.types import KoreAIConfig as KoreAIConfig

__all__ = ["KoreAIAdapter", "KoreAIConfig"]
