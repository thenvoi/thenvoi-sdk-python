"""
Utility functions for Anthropic message content handling.

Pure functions for extracting and filtering content blocks.
"""

from __future__ import annotations

from anthropic.types import (
    ContentBlock,
    TextBlock,
    ToolUseBlock,
)


def extract_text_content(content: list[ContentBlock]) -> str:
    """
    Extract text content from Anthropic response content blocks.

    Args:
        content: List of content blocks from Anthropic response

    Returns:
        Joined text content from all TextBlocks
    """
    texts = [
        block.text for block in content if isinstance(block, TextBlock) and block.text
    ]
    return " ".join(texts) if texts else ""


def filter_content_blocks(
    content: list[ContentBlock],
) -> list[TextBlock | ToolUseBlock]:
    """
    Filter content blocks to only include TextBlock and ToolUseBlock.

    Filters out empty text blocks and other block types (thinking, etc.).
    The returned blocks can be used directly in MessageParam.content
    since it accepts ContentBlock types.

    Args:
        content: List of content blocks from Anthropic response

    Returns:
        List of TextBlock and ToolUseBlock objects
    """
    filtered: list[TextBlock | ToolUseBlock] = []
    for block in content:
        if isinstance(block, ToolUseBlock):
            filtered.append(block)
        elif isinstance(block, TextBlock) and block.text:
            filtered.append(block)
    return filtered
