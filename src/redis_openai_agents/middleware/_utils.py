"""Shared utilities for middleware modules."""

from __future__ import annotations

from typing import Any


def extract_user_text(input: Any, *, fallback_to_last: bool = False) -> str:
    """Extract the most recent user message text from a model request input.

    Works with plain string inputs and OpenAI Agents SDK list inputs
    containing dicts with ``role`` / ``content`` keys.

    Args:
        input: The ``ModelRequest.input`` value — either a ``str`` or a
            ``list`` of message dicts.
        fallback_to_last: When ``True`` and no ``user``/``human`` role is
            found in a list input, stringify the last item as a fallback.

    Returns:
        The extracted user text, or ``""`` if nothing could be found.
    """
    if isinstance(input, str):
        return input

    if isinstance(input, list):
        for item in reversed(input):
            if isinstance(item, dict) and item.get("role") in ("user", "human"):
                content = item.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") in ("input_text", "text")
                    ]
                    return " ".join(p for p in parts if p)
        if fallback_to_last and input:
            return str(input[-1])

    return ""
