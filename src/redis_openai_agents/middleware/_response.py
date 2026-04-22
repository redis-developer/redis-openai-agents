"""Response construction helpers for short-circuiting middlewares.

When a middleware returns a response without calling the LLM (for
example a semantic-router match), the ``Runner`` still expects a real
``ModelResponse`` with ``.output`` and ``.usage``. These helpers build
one cheaply from plain Python values.
"""

from __future__ import annotations

import uuid
from typing import Any


def text_response(text: str, *, response_id: str | None = None) -> Any:
    """Build a :class:`ModelResponse` carrying a single assistant text message.

    Args:
        text: Plain text body.
        response_id: Optional ID to stamp on the response. A random UUID is
            used if not provided, mirroring the pattern LangGraph-redis
            uses for cache hits so the frontend treats each hit as a new
            message.

    Returns:
        A ``ModelResponse`` instance. Imports are deferred so the
        middleware package does not hard-depend on
        ``openai.types.responses`` at import time.
    """
    from agents.items import ModelResponse
    from agents.usage import Usage
    from openai.types.responses.response_output_message import ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText

    message = ResponseOutputMessage(
        id=response_id or f"msg_middleware_{uuid.uuid4().hex[:12]}",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(
                type="output_text",
                text=text,
                annotations=[],
            )
        ],
    )

    return ModelResponse(
        output=[message],
        usage=Usage(),
        response_id=response_id,
    )


def is_model_response(value: Any) -> bool:
    """True if ``value`` already looks like a ``ModelResponse``.

    Duck-typed rather than ``isinstance`` so third-party subclasses and
    test doubles work.
    """
    return hasattr(value, "output") and hasattr(value, "usage")
