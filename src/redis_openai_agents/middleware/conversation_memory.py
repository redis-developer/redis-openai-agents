"""ConversationMemoryMiddleware - inject semantically relevant past messages.

On each model call, looks up the top-K past messages most relevant to the
current user input and prepends them to the request. After the handler
returns, stores both the user turn and the assistant reply in the history
for future retrieval.

Backed by :class:`redisvl.extensions.message_history.SemanticMessageHistory`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from ._utils import extract_user_text
from .base import ModelCallHandler, ModelRequest

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redisvl.extensions.message_history import (  # type: ignore[import-untyped]
        SemanticMessageHistory,
    )


class ConversationMemoryMiddleware:
    """Prepend semantically relevant past messages into the request.

    Args:
        history: A :class:`SemanticMessageHistory` instance backing the
            retrieval and storage. Callers are responsible for its
            lifecycle.
        session_tag: Tag passed to history queries and inserts. Allows
            tenant / user / conversation isolation.
        top_k: Maximum number of past messages to prepend.
        distance_threshold: Optional override for relevance matching.
        persist_reply: When True (default), the assistant reply is also
            stored back into history so follow-up turns can retrieve it.
        response_text_extractor: Optional callable that turns a model
            response into the text to store. Defaults to best-effort
            extraction from OpenAI Responses-shaped responses.
    """

    def __init__(
        self,
        history: SemanticMessageHistory,
        *,
        session_tag: str | None = None,
        top_k: int = 5,
        distance_threshold: float | None = None,
        persist_reply: bool = True,
        response_text_extractor: Any = None,
    ) -> None:
        self._history = history
        self._session_tag = session_tag
        self._top_k = top_k
        self._distance_threshold = distance_threshold
        self._persist_reply = persist_reply
        self._extract_response_text = response_text_extractor or _default_extract

    async def awrap_model_call(self, request: ModelRequest, handler: ModelCallHandler) -> Any:
        prompt = self._extract_prompt(request)
        if prompt:
            relevant = await asyncio.to_thread(self._fetch_relevant, prompt)
            if relevant:
                request.input = self._merge_input(request.input, relevant)

        response = await handler(request)

        if prompt and self._persist_reply:
            reply_text = self._extract_response_text(response)
            await asyncio.to_thread(self._persist, prompt, reply_text)

        return response

    def _fetch_relevant(self, prompt: str) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "top_k": self._top_k,
            "raw": False,
            "as_text": False,
        }
        if self._session_tag is not None:
            kwargs["session_tag"] = self._session_tag
        if self._distance_threshold is not None:
            kwargs["distance_threshold"] = self._distance_threshold
        try:
            result = self._history.get_relevant(prompt, **kwargs)
        except Exception as exc:
            logger.debug("Failed to fetch relevant history: %s", exc)
            return []
        # get_relevant returns List[Dict[str,str]] when as_text=False.
        return [dict(item) for item in result if isinstance(item, dict)]

    @staticmethod
    def _extract_prompt(request: ModelRequest) -> str:
        return extract_user_text(request.input)

    @staticmethod
    def _merge_input(current: Any, prepend: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a list input with past turns in front of the current input."""
        normalized = [_normalize_turn(turn) for turn in prepend]
        if isinstance(current, list):
            return [*normalized, *current]
        return [*normalized, {"role": "user", "content": str(current)}]

    def _persist(self, user_text: str, assistant_text: str) -> None:
        try:
            messages = [{"role": "user", "content": user_text}]
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
            if self._session_tag is not None:
                self._history.add_messages(messages, session_tag=self._session_tag)
            else:
                self._history.add_messages(messages)
        except Exception as exc:
            # History persistence errors must not fail the model call.
            logger.debug("Failed to persist conversation history: %s", exc)


def _normalize_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Normalize a stored history turn for the Agents SDK input format.

    Older entries may carry the deprecated "llm" role; the SDK expects
    "assistant".
    """
    role = turn.get("role") or turn.get("type") or "user"
    if role == "llm":
        role = "assistant"
    content = turn.get("content", "")
    return {"role": role, "content": content}


def _default_extract(response: Any) -> str:
    """Best-effort extraction of the assistant text from a ModelResponse."""
    output = getattr(response, "output", None)
    if not output:
        return ""
    for item in output:
        content = getattr(item, "content", None)
        if isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    return text
    return ""
