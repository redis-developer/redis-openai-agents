"""Unit tests for ConversationMemoryMiddleware."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest


class RecordingModel:
    """Fake Model that records the incoming request for assertion."""

    def __init__(self, response: Any = "ok") -> None:
        self.response = response
        self.last_input: Any = None
        self.call_count = 0

    async def get_response(self, **kwargs: Any) -> Any:
        self.last_input = kwargs.get("input")
        self.call_count += 1
        return self.response

    def stream_response(self, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError


class TestConversationMemoryMiddleware:
    @pytest.mark.asyncio
    async def test_empty_history_passes_input_through(self, redis_url: str) -> None:
        """With no stored history, input is forwarded unchanged to the model."""
        from redisvl.extensions.message_history import SemanticMessageHistory

        from redis_openai_agents.middleware import (
            ConversationMemoryMiddleware,
            MiddlewareStack,
        )

        history = SemanticMessageHistory(
            name="mw_mem_empty",
            session_tag="session-empty",
            redis_url=redis_url,
            overwrite=True,
        )

        inner = RecordingModel()
        stack = MiddlewareStack(
            model=inner,
            middlewares=[
                ConversationMemoryMiddleware(
                    history=history,
                    session_tag="session-empty",
                    top_k=3,
                )
            ],
        )

        await stack.get_response(
            system_instructions=None,
            input="Hello",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        # Input was forwarded; with no relevant history nothing was prepended.
        assert inner.last_input == "Hello" or (
            isinstance(inner.last_input, list) and len(inner.last_input) == 1
        )

    @pytest.mark.asyncio
    async def test_relevant_history_is_prepended(self, redis_url: str) -> None:
        """Semantically relevant prior messages are prepended to input."""
        from redisvl.extensions.message_history import SemanticMessageHistory

        from redis_openai_agents.middleware import (
            ConversationMemoryMiddleware,
            MiddlewareStack,
        )

        history = SemanticMessageHistory(
            name="mw_mem_relevant",
            session_tag="session-relevant",
            redis_url=redis_url,
            overwrite=True,
            distance_threshold=0.7,
        )
        history.add_messages(
            [
                {"role": "user", "content": "My favourite colour is indigo."},
                {"role": "assistant", "content": "Noted: your favourite colour is indigo."},
            ]
        )

        inner = RecordingModel()
        mw = ConversationMemoryMiddleware(
            history=history,
            session_tag="session-relevant",
            top_k=5,
        )
        stack = MiddlewareStack(model=inner, middlewares=[mw])

        await stack.get_response(
            system_instructions=None,
            input="What colour did I mention earlier?",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        assert isinstance(inner.last_input, list)
        texts = " ".join(
            str(item.get("content", "")) if isinstance(item, dict) else str(item)
            for item in inner.last_input
        )
        assert "indigo" in texts

    @pytest.mark.asyncio
    async def test_existing_list_input_is_preserved(self, redis_url: str) -> None:
        """Prior list-form input is preserved; relevant history is prepended."""
        from redisvl.extensions.message_history import SemanticMessageHistory

        from redis_openai_agents.middleware import (
            ConversationMemoryMiddleware,
            MiddlewareStack,
        )

        history = SemanticMessageHistory(
            name="mw_mem_list_input",
            session_tag="session-list",
            redis_url=redis_url,
            overwrite=True,
            distance_threshold=0.7,
        )
        history.add_messages(
            [
                {"role": "user", "content": "I live in Amsterdam."},
                {"role": "assistant", "content": "Got it."},
            ]
        )

        inner = RecordingModel()
        mw = ConversationMemoryMiddleware(
            history=history,
            session_tag="session-list",
            top_k=3,
        )
        stack = MiddlewareStack(model=inner, middlewares=[mw])

        user_list = [{"role": "user", "content": "What city do I live in?"}]

        await stack.get_response(
            system_instructions=None,
            input=user_list,
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        assert isinstance(inner.last_input, list)
        assert inner.last_input[-1] == user_list[0]  # original preserved at tail
        # At least one prepended item mentioning the city
        assert any(
            isinstance(item, dict) and "Amsterdam" in str(item.get("content", ""))
            for item in inner.last_input[:-1]
        )

    @pytest.mark.asyncio
    async def test_persists_exchange_after_call(self, redis_url: str) -> None:
        """After a successful call, the new exchange is saved to history."""
        from redisvl.extensions.message_history import SemanticMessageHistory

        from redis_openai_agents.middleware import (
            ConversationMemoryMiddleware,
            MiddlewareStack,
            text_response,
        )

        history = SemanticMessageHistory(
            name="mw_mem_persist",
            session_tag="session-persist",
            redis_url=redis_url,
            overwrite=True,
        )

        class TextModel:
            async def get_response(self, **kwargs: Any) -> Any:
                return text_response("The capital is Paris.")

            def stream_response(self, **kwargs: Any) -> AsyncIterator[Any]:
                raise NotImplementedError

        mw = ConversationMemoryMiddleware(
            history=history,
            session_tag="session-persist",
            top_k=3,
        )
        stack = MiddlewareStack(model=TextModel(), middlewares=[mw])

        await stack.get_response(
            system_instructions=None,
            input="What is the capital of France?",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        # Both the user turn and the assistant reply should be in history now.
        recent = history.get_recent(top_k=5, session_tag="session-persist")
        assert isinstance(recent, list)
        joined = " ".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in recent
        )
        assert "France" in joined or "capital" in joined
        assert "Paris" in joined
