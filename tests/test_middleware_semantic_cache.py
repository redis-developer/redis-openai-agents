"""Unit tests for SemanticCacheMiddleware."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest


class FakeModel:
    def __init__(self, response: Any = "live-response") -> None:
        self.response = response
        self.call_count = 0

    async def get_response(self, **kwargs: Any) -> Any:
        self.call_count += 1
        return self.response

    def stream_response(self, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError


class TestSemanticCacheMiddleware:
    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self, redis_url: str) -> None:
        """First call caches; second call with same input hits the cache."""
        from redis_openai_agents import SemanticCache
        from redis_openai_agents.middleware import (
            MiddlewareStack,
            SemanticCacheMiddleware,
        )

        cache = SemanticCache(
            redis_url=redis_url,
            similarity_threshold=0.99,
            name="mw_cache_test",
        )
        inner = FakeModel(response={"text": "hello from LLM"})
        stack = MiddlewareStack(
            model=inner,
            middlewares=[SemanticCacheMiddleware(cache=cache)],
        )

        call_kwargs = dict(
            system_instructions=None,
            input="What is Redis?",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        first = await stack.get_response(**call_kwargs)
        second = await stack.get_response(**call_kwargs)

        assert first == {"text": "hello from LLM"}
        assert second == {"text": "hello from LLM"}
        assert inner.call_count == 1  # second call served from cache

    @pytest.mark.asyncio
    async def test_cache_skipped_when_tools_present(self, redis_url: str) -> None:
        """Tool calls make responses non-deterministic; cache must be skipped."""
        from redis_openai_agents import SemanticCache
        from redis_openai_agents.middleware import (
            MiddlewareStack,
            SemanticCacheMiddleware,
        )

        cache = SemanticCache(redis_url=redis_url, name="mw_cache_skip_tools")
        inner = FakeModel(response="response")
        stack = MiddlewareStack(
            model=inner,
            middlewares=[SemanticCacheMiddleware(cache=cache)],
        )

        call_kwargs = dict(
            system_instructions=None,
            input="query",
            model_settings=None,
            tools=["some_tool"],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        await stack.get_response(**call_kwargs)
        await stack.get_response(**call_kwargs)

        assert inner.call_count == 2  # neither call cached

    @pytest.mark.asyncio
    async def test_cache_skipped_when_handoffs_present(self, redis_url: str) -> None:
        from redis_openai_agents import SemanticCache
        from redis_openai_agents.middleware import (
            MiddlewareStack,
            SemanticCacheMiddleware,
        )

        cache = SemanticCache(redis_url=redis_url, name="mw_cache_skip_handoffs")
        inner = FakeModel()
        stack = MiddlewareStack(
            model=inner,
            middlewares=[SemanticCacheMiddleware(cache=cache)],
        )

        call_kwargs = dict(
            system_instructions=None,
            input="query",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=["some_handoff"],
            tracing=None,
        )

        await stack.get_response(**call_kwargs)
        await stack.get_response(**call_kwargs)

        assert inner.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_skipped_when_output_schema_present(self, redis_url: str) -> None:
        from redis_openai_agents import SemanticCache
        from redis_openai_agents.middleware import (
            MiddlewareStack,
            SemanticCacheMiddleware,
        )

        cache = SemanticCache(redis_url=redis_url, name="mw_cache_skip_schema")
        inner = FakeModel()
        stack = MiddlewareStack(
            model=inner,
            middlewares=[SemanticCacheMiddleware(cache=cache)],
        )

        call_kwargs = dict(
            system_instructions=None,
            input="query",
            model_settings=None,
            tools=[],
            output_schema="some_schema",
            handoffs=[],
            tracing=None,
        )

        await stack.get_response(**call_kwargs)
        await stack.get_response(**call_kwargs)

        assert inner.call_count == 2

    @pytest.mark.asyncio
    async def test_different_inputs_produce_distinct_cache_entries(self, redis_url: str) -> None:
        from redis_openai_agents import SemanticCache
        from redis_openai_agents.middleware import (
            MiddlewareStack,
            SemanticCacheMiddleware,
        )

        cache = SemanticCache(
            redis_url=redis_url,
            similarity_threshold=0.99,
            name="mw_cache_distinct",
        )
        responses = iter([{"r": "first"}, {"r": "second"}])

        class SequenceModel:
            def __init__(self) -> None:
                self.call_count = 0

            async def get_response(self, **kwargs: Any) -> Any:
                self.call_count += 1
                return next(responses)

            def stream_response(self, **kwargs: Any) -> AsyncIterator[Any]:
                raise NotImplementedError

        inner = SequenceModel()
        stack = MiddlewareStack(
            model=inner,
            middlewares=[SemanticCacheMiddleware(cache=cache)],
        )

        first = await stack.get_response(
            system_instructions=None,
            input="question A",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )
        second = await stack.get_response(
            system_instructions=None,
            input="totally different question B",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        assert first == {"r": "first"}
        assert second == {"r": "second"}
        assert inner.call_count == 2
