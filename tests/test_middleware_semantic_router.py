"""Unit tests for SemanticRouterMiddleware."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest


class FakeModel:
    def __init__(self, response: Any = "live-llm") -> None:
        self.response = response
        self.call_count = 0

    async def get_response(self, **kwargs: Any) -> Any:
        self.call_count += 1
        return self.response

    def stream_response(self, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError


class TestSemanticRouterMiddleware:
    @pytest.mark.asyncio
    async def test_matched_intent_short_circuits(self, redis_url: str) -> None:
        """When router matches a configured intent, return the canned response."""
        from redis_openai_agents import Route, SemanticRouter
        from redis_openai_agents.middleware import (
            MiddlewareStack,
            SemanticRouterMiddleware,
        )

        router = SemanticRouter(
            name="mw_router_match",
            routes=[
                Route(
                    name="greeting",
                    references=["hello", "hi there", "hey"],
                    distance_threshold=0.5,
                ),
                Route(
                    name="farewell",
                    references=["goodbye", "bye", "see you later"],
                    distance_threshold=0.5,
                ),
            ],
            redis_url=redis_url,
        )

        inner = FakeModel(response="live-llm")
        stack = MiddlewareStack(
            model=inner,
            middlewares=[
                SemanticRouterMiddleware(
                    router=router,
                    responses={"greeting": "Hi there!", "farewell": "Goodbye!"},
                )
            ],
        )

        result = await stack.get_response(
            system_instructions=None,
            input="hello",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        # String responses are auto-wrapped into a ModelResponse so the
        # Runner can consume them.
        assert hasattr(result, "output")
        assert result.output[0].content[0].text == "Hi there!"
        assert inner.call_count == 0

    @pytest.mark.asyncio
    async def test_unmatched_intent_falls_through(self, redis_url: str) -> None:
        """Unrecognized inputs delegate to the inner model."""
        from redis_openai_agents import Route, SemanticRouter
        from redis_openai_agents.middleware import (
            MiddlewareStack,
            SemanticRouterMiddleware,
        )

        router = SemanticRouter(
            name="mw_router_miss",
            routes=[
                Route(
                    name="greeting",
                    references=["hello", "hi"],
                    distance_threshold=0.3,  # strict
                ),
            ],
            redis_url=redis_url,
        )

        inner = FakeModel(response="live-llm")
        stack = MiddlewareStack(
            model=inner,
            middlewares=[
                SemanticRouterMiddleware(
                    router=router,
                    responses={"greeting": "Hi there!"},
                )
            ],
        )

        result = await stack.get_response(
            system_instructions=None,
            input="what is the square root of 144",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        assert result == "live-llm"
        assert inner.call_count == 1

    @pytest.mark.asyncio
    async def test_match_without_configured_response_falls_through(self, redis_url: str) -> None:
        """A matched route with no canned response delegates to the inner model."""
        from redis_openai_agents import Route, SemanticRouter
        from redis_openai_agents.middleware import (
            MiddlewareStack,
            SemanticRouterMiddleware,
        )

        router = SemanticRouter(
            name="mw_router_no_response",
            routes=[
                Route(name="greeting", references=["hello", "hi"]),
                Route(name="nested", references=["what is recursion?"]),
            ],
            redis_url=redis_url,
        )

        inner = FakeModel(response="live-llm")
        stack = MiddlewareStack(
            model=inner,
            middlewares=[
                SemanticRouterMiddleware(
                    router=router,
                    responses={"greeting": "Hi!"},  # no "nested" entry
                )
            ],
        )

        result = await stack.get_response(
            system_instructions=None,
            input="what is recursion?",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        assert result == "live-llm"
        assert inner.call_count == 1

    @pytest.mark.asyncio
    async def test_response_factory_takes_precedence_over_mapping(self, redis_url: str) -> None:
        """A response_factory callable overrides the static mapping."""
        from redis_openai_agents import Route, SemanticRouter
        from redis_openai_agents.middleware import (
            MiddlewareStack,
            SemanticRouterMiddleware,
        )

        router = SemanticRouter(
            name="mw_router_factory",
            routes=[Route(name="greeting", references=["hello"])],
            redis_url=redis_url,
        )

        inner = FakeModel(response="live-llm")
        stack = MiddlewareStack(
            model=inner,
            middlewares=[
                SemanticRouterMiddleware(
                    router=router,
                    response_factory=lambda match: {
                        "intent": match.name,
                        "distance": match.distance,
                    },
                )
            ],
        )

        result = await stack.get_response(
            system_instructions=None,
            input="hello",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        assert isinstance(result, dict)
        assert result["intent"] == "greeting"
        assert inner.call_count == 0
