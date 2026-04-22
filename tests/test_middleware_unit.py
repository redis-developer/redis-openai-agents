"""Unit tests for the AgentMiddleware protocol and MiddlewareStack.

These tests use a fake in-memory Model to verify composition behavior
without requiring OpenAI API access.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest


class FakeModel:
    """Minimal Model stand-in that records calls and returns canned responses."""

    def __init__(self, response: Any = "fake-response") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def get_response(
        self,
        system_instructions: str | None = None,
        input: Any = None,
        model_settings: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        handoffs: Any = None,
        tracing: Any = None,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> Any:
        self.calls.append(
            {
                "system_instructions": system_instructions,
                "input": input,
                "tools": tools,
                "output_schema": output_schema,
                "handoffs": handoffs,
            }
        )
        return self.response

    def stream_response(self, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class TestMiddlewareStack:
    """Composition semantics for MiddlewareStack."""

    @pytest.mark.asyncio
    async def test_no_middleware_passes_through_to_inner_model(self) -> None:
        from redis_openai_agents.middleware import MiddlewareStack

        inner = FakeModel(response="from-inner")
        stack = MiddlewareStack(model=inner, middlewares=[])

        result = await stack.get_response(
            system_instructions="sys",
            input="hi",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        assert result == "from-inner"
        assert len(inner.calls) == 1

    @pytest.mark.asyncio
    async def test_middleware_can_short_circuit(self) -> None:
        """A middleware that returns without calling handler prevents the LLM call."""
        from redis_openai_agents.middleware import MiddlewareStack

        class ShortCircuitMiddleware:
            async def awrap_model_call(self, request: Any, handler: Any) -> Any:
                return "cached"

        inner = FakeModel(response="live")
        stack = MiddlewareStack(model=inner, middlewares=[ShortCircuitMiddleware()])

        result = await stack.get_response(
            system_instructions=None,
            input="hi",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        assert result == "cached"
        assert inner.calls == []  # inner was never reached

    @pytest.mark.asyncio
    async def test_middleware_can_mutate_response(self) -> None:
        from redis_openai_agents.middleware import MiddlewareStack

        class UppercaseMiddleware:
            async def awrap_model_call(self, request: Any, handler: Any) -> Any:
                response = await handler(request)
                return response.upper()

        inner = FakeModel(response="hello")
        stack = MiddlewareStack(model=inner, middlewares=[UppercaseMiddleware()])

        result = await stack.get_response(
            system_instructions=None,
            input="x",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_middleware_chain_runs_outer_to_inner_then_inner_to_outer(self) -> None:
        """First middleware sees the request first and the response last (onion model)."""
        from redis_openai_agents.middleware import MiddlewareStack

        order: list[str] = []

        class TracingMiddleware:
            def __init__(self, label: str) -> None:
                self.label = label

            async def awrap_model_call(self, request: Any, handler: Any) -> Any:
                order.append(f"{self.label}:before")
                response = await handler(request)
                order.append(f"{self.label}:after")
                return response

        inner = FakeModel(response="ok")
        stack = MiddlewareStack(
            model=inner,
            middlewares=[TracingMiddleware("outer"), TracingMiddleware("inner")],
        )

        await stack.get_response(
            system_instructions=None,
            input="x",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=None,
        )

        assert order == [
            "outer:before",
            "inner:before",
            "inner:after",
            "outer:after",
        ]

    @pytest.mark.asyncio
    async def test_middleware_receives_request_with_all_fields(self) -> None:
        from redis_openai_agents.middleware import MiddlewareStack, ModelRequest

        captured: list[ModelRequest] = []

        class CaptureMiddleware:
            async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
                captured.append(request)
                return await handler(request)

        inner = FakeModel(response="ok")
        stack = MiddlewareStack(model=inner, middlewares=[CaptureMiddleware()])

        await stack.get_response(
            system_instructions="sys",
            input="hi",
            model_settings="settings-sentinel",
            tools=["tool1"],
            output_schema="schema-sentinel",
            handoffs=["handoff1"],
            tracing="tracing-sentinel",
            previous_response_id="prev",
            conversation_id="conv",
            prompt="prompt-sentinel",
        )

        assert len(captured) == 1
        req = captured[0]
        assert req.system_instructions == "sys"
        assert req.input == "hi"
        assert req.model_settings == "settings-sentinel"
        assert req.tools == ["tool1"]
        assert req.output_schema == "schema-sentinel"
        assert req.handoffs == ["handoff1"]
        assert req.tracing == "tracing-sentinel"
        assert req.previous_response_id == "prev"
        assert req.conversation_id == "conv"
        assert req.prompt == "prompt-sentinel"

    @pytest.mark.asyncio
    async def test_stream_response_delegates_to_inner(self) -> None:
        """Streaming passes through the stack unmodified for now."""
        from redis_openai_agents.middleware import MiddlewareStack

        inner = AsyncMock()
        inner.stream_response = lambda **kw: _yield("a", "b", "c")
        stack = MiddlewareStack(model=inner, middlewares=[])

        chunks = [
            c
            async for c in stack.stream_response(
                system_instructions=None,
                input="x",
                model_settings=None,
                tools=[],
                output_schema=None,
                handoffs=[],
                tracing=None,
            )
        ]
        assert chunks == ["a", "b", "c"]


async def _yield(*items: Any) -> AsyncIterator[Any]:
    for item in items:
        yield item


class TestExtractUserText:
    """Unit tests for the shared extract_user_text utility."""

    def test_string_input(self) -> None:
        from redis_openai_agents.middleware._utils import extract_user_text

        assert extract_user_text("hello") == "hello"

    def test_list_input_with_user_role(self) -> None:
        from redis_openai_agents.middleware._utils import extract_user_text

        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Redis?"},
        ]
        assert extract_user_text(msgs) == "What is Redis?"

    def test_list_input_picks_last_user_message(self) -> None:
        from redis_openai_agents.middleware._utils import extract_user_text

        msgs = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
        ]
        assert extract_user_text(msgs) == "follow-up"

    def test_content_block_list(self) -> None:
        from redis_openai_agents.middleware._utils import extract_user_text

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        assert extract_user_text(msgs) == "part one part two"

    def test_no_user_returns_empty(self) -> None:
        from redis_openai_agents.middleware._utils import extract_user_text

        msgs = [{"role": "system", "content": "sys"}]
        assert extract_user_text(msgs) == ""

    def test_fallback_to_last_item(self) -> None:
        from redis_openai_agents.middleware._utils import extract_user_text

        msgs = [{"role": "system", "content": "sys"}]
        result = extract_user_text(msgs, fallback_to_last=True)
        assert result == str(msgs[-1])

    def test_empty_input(self) -> None:
        from redis_openai_agents.middleware._utils import extract_user_text

        assert extract_user_text("") == ""
        assert extract_user_text([]) == ""
        assert extract_user_text(None) == ""
