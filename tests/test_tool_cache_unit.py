"""Unit tests for the cached_tool decorator.

cached_tool wraps a Python callable to memoize its return value in Redis,
keyed by a deterministic hash of the arguments. It's intended to be applied
to the body of an ``@function_tool`` to give the agent a cheap idempotent
execution path for deterministic tools.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


class TestCachedToolSync:
    def test_repeated_call_is_served_from_cache(self, redis_url: str) -> None:
        from redis_openai_agents import cached_tool

        call_count = {"n": 0}

        @cached_tool(
            name="mw_tool_cache_sync",
            redis_url=redis_url,
            ttl=60,
        )
        def get_weather(city: str) -> str:
            call_count["n"] += 1
            return f"sunny in {city}"

        assert get_weather("Paris") == "sunny in Paris"
        assert get_weather("Paris") == "sunny in Paris"
        assert call_count["n"] == 1

    def test_different_args_miss_the_cache(self, redis_url: str) -> None:
        from redis_openai_agents import cached_tool

        call_count = {"n": 0}

        @cached_tool(name="mw_tool_cache_distinct", redis_url=redis_url)
        def echo(value: str) -> str:
            call_count["n"] += 1
            return value

        echo("a")
        echo("b")
        echo("a")
        assert call_count["n"] == 2

    def test_kwargs_order_does_not_matter(self, redis_url: str) -> None:
        """Swapping kwarg order must produce the same cache key."""
        from redis_openai_agents import cached_tool

        call_count = {"n": 0}

        @cached_tool(name="mw_tool_cache_kwargs", redis_url=redis_url)
        def add(a: int, b: int) -> int:
            call_count["n"] += 1
            return a + b

        add(a=1, b=2)
        add(b=2, a=1)
        assert call_count["n"] == 1

    def test_volatile_args_bypass_cache(self, redis_url: str) -> None:
        """Tools with volatile arg names (e.g. timestamp) skip caching entirely."""
        from redis_openai_agents import cached_tool

        call_count = {"n": 0}

        @cached_tool(
            name="mw_tool_cache_volatile",
            redis_url=redis_url,
            volatile_arg_names={"timestamp"},
        )
        def fetch(query: str, timestamp: int) -> str:
            call_count["n"] += 1
            return f"{query}@{timestamp}"

        fetch("q", 1)
        fetch("q", 1)
        assert call_count["n"] == 2  # volatile arg present -> no caching

    def test_ignored_args_do_not_bust_the_key(self, redis_url: str) -> None:
        """Args declared as ignored are stripped before key hashing."""
        from redis_openai_agents import cached_tool

        call_count = {"n": 0}

        @cached_tool(
            name="mw_tool_cache_ignored",
            redis_url=redis_url,
            ignored_arg_names={"trace_id"},
        )
        def run(query: str, trace_id: str) -> str:
            call_count["n"] += 1
            return query.upper()

        run("hello", trace_id="t1")
        run("hello", trace_id="t2")
        assert call_count["n"] == 1  # trace_id ignored

    def test_side_effect_prefix_skips_caching(self, redis_url: str) -> None:
        from redis_openai_agents import cached_tool

        call_count = {"n": 0}

        @cached_tool(
            name="send_email",
            redis_url=redis_url,
            side_effect_prefixes=("send_", "delete_"),
        )
        def send_email(to: str, body: str) -> str:
            call_count["n"] += 1
            return "ok"

        send_email("a@b.com", "hi")
        send_email("a@b.com", "hi")
        assert call_count["n"] == 2


class TestCachedToolAsync:
    @pytest.mark.asyncio
    async def test_async_tool_caches(self, redis_url: str) -> None:
        from redis_openai_agents import cached_tool

        call_count = {"n": 0}

        @cached_tool(name="mw_tool_cache_async", redis_url=redis_url)
        async def fetch_user(user_id: int) -> dict:
            call_count["n"] += 1
            await asyncio.sleep(0)
            return {"id": user_id, "name": f"user-{user_id}"}

        first = await fetch_user(7)
        second = await fetch_user(7)
        assert first == second == {"id": 7, "name": "user-7"}
        assert call_count["n"] == 1


class TestCachedToolComplexArgs:
    def test_dict_args_canonicalized(self, redis_url: str) -> None:
        """Dict args with different insertion order hash the same."""
        from redis_openai_agents import cached_tool

        call_count = {"n": 0}

        @cached_tool(name="mw_tool_cache_dict_args", redis_url=redis_url)
        def with_filters(filters: dict[str, Any]) -> str:
            call_count["n"] += 1
            return str(filters)

        with_filters({"b": 1, "a": 2})
        with_filters({"a": 2, "b": 1})
        assert call_count["n"] == 1
