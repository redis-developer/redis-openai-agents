"""Integration tests for async support across all components.

Tests real async method behavior using a live Redis instance,
replacing previous mock-wiring tests that only verified internal state.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
class TestAsyncSemanticCache:
    """Tests for async SemanticCache methods."""

    async def test_async_get_cache_miss(self, redis_url: str) -> None:
        """Async get returns None on cache miss."""
        from redis_openai_agents import SemanticCache

        cache = SemanticCache(redis_url=redis_url, name="test_async_miss")

        result = await cache.aget("query that was never cached")

        assert result is None

    async def test_async_set_and_get_roundtrip(self, redis_url: str) -> None:
        """Async set stores value that async get can retrieve."""
        from redis_openai_agents import SemanticCache

        cache = SemanticCache(
            redis_url=redis_url,
            name="test_async_roundtrip",
            similarity_threshold=0.99,
        )

        await cache.aset(query="What is Redis?", response="An in-memory data store.")
        result = await cache.aget("What is Redis?")

        assert result is not None
        assert result.response == "An in-memory data store."


@pytest.mark.asyncio
class TestAsyncVectorStore:
    """Tests for async RedisVectorStore methods."""

    async def test_async_add_documents_returns_ids(self, redis_url: str) -> None:
        """Async add returns document IDs."""
        from redis_openai_agents import RedisVectorStore

        store = RedisVectorStore(name="test_async_add", redis_url=redis_url)

        docs = [
            {"content": "Python is a programming language.", "metadata": {}},
            {"content": "Redis is a fast database.", "metadata": {}},
        ]
        ids = await store.aadd_documents(docs)

        assert len(ids) == 2
        assert all(isinstance(id_, str) for id_ in ids)

    async def test_async_search_returns_results(self, redis_url: str) -> None:
        """Async search returns matching documents."""
        from redis_openai_agents import RedisVectorStore

        store = RedisVectorStore(name="test_async_search", redis_url=redis_url)

        await store.aadd_documents(
            [
                {"content": "Redis is a fast in-memory database.", "metadata": {}},
                {"content": "Python is a programming language.", "metadata": {}},
            ]
        )

        results = await store.asearch("Redis database", k=2)

        assert len(results) >= 1
        assert "Redis" in results[0]["content"]


@pytest.mark.asyncio
class TestAsyncMetrics:
    """Tests for async AgentMetrics methods."""

    async def test_async_record_stores_metrics(self, redis_url: str) -> None:
        """Async record stores metrics in Redis TimeSeries."""
        from redis_openai_agents import AgentMetrics

        metrics = AgentMetrics(name="test_async_record", redis_url=redis_url)

        # Should not raise
        await metrics.arecord(latency_ms=100.0, cache_hit=True)
        await metrics.arecord(latency_ms=200.0, cache_hit=False)

        stats = metrics.get_stats()
        assert stats["count"] >= 2
        assert stats["latency_avg"] > 0


@pytest.mark.asyncio
class TestAsyncStreamTransport:
    """Tests for async RedisStreamTransport methods."""

    async def test_async_publish_returns_event_id(self, redis_url: str) -> None:
        """Async publish returns a Redis stream event ID."""
        from redis_openai_agents import RedisStreamTransport

        stream = RedisStreamTransport(
            stream_name="test_async_publish",
            redis_url=redis_url,
        )

        event_id = await stream.apublish(event_type="test", data={"key": "value"})

        assert event_id is not None
        assert isinstance(event_id, str)
        # Redis stream IDs have format "timestamp-sequence"
        assert "-" in event_id

    async def test_async_subscribe_yields_published_events(self, redis_url: str) -> None:
        """Async subscribe yields events that were previously published."""
        from redis_openai_agents import RedisStreamTransport

        stream = RedisStreamTransport(
            stream_name="test_async_subscribe",
            redis_url=redis_url,
        )

        await stream.apublish(event_type="greeting", data={"msg": "hello"})

        events = []
        async for event in stream.asubscribe(last_id="0"):
            events.append(event)
            break  # Just get one event

        assert len(events) == 1
        assert events[0]["type"] == "greeting"


@pytest.mark.asyncio
class TestAsyncRunner:
    """Tests for async RedisAgentRunner methods."""

    async def test_async_run_with_cache_hit(self) -> None:
        """Async run returns cached response on cache hit."""
        from redis_openai_agents import RedisAgentRunner

        mock_cache = MagicMock()
        mock_cache_result = MagicMock()
        mock_cache_result.response = "Cached answer"
        mock_cache_result.similarity = 0.99
        mock_cache.aget = AsyncMock(return_value=mock_cache_result)

        mock_metrics = MagicMock()
        mock_metrics.arecord = AsyncMock()

        runner = RedisAgentRunner(cache=mock_cache, metrics=mock_metrics)

        result = await runner.arun(
            agent=MagicMock(),
            input="What is Redis?",
        )

        assert result.response == "Cached answer"
        assert result.cache_hit is True
        mock_metrics.arecord.assert_called_once()

    async def test_async_run_cache_miss_calls_sdk(self) -> None:
        """Async run calls SDK on cache miss."""
        from redis_openai_agents import RedisAgentRunner

        mock_cache = MagicMock()
        mock_cache.aget = AsyncMock(return_value=None)  # Cache miss
        mock_cache.aset = AsyncMock()

        mock_metrics = MagicMock()
        mock_metrics.arecord = AsyncMock()

        mock_session = MagicMock()
        mock_session.astore_agent_result = AsyncMock()

        mock_sdk_result = MagicMock()
        mock_sdk_result.final_output = "SDK response"

        with patch(
            "redis_openai_agents.runner._acall_sdk_runner",
            AsyncMock(return_value=mock_sdk_result),
        ):
            runner = RedisAgentRunner(
                cache=mock_cache,
                metrics=mock_metrics,
                session=mock_session,
            )

            await runner.arun(
                agent=MagicMock(),
                input="What is Redis?",
            )

        mock_cache.aset.assert_called_once()
        mock_session.astore_agent_result.assert_called_once_with(mock_sdk_result)


@pytest.mark.asyncio
class TestAsyncSession:
    """Tests for async AgentSession methods."""

    async def test_async_add_and_get_messages(self, redis_url: str) -> None:
        """Async add_message stores messages retrievable via aget_messages."""
        from redis_openai_agents import AgentSession

        session = AgentSession.create(user_id="test_async_session", redis_url=redis_url)

        await session.aadd_message(role="user", content="Hello")
        await session.aadd_message(role="assistant", content="Hi there!")

        messages = await session.aget_messages()

        assert len(messages) >= 2
        contents = [m.get("content", "") for m in messages]
        assert "Hello" in contents
        assert "Hi there!" in contents
