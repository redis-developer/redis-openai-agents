"""Unit tests for connection pooling and shared Redis connections."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestRedisConnectionPool:
    """Tests for RedisConnectionPool class."""

    def test_create_pool(self) -> None:
        """Pool should be created with given URL."""
        from redis_openai_agents.pool import RedisConnectionPool

        pool = RedisConnectionPool(redis_url="redis://localhost:6379")

        assert pool.redis_url == "redis://localhost:6379"
        assert pool.sync_pool is not None

    def test_get_sync_client(self) -> None:
        """Should return a sync Redis client from the pool."""
        from redis_openai_agents.pool import RedisConnectionPool

        pool = RedisConnectionPool(redis_url="redis://localhost:6379")
        client = pool.get_sync_client()

        assert client is not None

    def test_get_async_client(self) -> None:
        """Should return an async Redis client from the pool."""
        from redis_openai_agents.pool import RedisConnectionPool

        pool = RedisConnectionPool(redis_url="redis://localhost:6379")
        client = pool.get_async_client()

        assert client is not None

    def test_same_client_returned(self) -> None:
        """Same client should be returned on multiple calls."""
        from redis_openai_agents.pool import RedisConnectionPool

        pool = RedisConnectionPool(redis_url="redis://localhost:6379")
        client1 = pool.get_sync_client()
        client2 = pool.get_sync_client()

        assert client1 is client2

    def test_pool_settings(self) -> None:
        """Pool should accept connection settings."""
        from redis_openai_agents.pool import RedisConnectionPool

        pool = RedisConnectionPool(
            redis_url="redis://localhost:6379",
            max_connections=50,
            decode_responses=True,
        )

        assert pool.max_connections == 50


class TestSharedConnectionPool:
    """Tests for shared/global connection pool."""

    @pytest.fixture(autouse=True)
    def _reset_global_pool(self) -> None:
        from redis_openai_agents.pool import reset_pool

        reset_pool()
        yield
        reset_pool()

    def test_get_default_pool(self) -> None:
        """get_pool() should return a default pool."""
        from redis_openai_agents.pool import get_pool

        pool = get_pool()

        assert pool is not None

    def test_same_pool_returned(self) -> None:
        """Same pool should be returned on multiple calls."""
        from redis_openai_agents.pool import get_pool

        pool1 = get_pool()
        pool2 = get_pool()

        assert pool1 is pool2

    def test_configure_default_pool(self) -> None:
        """configure_pool() should set pool options."""
        from redis_openai_agents.pool import configure_pool, get_pool

        configure_pool(
            redis_url="redis://custom:6379",
            max_connections=100,
        )

        pool = get_pool()
        assert pool.redis_url == "redis://custom:6379"
        assert pool.max_connections == 100


class TestComponentsUsePool:
    """Tests verifying components can use shared pool."""

    def test_cache_accepts_pool(self) -> None:
        """SemanticCache should accept a connection pool."""
        from redis_openai_agents.pool import RedisConnectionPool

        pool = RedisConnectionPool(redis_url="redis://localhost:6379")

        # This should not raise
        from redis_openai_agents import SemanticCache

        # Mock the RedisVL cache to avoid actual connection
        with patch("redis_openai_agents.cache.RVLSemanticCache"):
            cache = SemanticCache(pool=pool, name="test")

        assert cache._pool is pool

    def test_session_accepts_pool(self) -> None:
        """AgentSession should accept a connection pool."""
        from redis_openai_agents.pool import RedisConnectionPool

        pool = RedisConnectionPool(redis_url="redis://localhost:6379")

        from redis_openai_agents import AgentSession

        with patch("redis_openai_agents.session.MessageHistory"):
            session = AgentSession(
                user_id="test",
                pool=pool,
            )

        assert session._pool is pool

    def test_metrics_accepts_pool(self) -> None:
        """AgentMetrics should accept a connection pool."""
        from redis_openai_agents.pool import RedisConnectionPool

        pool = RedisConnectionPool(redis_url="redis://localhost:6379")

        from redis_openai_agents import AgentMetrics

        metrics = AgentMetrics(name="test", pool=pool)

        assert metrics._pool is pool

    def test_stream_accepts_pool(self) -> None:
        """RedisStreamTransport should accept a connection pool."""
        from redis_openai_agents.pool import RedisConnectionPool

        pool = RedisConnectionPool(redis_url="redis://localhost:6379")

        from redis_openai_agents import RedisStreamTransport

        # Mock the xgroup_create to avoid actual connection
        with patch.object(pool.get_sync_client(), "xgroup_create"):
            stream = RedisStreamTransport(
                stream_name="test",
                pool=pool,
            )

        assert stream._pool is pool
