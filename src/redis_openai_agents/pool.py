"""RedisConnectionPool - Shared connection pooling for Redis.

This module provides connection pooling to share Redis connections
across multiple components, reducing connection overhead.

Features:
- Shared connection pool across components
- Configurable pool size
- Support for both sync and async clients
- Global default pool for easy setup
"""

from __future__ import annotations

from typing import Any

from redis import ConnectionPool, Redis
from redis import asyncio as aioredis


class RedisConnectionPool:
    """Shared Redis connection pool for all components.

    Provides connection pooling to reduce connection overhead and
    improve resource management across components.

    Example:
        >>> pool = RedisConnectionPool(
        ...     redis_url="redis://localhost:6379",
        ...     max_connections=50,
        ... )
        >>> client = pool.get_sync_client()
        >>> async_client = pool.get_async_client()

    Args:
        redis_url: Redis connection URL
        max_connections: Maximum pool size (default: 20)
        decode_responses: Whether to decode responses (default: True)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_connections: int = 20,
        decode_responses: bool = True,
    ) -> None:
        """Initialize the connection pool.

        Args:
            redis_url: Redis connection URL
            max_connections: Maximum pool size
            decode_responses: Whether to decode responses
        """
        self._redis_url = redis_url
        self._max_connections = max_connections
        self._decode_responses = decode_responses

        # Create sync connection pool
        self._sync_pool = ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            decode_responses=decode_responses,
        )

        # Lazy-initialized clients
        self._sync_client: Redis | None = None
        self._async_client: aioredis.Redis | None = None

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        return self._redis_url

    @property
    def max_connections(self) -> int:
        """Maximum pool size."""
        return self._max_connections

    @property
    def sync_pool(self) -> ConnectionPool:
        """Underlying sync connection pool."""
        return self._sync_pool

    def get_sync_client(self) -> Redis:
        """Get a sync Redis client from the pool.

        Returns:
            Redis client using the shared connection pool
        """
        if self._sync_client is None:
            self._sync_client = Redis(connection_pool=self._sync_pool)
        return self._sync_client

    def get_async_client(self) -> aioredis.Redis:
        """Get an async Redis client from the pool.

        Returns:
            Async Redis client
        """
        if self._async_client is None:
            self._async_client = aioredis.from_url(
                self._redis_url,
                max_connections=self._max_connections,
                decode_responses=self._decode_responses,
            )
        return self._async_client

    def close(self) -> None:
        """Close sync connections in the pool.

        For async contexts, use :meth:`aclose` instead to properly close
        the async client.
        """
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

        self._sync_pool.disconnect()

    async def aclose(self) -> None:
        """Close all connections including the async client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

        self.close()


# Global default pool
_default_pool: RedisConnectionPool | None = None
_default_config: dict[str, Any] = {
    "redis_url": "redis://localhost:6379",
    "max_connections": 20,
    "decode_responses": True,
}


def configure_pool(
    redis_url: str = "redis://localhost:6379",
    max_connections: int = 20,
    decode_responses: bool = True,
) -> None:
    """Configure the default connection pool.

    Call this before get_pool() to set connection options.

    Args:
        redis_url: Redis connection URL
        max_connections: Maximum pool size
        decode_responses: Whether to decode responses
    """
    global _default_config, _default_pool

    _default_config = {
        "redis_url": redis_url,
        "max_connections": max_connections,
        "decode_responses": decode_responses,
    }

    # Reset existing pool if any
    if _default_pool is not None:
        _default_pool.close()
        _default_pool = None


def get_pool() -> RedisConnectionPool:
    """Get the default shared connection pool.

    Creates the pool on first call using configured options.

    Returns:
        The default RedisConnectionPool instance
    """
    global _default_pool

    if _default_pool is None:
        _default_pool = RedisConnectionPool(**_default_config)

    return _default_pool


def reset_pool() -> None:
    """Reset the default pool.

    Closes existing connections and clears the pool.
    Useful for testing and cleanup.
    """
    global _default_pool

    if _default_pool is not None:
        _default_pool.close()
        _default_pool = None
