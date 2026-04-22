"""SemanticCache - Two-level caching for LLM responses.

This module provides semantic caching for LLM responses using Redis,
built on top of RedisVL's SemanticCache.

Features:
- Level 1: Exact string match (fastest) - O(1) hash lookup
- Level 2: Semantic similarity match (vector search)
- TTL-based expiration
- Hit/miss statistics (L1/L2 breakdown)
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from redis import Redis
from redis import asyncio as aioredis
from redisvl.extensions.cache.llm import (  # type: ignore[import-untyped]
    SemanticCache as RVLSemanticCache,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .pool import RedisConnectionPool


@dataclass
class CacheResult:
    """Result from a cache lookup.

    Attributes:
        response: The cached LLM response
        similarity: Similarity score (1.0 = exact match)
        metadata: Optional metadata stored with the entry
    """

    response: str
    similarity: float
    metadata: dict[str, Any] | None = field(default=None)


class SemanticCache:
    """Two-level semantic cache for LLM responses.

    Uses Redis for persistent caching with semantic similarity matching.
    Level 1 uses fast O(1) hash lookup for exact matches, while Level 2
    uses vector similarity search for semantic matches.

    Example:
        >>> cache = SemanticCache(redis_url="redis://localhost:6379")
        >>> cache.set(query="What is Redis?", response="Redis is a database.")
        >>> result = cache.get(query="Tell me about Redis")
        >>> if result:
        ...     print(f"Hit! {result.response}")

    Args:
        redis_url: Redis connection URL
        similarity_threshold: Minimum similarity for semantic matches (0.0-1.0)
        ttl: Time-to-live in seconds (None = no expiration)
        name: Cache index name in Redis
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        similarity_threshold: float = 0.90,
        ttl: int | None = None,
        name: str = "llm_cache",
        pool: Optional["RedisConnectionPool"] = None,
    ) -> None:
        """Initialize the semantic cache.

        Args:
            redis_url: Redis connection URL
            similarity_threshold: Minimum similarity for semantic matches (0.0-1.0).
                Higher values require closer matches.
            ttl: Time-to-live in seconds (None = no expiration)
            name: Cache index name in Redis
            pool: Optional shared connection pool
        """
        self._similarity_threshold = similarity_threshold
        self._ttl = ttl
        self._name = name
        self._pool = pool

        # Use pool's URL if provided
        if pool is not None:
            self._redis_url = pool.redis_url
            self._redis = pool.get_sync_client()
        else:
            self._redis_url = redis_url
            self._redis = Redis.from_url(redis_url, decode_responses=True)

        self._l1_key = f"cache:{name}:exact"

        # Level 2: RedisVL semantic cache for vector similarity
        # Convert similarity to distance (RedisVL uses distance, not similarity)
        # distance = 1 - similarity for cosine
        distance_threshold = 1.0 - similarity_threshold

        self._cache = RVLSemanticCache(
            name=name,
            redis_url=redis_url,
            distance_threshold=distance_threshold,
            ttl=ttl,
            overwrite=True,  # Allow recreating index on threshold changes
        )

        # Statistics tracking (with L1/L2 breakdown)
        self._hits = 0
        self._misses = 0
        self._l1_hits = 0
        self._l2_hits = 0

    def _hash_query(self, query: str) -> str:
        """Generate a hash key for exact match lookup."""
        return hashlib.sha256(query.encode()).hexdigest()[:32]

    @property
    def similarity_threshold(self) -> float:
        """Minimum similarity score for cache hits."""
        return self._similarity_threshold

    @property
    def ttl(self) -> int | None:
        """Time-to-live in seconds for cache entries."""
        return self._ttl

    @property
    def name(self) -> str:
        """Cache index name in Redis."""
        return self._name

    def get(self, query: str) -> CacheResult | None:
        """Check cache for a matching response.

        Level 1: Check exact hash match (O(1) lookup).
        Level 2: Check semantic similarity match (vector search).

        Args:
            query: The query string to look up

        Returns:
            CacheResult if found, None on cache miss
        """
        # Level 1: Exact hash match (fast O(1) lookup)
        try:
            query_hash = self._hash_query(query)
            cached_data = self._redis.hget(self._l1_key, query_hash)

            if cached_data and isinstance(cached_data, str):
                # L1 hit - parse and return
                data = json.loads(cached_data)
                metadata = data.get("metadata")

                self._hits += 1
                self._l1_hits += 1
                return CacheResult(
                    response=data["response"],
                    similarity=1.0,  # Exact match
                    metadata=metadata,
                )
        except Exception as exc:
            logger.debug("L1 cache lookup failed: %s", exc)

        # Level 2: Semantic similarity match (vector search)
        try:
            results = self._cache.check(prompt=query, num_results=1)

            if not results:
                self._misses += 1
                return None

            # Got a result
            result = results[0]
            response = result.get("response", "")

            # Calculate similarity from distance
            # RedisVL returns vector_distance, similarity = 1 - distance
            distance = float(result.get("vector_distance", 0.0))
            similarity = 1.0 - distance

            # Extract metadata if present
            metadata = result.get("metadata")
            if metadata and isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {"raw": metadata}

            self._hits += 1
            self._l2_hits += 1
            return CacheResult(
                response=response,
                similarity=similarity,
                metadata=metadata,
            )

        except Exception as exc:
            logger.debug("L2 cache lookup failed: %s", exc)
            self._misses += 1
            return None

    def set(
        self,
        query: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a query-response pair in the cache.

        Stores in both Level 1 (exact hash) and Level 2 (semantic).

        Args:
            query: The query string
            response: The LLM response to cache
            metadata: Optional metadata to store with the entry
        """
        # Level 1: Store in exact hash cache
        try:
            query_hash = self._hash_query(query)
            cache_data = json.dumps(
                {
                    "response": response,
                    "metadata": metadata,
                }
            )
            self._redis.hset(self._l1_key, query_hash, cache_data)

            # Apply TTL to L1 cache key if configured
            if self._ttl is not None:
                self._redis.expire(self._l1_key, self._ttl)
        except Exception as exc:
            logger.debug("L1 cache store failed: %s", exc)

        # Level 2: Store in semantic cache
        try:
            self._cache.store(
                prompt=query,
                response=response,
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("L2 cache store failed: %s", exc)

    def get_stats(self) -> dict[str, int]:
        """Get cache hit/miss statistics.

        Returns:
            Dictionary with 'hits', 'misses', 'l1_hits', and 'l2_hits' counts
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
        }

    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        # Clear Level 1 (exact hash cache)
        try:
            self._redis.delete(self._l1_key)
        except Exception as exc:
            logger.debug("L1 cache clear failed: %s", exc)

        # Clear Level 2 (semantic cache)
        try:
            # Delete the cache index
            self._cache.delete()

            # Recreate empty cache
            distance_threshold = 1.0 - self._similarity_threshold
            self._cache = RVLSemanticCache(
                name=self._name,
                redis_url=self._redis_url,
                distance_threshold=distance_threshold,
                ttl=self._ttl,
                overwrite=True,
            )
        except Exception as exc:
            logger.error("L2 cache clear/recreate failed: %s", exc)

        # Reset statistics
        self._hits = 0
        self._misses = 0
        self._l1_hits = 0
        self._l2_hits = 0

    # Async methods

    def _get_async_redis(self) -> aioredis.Redis:
        """Get or create async Redis client."""
        if not hasattr(self, "_async_redis"):
            self._async_redis: aioredis.Redis = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._async_redis

    async def aget(self, query: str) -> CacheResult | None:
        """Async version of get() - check cache for a matching response.

        Args:
            query: The query string to look up

        Returns:
            CacheResult if found, None on cache miss
        """
        redis = self._get_async_redis()

        # Level 1: Exact hash match (fast O(1) lookup)
        try:
            query_hash = self._hash_query(query)
            cached_data = await redis.hget(self._l1_key, query_hash)  # type: ignore[misc]

            if cached_data and isinstance(cached_data, str):
                # L1 hit - parse and return
                data = json.loads(cached_data)
                metadata = data.get("metadata")

                self._hits += 1
                self._l1_hits += 1
                return CacheResult(
                    response=data["response"],
                    similarity=1.0,
                    metadata=metadata,
                )
        except Exception as exc:
            logger.debug("async L1 cache lookup failed: %s", exc)

        # Level 2: Semantic similarity match (vector search)
        # RedisVL's check() is sync, so run it in a thread
        try:
            results = await asyncio.to_thread(self._cache.check, prompt=query, num_results=1)

            if not results:
                self._misses += 1
                return None

            result = results[0]
            response = result.get("response", "")
            distance = float(result.get("vector_distance", 0.0))
            similarity = 1.0 - distance

            metadata = result.get("metadata")
            if metadata and isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {"raw": metadata}

            self._hits += 1
            self._l2_hits += 1
            return CacheResult(
                response=response,
                similarity=similarity,
                metadata=metadata,
            )

        except Exception as exc:
            logger.debug("async L2 cache lookup failed: %s", exc)
            self._misses += 1
            return None

    async def aset(
        self,
        query: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Async version of set() - store a query-response pair in the cache.

        Args:
            query: The query string
            response: The LLM response to cache
            metadata: Optional metadata to store with the entry
        """
        redis = self._get_async_redis()

        # Level 1: Store in exact hash cache
        try:
            query_hash = self._hash_query(query)
            cache_data = json.dumps(
                {
                    "response": response,
                    "metadata": metadata,
                }
            )
            await redis.hset(self._l1_key, query_hash, cache_data)  # type: ignore[misc]

            if self._ttl is not None:
                await redis.expire(self._l1_key, self._ttl)
        except Exception as exc:
            logger.debug("async L1 cache store failed: %s", exc)

        # Level 2: Store in semantic cache (sync call, run in thread)
        try:
            await asyncio.to_thread(
                self._cache.store,
                prompt=query,
                response=response,
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("async L2 cache store failed: %s", exc)
