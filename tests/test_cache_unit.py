"""Unit tests for SemanticCache - TDD RED phase.

These tests define the expected API and behavior for the SemanticCache class.
They should fail initially until the implementation is complete.
"""

import time

from redis_openai_agents import SemanticCache


class TestSemanticCacheBasics:
    """Test basic cache operations."""

    def test_create_cache(self, redis_url: str) -> None:
        """Cache can be instantiated with redis_url."""
        cache = SemanticCache(redis_url=redis_url)
        assert cache is not None

    def test_create_cache_with_options(self, redis_url: str) -> None:
        """Cache can be created with custom options."""
        cache = SemanticCache(
            redis_url=redis_url,
            similarity_threshold=0.90,
            ttl=3600,
            name="test_cache",
        )
        assert cache.similarity_threshold == 0.90
        assert cache.ttl == 3600
        assert cache.name == "test_cache"

    def test_cache_miss_returns_none(self, redis_url: str) -> None:
        """Cache miss returns None."""
        cache = SemanticCache(redis_url=redis_url, name="test_miss")
        result = cache.get(query="never seen before query xyz123")
        assert result is None

    def test_exact_match_cache_hit(self, redis_url: str) -> None:
        """Exact query match returns cached response."""
        cache = SemanticCache(redis_url=redis_url, name="test_exact")

        cache.set(query="What is Redis?", response="Redis is a database.")
        result = cache.get(query="What is Redis?")

        assert result is not None
        assert result.response == "Redis is a database."
        assert result.similarity >= 0.99  # Near-perfect match

    def test_cache_with_ttl_expiration(self, redis_url: str) -> None:
        """Cached entries expire after TTL."""
        cache = SemanticCache(redis_url=redis_url, ttl=1, name="test_ttl")

        cache.set(query="test query", response="test response")

        # Verify it's there
        result_before = cache.get(query="test query")
        assert result_before is not None

        # Wait for TTL expiration
        time.sleep(2)

        # Should be gone
        result_after = cache.get(query="test query")
        assert result_after is None


class TestSemanticMatching:
    """Test semantic similarity matching."""

    def test_semantic_match_similar_query(self, redis_url: str) -> None:
        """Similar queries return cached response above threshold."""
        cache = SemanticCache(
            redis_url=redis_url,
            similarity_threshold=0.85,
            name="test_semantic",
        )

        cache.set(query="What is Redis?", response="Redis is a database.")

        # Semantically similar query
        result = cache.get(query="Tell me about Redis")

        assert result is not None
        assert "Redis" in result.response or "database" in result.response

    def test_semantic_no_match_below_threshold(self, redis_url: str) -> None:
        """Dissimilar queries don't match."""
        cache = SemanticCache(
            redis_url=redis_url,
            similarity_threshold=0.95,
            name="test_no_match",
        )

        cache.set(query="What is Redis?", response="Redis is a database.")

        # Completely different query
        result = cache.get(query="How do I make pizza?")

        assert result is None


class TestCacheMetadata:
    """Test metadata storage and retrieval."""

    def test_store_and_retrieve_metadata(self, redis_url: str) -> None:
        """Metadata is stored and retrieved with cache entry."""
        cache = SemanticCache(redis_url=redis_url, name="test_metadata")

        cache.set(
            query="test query",
            response="test response",
            metadata={"model": "gpt-4", "tokens": 100},
        )

        result = cache.get(query="test query")

        assert result is not None
        assert result.metadata is not None
        assert result.metadata.get("model") == "gpt-4"
        assert result.metadata.get("tokens") == 100

    def test_cache_without_metadata(self, redis_url: str) -> None:
        """Cache works without metadata."""
        cache = SemanticCache(redis_url=redis_url, name="test_no_meta")

        cache.set(query="test query", response="test response")

        result = cache.get(query="test query")

        assert result is not None
        assert result.response == "test response"


class TestCacheStatistics:
    """Test cache hit/miss statistics."""

    def test_track_hit_count(self, redis_url: str) -> None:
        """Cache tracks hit count."""
        cache = SemanticCache(redis_url=redis_url, name="test_hits")

        cache.set(query="test", response="response")
        cache.get(query="test")  # Hit
        cache.get(query="test")  # Hit

        stats = cache.get_stats()
        assert stats["hits"] == 2

    def test_track_miss_count(self, redis_url: str) -> None:
        """Cache tracks miss count."""
        cache = SemanticCache(redis_url=redis_url, name="test_misses")

        cache.get(query="unknown1")  # Miss
        cache.get(query="unknown2")  # Miss

        stats = cache.get_stats()
        assert stats["misses"] == 2

    def test_stats_initially_zero(self, redis_url: str) -> None:
        """Stats start at zero."""
        cache = SemanticCache(redis_url=redis_url, name="test_zero_stats")

        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestTwoLevelCache:
    """Test two-level caching (exact hash + semantic)."""

    def test_exact_match_is_level1_hit(self, redis_url: str) -> None:
        """Exact string match uses Level 1 (hash) cache."""
        cache = SemanticCache(redis_url=redis_url, name="test_l1")

        cache.set(query="What is Redis?", response="Redis is a database.")
        result = cache.get(query="What is Redis?")

        assert result is not None
        assert result.response == "Redis is a database."
        # Exact matches should have similarity of 1.0
        assert result.similarity == 1.0

        # Check stats show L1 hit
        stats = cache.get_stats()
        assert stats.get("l1_hits", 0) >= 1

    def test_semantic_match_is_level2_hit(self, redis_url: str) -> None:
        """Semantic similarity uses Level 2 (vector) cache."""
        cache = SemanticCache(
            redis_url=redis_url,
            similarity_threshold=0.75,  # Lower threshold for semantic matching
            name="test_l2",
        )

        cache.set(query="What is Redis used for?", response="Redis is a database.")

        # Semantically similar but not exact - different phrasing of same question
        result = cache.get(query="What can Redis be used for?")

        assert result is not None
        assert "Redis" in result.response

        # Check stats show L2 hit (not L1)
        stats = cache.get_stats()
        assert stats.get("l2_hits", 0) >= 1

    def test_l1_faster_than_l2(self, redis_url: str) -> None:
        """Level 1 lookups should be faster than Level 2."""
        import time

        cache = SemanticCache(redis_url=redis_url, name="test_speed")

        # Store entry
        cache.set(query="benchmark query", response="benchmark response")

        # Measure L1 (exact match) time
        start = time.time()
        for _ in range(10):
            cache.get(query="benchmark query")
        l1_time = time.time() - start

        # L1 hits should be fast (< 500ms for 10 lookups)
        assert l1_time < 0.5

    def test_stats_track_both_levels(self, redis_url: str) -> None:
        """Statistics track L1 and L2 hits separately."""
        cache = SemanticCache(
            redis_url=redis_url,
            similarity_threshold=0.80,
            name="test_l1l2_stats",
        )

        # Create entry
        cache.set(query="What is Python?", response="Python is a language.")

        # L1 hit (exact match)
        cache.get(query="What is Python?")

        # L2 hit (semantic match)
        cache.get(query="Tell me about Python programming")

        stats = cache.get_stats()
        assert "l1_hits" in stats
        assert "l2_hits" in stats
        assert stats["l1_hits"] >= 1
        # L2 might or might not hit depending on threshold

    def test_clear_clears_both_levels(self, redis_url: str) -> None:
        """Clear removes entries from both L1 and L2."""
        cache = SemanticCache(redis_url=redis_url, name="test_clear_both")

        cache.set(query="test entry", response="test response")

        # Verify both levels work
        assert cache.get(query="test entry") is not None

        cache.clear()

        # Both levels should be empty
        assert cache.get(query="test entry") is None


class TestCacheClear:
    """Test cache clearing operations."""

    def test_clear_cache(self, redis_url: str) -> None:
        """Clear removes all entries."""
        cache = SemanticCache(redis_url=redis_url, name="test_clear")

        cache.set(query="query1", response="response1")
        cache.set(query="query2", response="response2")

        # Verify entries exist
        assert cache.get(query="query1") is not None
        assert cache.get(query="query2") is not None

        # Clear
        cache.clear()

        # Verify entries removed
        assert cache.get(query="query1") is None
        assert cache.get(query="query2") is None

    def test_clear_resets_stats(self, redis_url: str) -> None:
        """Clear resets statistics."""
        cache = SemanticCache(redis_url=redis_url, name="test_clear_stats")

        cache.set(query="test", response="response")
        cache.get(query="test")
        cache.get(query="unknown")

        # Stats should have values
        stats_before = cache.get_stats()
        assert stats_before["hits"] > 0 or stats_before["misses"] > 0

        # Clear and check stats reset
        cache.clear()
        stats_after = cache.get_stats()
        assert stats_after["hits"] == 0
        assert stats_after["misses"] == 0
