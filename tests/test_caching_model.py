"""Unit tests for RedisCachingModel."""

import json
from unittest.mock import AsyncMock, MagicMock

from redis_openai_agents.caching_model import (
    CachedModelResponse,
    CachedUsage,
    CacheMetrics,
    RedisCachingModel,
)


class TestCacheMetrics:
    """Test CacheMetrics dataclass."""

    def test_default_values(self) -> None:
        """CacheMetrics has correct defaults."""
        metrics = CacheMetrics()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.semantic_hits == 0

    def test_hit_rate_zero_total(self) -> None:
        """Hit rate is 0 when no requests."""
        metrics = CacheMetrics()

        assert metrics.hit_rate == 0.0

    def test_hit_rate_all_hits(self) -> None:
        """Hit rate is 1.0 when all hits."""
        metrics = CacheMetrics(hits=10, misses=0)

        assert metrics.hit_rate == 1.0

    def test_hit_rate_all_misses(self) -> None:
        """Hit rate is 0 when all misses."""
        metrics = CacheMetrics(hits=0, misses=10)

        assert metrics.hit_rate == 0.0

    def test_hit_rate_mixed(self) -> None:
        """Hit rate calculated correctly with mixed results."""
        metrics = CacheMetrics(hits=3, misses=7)

        assert metrics.hit_rate == 0.3


class TestCachedUsage:
    """Test CachedUsage dataclass."""

    def test_default_values(self) -> None:
        """CachedUsage has correct defaults."""
        usage = CachedUsage()

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.requests == 1

    def test_custom_values(self) -> None:
        """CachedUsage accepts custom values."""
        usage = CachedUsage(input_tokens=100, output_tokens=50, requests=2)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.requests == 2


class TestCachedModelResponse:
    """Test CachedModelResponse dataclass."""

    def test_basic_creation(self) -> None:
        """Can create CachedModelResponse."""
        response = CachedModelResponse(
            output=[{"role": "assistant", "content": "Hello"}],
            usage=CachedUsage(input_tokens=10, output_tokens=5),
            response_id="resp_123",
        )

        assert response.output == [{"role": "assistant", "content": "Hello"}]
        assert response.usage.input_tokens == 10
        assert response.response_id == "resp_123"

    def test_to_input_items_with_dicts(self) -> None:
        """to_input_items returns dict items as-is."""
        response = CachedModelResponse(
            output=[
                {"role": "assistant", "content": "Hello"},
                {"role": "assistant", "content": "World"},
            ],
            usage=CachedUsage(),
            response_id=None,
        )

        items = response.to_input_items()

        assert len(items) == 2
        assert items[0] == {"role": "assistant", "content": "Hello"}

    def test_to_input_items_with_model_dump(self) -> None:
        """to_input_items calls model_dump on objects that have it."""
        mock_item = MagicMock()
        mock_item.model_dump.return_value = {"role": "assistant", "content": "Test"}

        response = CachedModelResponse(
            output=[mock_item],
            usage=CachedUsage(),
            response_id=None,
        )

        items = response.to_input_items()

        assert len(items) == 1
        mock_item.model_dump.assert_called_once_with(exclude_unset=True)


class TestRedisCachingModelInit:
    """Test RedisCachingModel initialization."""

    def test_default_values(self) -> None:
        """RedisCachingModel has correct default values."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        assert cached_model._model == mock_model
        assert cached_model._redis_url == "redis://localhost:6379"
        assert cached_model._cache_prefix == "model_cache"
        assert cached_model._cache_ttl == 3600
        assert cached_model._enable_semantic_cache is False
        assert cached_model._semantic_threshold == 0.95
        assert cached_model._client is None

    def test_custom_values(self) -> None:
        """RedisCachingModel accepts custom configuration."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(
            model=mock_model,
            redis_url="redis://custom:6380",
            cache_prefix="custom_cache",
            cache_ttl=7200,
            enable_semantic_cache=True,
            semantic_threshold=0.90,
        )

        assert cached_model._redis_url == "redis://custom:6380"
        assert cached_model._cache_prefix == "custom_cache"
        assert cached_model._cache_ttl == 7200
        assert cached_model._enable_semantic_cache is True
        assert cached_model._semantic_threshold == 0.90


class TestRedisCachingModelCacheKey:
    """Test cache key computation."""

    def test_compute_cache_key_string_input(self) -> None:
        """Cache key computed correctly for string input."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        key1 = cached_model._compute_cache_key("System instructions", "Hello")
        key2 = cached_model._compute_cache_key("System instructions", "Hello")

        # Same inputs produce same key
        assert key1 == key2
        assert key1.startswith("model_cache:exact:")

    def test_compute_cache_key_different_inputs(self) -> None:
        """Different inputs produce different keys."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        key1 = cached_model._compute_cache_key("System", "Input A")
        key2 = cached_model._compute_cache_key("System", "Input B")

        assert key1 != key2

    def test_compute_cache_key_different_system(self) -> None:
        """Different system instructions produce different keys."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        key1 = cached_model._compute_cache_key("System A", "Input")
        key2 = cached_model._compute_cache_key("System B", "Input")

        assert key1 != key2

    def test_compute_cache_key_list_input(self) -> None:
        """Cache key handles list input."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        key = cached_model._compute_cache_key("System", [{"role": "user", "content": "Hello"}])

        assert key.startswith("model_cache:exact:")

    def test_compute_cache_key_none_system(self) -> None:
        """Cache key handles None system instructions."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        key = cached_model._compute_cache_key(None, "Input")

        assert key.startswith("model_cache:exact:")


class TestRedisCachingModelBypass:
    """Test cache bypass logic."""

    def test_bypass_with_tools(self) -> None:
        """Cache bypassed when tools provided."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        result = cached_model._should_bypass_cache(
            tools=[MagicMock()],
            handoffs=[],
            output_schema=None,
        )

        assert result is True

    def test_bypass_with_handoffs(self) -> None:
        """Cache bypassed when handoffs provided."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        result = cached_model._should_bypass_cache(
            tools=[],
            handoffs=[MagicMock()],
            output_schema=None,
        )

        assert result is True

    def test_bypass_with_output_schema(self) -> None:
        """Cache bypassed when output schema provided."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        result = cached_model._should_bypass_cache(
            tools=[],
            handoffs=[],
            output_schema=MagicMock(),
        )

        assert result is True

    def test_no_bypass_when_empty(self) -> None:
        """Cache not bypassed when all empty."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        result = cached_model._should_bypass_cache(
            tools=[],
            handoffs=[],
            output_schema=None,
        )

        assert result is False


class TestRedisCachingModelSerialize:
    """Test response serialization."""

    def test_serialize_response(self) -> None:
        """Response serialized correctly."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        mock_output_item = MagicMock()
        mock_output_item.model_dump.return_value = {
            "role": "assistant",
            "content": "Hello",
        }

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5
        mock_usage.requests = 1

        mock_response = MagicMock()
        mock_response.output = [mock_output_item]
        mock_response.usage = mock_usage
        mock_response.response_id = "resp_123"

        result = cached_model._serialize_response(mock_response)

        assert result["output"] == [{"role": "assistant", "content": "Hello"}]
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        assert result["response_id"] == "resp_123"
        assert "cached_at" in result


class TestRedisCachingModelDeserialize:
    """Test response deserialization."""

    def test_deserialize_response(self) -> None:
        """Response deserialized correctly."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        cached_data = {
            "output": [{"role": "assistant", "content": "Hello"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "requests": 1,
            },
            "response_id": "resp_123",
        }

        result = cached_model._deserialize_response(cached_data)

        assert isinstance(result, CachedModelResponse)
        assert result.output == [{"role": "assistant", "content": "Hello"}]
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.response_id == "resp_123"

    def test_deserialize_response_missing_fields(self) -> None:
        """Deserialization handles missing fields."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        cached_data = {}

        result = cached_model._deserialize_response(cached_data)

        assert result.output == []
        assert result.usage.input_tokens == 0
        assert result.response_id is None


class TestRedisCachingModelCheckCache:
    """Test cache checking."""

    async def test_check_cache_no_client(self) -> None:
        """Returns None when Redis not initialized."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        result = await cached_model.check_cache("System", "Input")

        assert result is None

    async def test_check_cache_hit(self) -> None:
        """Returns cached data on hit."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        cached_data = {"output": [], "usage": {}}

        mock_client = AsyncMock()
        mock_client.get.return_value = json.dumps(cached_data)
        cached_model._client = mock_client

        result = await cached_model.check_cache("System", "Input")

        assert result == cached_data

    async def test_check_cache_miss(self) -> None:
        """Returns None on cache miss."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        mock_client = AsyncMock()
        mock_client.get.return_value = None
        cached_model._client = mock_client

        result = await cached_model.check_cache("System", "Input")

        assert result is None


class TestRedisCachingModelGetMetrics:
    """Test metrics retrieval."""

    async def test_get_metrics(self) -> None:
        """Returns correct metrics."""
        mock_model = MagicMock()
        cached_model = RedisCachingModel(model=mock_model)

        cached_model._metrics.hits = 5
        cached_model._metrics.misses = 3
        cached_model._metrics.semantic_hits = 2

        metrics = await cached_model.get_metrics()

        assert metrics["cache_hits"] == 5
        assert metrics["cache_misses"] == 3
        assert metrics["semantic_hits"] == 2
        assert metrics["hit_rate"] == 5 / 8
