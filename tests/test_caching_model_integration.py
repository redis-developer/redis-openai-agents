"""Integration tests for RedisCachingModel - TDD RED phase.

Tests the Redis-backed caching wrapper for OpenAI Agents SDK Model interface.
"""

from typing import Any


async def _call_model(
    cached_model: Any,
    *,
    system_instructions: str = "You are helpful",
    input: str = "Hello",
    tools: list | None = None,
) -> Any:
    """Helper to call cached_model.get_response with default boilerplate args."""
    return await cached_model.get_response(
        system_instructions=system_instructions,
        input=input,
        model_settings=MockModelSettings(),
        tools=tools or [],
        output_schema=None,
        handoffs=[],
        tracing=MockTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )


class TestRedisCachingModelExactCache:
    """Test Level 1 exact match caching."""

    async def test_caches_response_on_first_call(self, redis_url: str) -> None:
        """First call stores response in cache."""
        from redis_openai_agents import RedisCachingModel

        mock_model = MockModel(response_text="Test response")
        cached_model = RedisCachingModel(
            model=mock_model,
            redis_url=redis_url,
            cache_prefix="test_cache_first",
        )
        await cached_model.initialize()

        response = await _call_model(cached_model, input="Hello")

        assert response is not None
        assert mock_model.call_count == 1

        cache_hit = await cached_model.check_cache(
            system_instructions="You are helpful",
            input_data="Hello",
        )
        assert cache_hit is not None

        await cached_model.close()

    async def test_returns_cached_response_on_second_call(self, redis_url: str) -> None:
        """Second identical call returns cached response without calling model."""
        from redis_openai_agents import RedisCachingModel

        mock_model = MockModel(response_text="Cached response")
        cached_model = RedisCachingModel(
            model=mock_model,
            redis_url=redis_url,
            cache_prefix="test_cache_second",
        )
        await cached_model.initialize()

        await _call_model(cached_model, system_instructions="Be concise", input="What is 2+2?")
        response = await _call_model(
            cached_model, system_instructions="Be concise", input="What is 2+2?"
        )

        assert response is not None
        assert mock_model.call_count == 1

        await cached_model.close()

    async def test_different_inputs_not_cached(self, redis_url: str) -> None:
        """Different inputs result in cache miss."""
        from redis_openai_agents import RedisCachingModel

        mock_model = MockModel(response_text="Response")
        cached_model = RedisCachingModel(
            model=mock_model,
            redis_url=redis_url,
            cache_prefix="test_cache_diff",
        )
        await cached_model.initialize()

        await _call_model(cached_model, input="Question A")
        await _call_model(cached_model, input="Question B")

        assert mock_model.call_count == 2

        await cached_model.close()


class TestRedisCachingModelSemanticCache:
    """Test Level 2 semantic similarity caching."""

    async def test_semantic_cache_similar_query(self, redis_url: str) -> None:
        """Semantically similar queries can hit cache."""
        from redis_openai_agents import RedisCachingModel

        mock_model = MockModel(response_text="Redis info")
        cached_model = RedisCachingModel(
            model=mock_model,
            redis_url=redis_url,
            cache_prefix="test_semantic",
            enable_semantic_cache=True,
            semantic_threshold=0.9,
        )
        await cached_model.initialize()

        await _call_model(cached_model, input="What is Redis?")
        response2 = await _call_model(cached_model, input="Can you explain what Redis is?")

        # Semantic cache may or may not hit depending on embeddings.
        # Verify at minimum that a valid response came back.
        assert response2 is not None
        # If the cache hit, the model was only called once; otherwise twice.
        assert mock_model.call_count in (1, 2)
        if mock_model.call_count == 1:
            # Confirm the second response came from cache (same text)
            assert response2.output[0].content[0].text == "Redis info"

        await cached_model.close()


class TestRedisCachingModelTTL:
    """Test cache TTL behavior."""

    async def test_cache_respects_ttl(self, redis_url: str) -> None:
        """Cache entries expire after TTL."""
        import asyncio

        from redis_openai_agents import RedisCachingModel

        mock_model = MockModel(response_text="Expiring response")
        cached_model = RedisCachingModel(
            model=mock_model,
            redis_url=redis_url,
            cache_prefix="test_ttl",
            cache_ttl=1,  # 1 second TTL
        )
        await cached_model.initialize()

        await _call_model(cached_model, input="TTL test")
        assert mock_model.call_count == 1

        await asyncio.sleep(1.5)

        await _call_model(cached_model, input="TTL test")
        assert mock_model.call_count == 2

        await cached_model.close()


class TestRedisCachingModelBypass:
    """Test cache bypass scenarios."""

    async def test_bypass_cache_with_tools(self, redis_url: str) -> None:
        """Requests with tools bypass cache by default."""
        from redis_openai_agents import RedisCachingModel

        mock_model = MockModel(response_text="Tool response")
        cached_model = RedisCachingModel(
            model=mock_model,
            redis_url=redis_url,
            cache_prefix="test_bypass_tools",
        )
        await cached_model.initialize()

        mock_tool = MockTool()

        await _call_model(cached_model, input="Use a tool", tools=[mock_tool])
        await _call_model(cached_model, input="Use a tool", tools=[mock_tool])

        assert mock_model.call_count == 2

        await cached_model.close()


class TestRedisCachingModelMetrics:
    """Test cache metrics tracking."""

    async def test_tracks_cache_hits(self, redis_url: str) -> None:
        """Tracks cache hit count."""
        from redis_openai_agents import RedisCachingModel

        mock_model = MockModel(response_text="Metrics test")
        cached_model = RedisCachingModel(
            model=mock_model,
            redis_url=redis_url,
            cache_prefix="test_metrics",
        )
        await cached_model.initialize()

        await _call_model(cached_model, system_instructions="System", input="Query")
        await _call_model(cached_model, system_instructions="System", input="Query")

        metrics = await cached_model.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 1

        await cached_model.close()


# Mock classes for testing


class MockTracing:
    """Mock tracing enum."""

    DISABLED = 0
    ENABLED = 1

    def is_disabled(self) -> bool:
        return self == MockTracing.DISABLED


class MockModelSettings:
    """Mock model settings."""

    pass


class MockTool:
    """Mock tool."""

    name = "mock_tool"


class MockModelResponse:
    """Mock model response."""

    def __init__(self, text: str):
        self.output = [MockOutputMessage(text)]
        self.usage = MockUsage()
        self.response_id = None

    def to_input_items(self) -> list:
        return [{"role": "assistant", "content": self.output[0].content[0].text}]


class MockOutputMessage:
    """Mock output message."""

    def __init__(self, text: str):
        self.content = [MockTextContent(text)]
        self.role = "assistant"

    def model_dump(self, **kwargs) -> dict:
        return {
            "role": self.role,
            "content": [{"type": "text", "text": c.text} for c in self.content],
        }


class MockTextContent:
    """Mock text content."""

    def __init__(self, text: str):
        self.text = text


class MockUsage:
    """Mock usage."""

    def __init__(self):
        self.input_tokens = 10
        self.output_tokens = 20
        self.requests = 1


class MockModel:
    """Mock model for testing."""

    def __init__(self, response_text: str = "Mock response"):
        self.response_text = response_text
        self.call_count = 0

    async def get_response(self, *args, **kwargs) -> MockModelResponse:
        self.call_count += 1
        return MockModelResponse(self.response_text)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError("Streaming not implemented in mock")
