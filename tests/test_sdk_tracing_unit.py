"""Unit tests for RedisTracingProcessor - testing with mocks."""

from unittest.mock import AsyncMock, MagicMock, patch

from conftest import MockSpan, MockTrace


class TestRedisTracingProcessorCreation:
    """Test processor creation and configuration."""

    def test_create_with_defaults(self) -> None:
        """Processor can be created with defaults."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()

        assert processor._redis_url == "redis://localhost:6379"
        assert processor._stream_name == "agent_traces"
        assert processor._buffer_size == 100

    def test_create_with_custom_url(self) -> None:
        """Processor can be created with custom Redis URL."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor(redis_url="redis://custom:6380")

        assert processor._redis_url == "redis://custom:6380"

    def test_create_with_custom_stream_name(self) -> None:
        """Processor can be created with custom stream name."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor(stream_name="my_traces")

        assert processor._stream_name == "my_traces"

    def test_create_with_custom_buffer_size(self) -> None:
        """Processor can be created with custom buffer size."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor(buffer_size=50)

        assert processor._buffer_size == 50

    def test_create_with_custom_ttl(self) -> None:
        """Processor can be created with custom TTL."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor(trace_ttl=3600)

        assert processor._trace_ttl == 3600


class TestRedisTracingProcessorBuffering:
    """Test buffering behavior."""

    def test_trace_start_adds_to_buffer(self) -> None:
        """on_trace_start adds event to buffer."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        trace = MockTrace("trace_1", "test")

        processor.on_trace_start(trace)

        assert len(processor._buffer) == 1
        assert processor._buffer[0]["event_type"] == "trace_start"

    def test_trace_end_adds_to_buffer(self) -> None:
        """on_trace_end adds event to buffer."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        trace = MockTrace("trace_1", "test")

        processor.on_trace_end(trace)

        assert len(processor._buffer) == 1
        assert processor._buffer[0]["event_type"] == "trace_end"

    def test_span_start_adds_to_buffer(self) -> None:
        """on_span_start adds event to buffer."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        span = MockSpan("trace_1", "span_1", "test_span")

        processor.on_span_start(span)

        assert len(processor._buffer) == 1
        assert processor._buffer[0]["event_type"] == "span_start"

    def test_span_end_adds_to_buffer(self) -> None:
        """on_span_end adds event to buffer."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        span = MockSpan("trace_1", "span_1", "test_span")

        processor.on_span_end(span)

        assert len(processor._buffer) == 1
        assert processor._buffer[0]["event_type"] == "span_end"

    async def test_buffer_cleared_on_flush(self) -> None:
        """Buffer is cleared after flush."""
        from unittest.mock import AsyncMock, MagicMock

        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        # Provide a mock async Redis client so _flush_async proceeds
        mock_client = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.xadd = MagicMock()
        mock_pipeline.hset = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[])
        mock_client.pipeline = MagicMock(return_value=mock_pipeline)
        processor._client = mock_client

        trace = MockTrace("trace_1", "test")
        processor.on_trace_start(trace)
        assert len(processor._buffer) == 1

        await processor.aforce_flush()

        assert len(processor._buffer) == 0

    def test_buffer_accumulates_events(self) -> None:
        """Multiple events accumulate in buffer."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor(buffer_size=100)  # Large buffer

        for i in range(5):
            trace = MockTrace(f"trace_{i}", "test")
            processor.on_trace_start(trace)

        assert len(processor._buffer) == 5


class TestRedisTracingProcessorEventData:
    """Test event data extraction."""

    def test_trace_start_captures_trace_id(self) -> None:
        """on_trace_start captures trace_id."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        trace = MockTrace("test_trace_123", "test")

        processor.on_trace_start(trace)

        assert processor._buffer[0]["trace_id"] == "test_trace_123"

    def test_trace_start_captures_name(self) -> None:
        """on_trace_start captures trace name."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        trace = MockTrace("trace_1", "my_agent_run")

        processor.on_trace_start(trace)

        assert processor._buffer[0]["name"] == "my_agent_run"

    def test_trace_end_captures_completion(self) -> None:
        """on_trace_end captures completion time."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        trace = MockTrace("trace_1", "test")
        trace.completed_at = 1234567890.0

        processor.on_trace_end(trace)

        assert processor._buffer[0]["completed_at"] == 1234567890.0

    def test_trace_end_captures_error(self) -> None:
        """on_trace_end captures error if present."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        trace = MockTrace("trace_1", "test")
        trace.error = "Something went wrong"

        processor.on_trace_end(trace)

        assert processor._buffer[0]["error"] == "Something went wrong"

    def test_span_captures_parent_id(self) -> None:
        """on_span_start captures parent_id."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        span = MockSpan("trace_1", "span_1", "test", parent_id="parent_span")

        processor.on_span_start(span)

        assert processor._buffer[0]["parent_id"] == "parent_span"

    def test_span_end_exports_span_data(self) -> None:
        """on_span_end exports span_data."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        span = MockSpan("trace_1", "span_1", "web_search", span_type="function")

        processor.on_span_end(span)

        span_data = processor._buffer[0]["span_data"]
        assert span_data["type"] == "function"
        assert span_data["name"] == "web_search"


class TestRedisTracingProcessorInitialization:
    """Test initialization behavior."""

    async def test_initialize_creates_client(self) -> None:
        """Initialize creates Redis client."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        assert processor._client is None

        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_from_url.return_value = mock_client

            await processor.initialize()

            assert processor._client is not None
            assert processor._initialized is True

    async def test_close_closes_client(self) -> None:
        """Close closes Redis client."""
        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()

        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_from_url.return_value = mock_client

            await processor.initialize()
            await processor.close()

            mock_client.aclose.assert_called_once()
            assert processor._client is None
            assert processor._initialized is False


class TestRedisTracingProcessorAsyncFlush:
    """Test async flushing methods."""

    async def test_aforce_flush_is_async(self) -> None:
        """aforce_flush is a coroutine."""
        import inspect

        from redis_openai_agents.tracing import RedisTracingProcessor

        processor = RedisTracingProcessor()
        assert inspect.iscoroutinefunction(processor.aforce_flush)
