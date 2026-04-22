"""Integration tests for RedisTracingProcessor - TDD RED phase.

These tests define the expected behavior for storing OpenAI Agents SDK
traces in Redis for observability.
"""

import time

from conftest import MockSpan, MockSpanData, MockTrace


class TestRedisTracingProcessorTraceStorage:
    """Test trace storage functionality."""

    async def test_store_trace_start(self, redis_url: str) -> None:
        """Can store trace start event."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_traces",
        )
        await processor.initialize()

        # Create a mock trace object
        trace_id = f"trace_{int(time.time())}"
        trace = MockTrace(trace_id=trace_id, name="test_agent_run")

        # Record trace start
        processor.on_trace_start(trace)
        await processor.aforce_flush()

        # Verify trace was stored
        stored = await processor.get_trace(trace_id)
        assert stored is not None
        assert stored["trace_id"] == trace_id

        await processor.close()

    async def test_store_trace_end(self, redis_url: str) -> None:
        """Can store trace end event with completion data."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_traces_end",
        )
        await processor.initialize()

        trace_id = f"trace_{int(time.time())}"
        trace = MockTrace(trace_id=trace_id, name="test_agent_run")

        # Record trace lifecycle
        processor.on_trace_start(trace)
        trace.completed_at = time.time()
        processor.on_trace_end(trace)
        await processor.aforce_flush()

        # Verify completion was recorded
        stored = await processor.get_trace(trace_id)
        assert stored is not None
        assert "completed_at" in stored

        await processor.close()


class TestRedisTracingProcessorSpanStorage:
    """Test span storage functionality."""

    async def test_store_span_start(self, redis_url: str) -> None:
        """Can store span start event."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_spans",
        )
        await processor.initialize()

        trace_id = f"trace_{int(time.time())}"
        span_id = f"span_{int(time.time())}"
        span = MockSpan(
            trace_id=trace_id,
            span_id=span_id,
            name="function_call",
            span_type="function",
        )

        processor.on_span_start(span)
        await processor.aforce_flush()

        # Verify span was stored
        spans = await processor.get_spans(trace_id)
        assert len(spans) >= 1
        assert any(s["span_id"] == span_id for s in spans)

        await processor.close()

    async def test_store_span_end_with_data(self, redis_url: str) -> None:
        """Can store span end event with operation data."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_spans_end",
        )
        await processor.initialize()

        trace_id = f"trace_{int(time.time())}"
        span_id = f"span_{int(time.time())}"
        span = MockSpan(
            trace_id=trace_id,
            span_id=span_id,
            name="web_search",
            span_type="function",
        )
        span.span_data = MockSpanData(
            type="function",
            name="web_search",
            input={"query": "redis docs"},
            output={"results": ["result1", "result2"]},
        )

        processor.on_span_start(span)
        processor.on_span_end(span)
        await processor.aforce_flush()

        # Verify span data was stored
        spans = await processor.get_spans(trace_id)
        completed_span = next((s for s in spans if s["span_id"] == span_id), None)
        assert completed_span is not None

        await processor.close()

    async def test_span_parent_relationship(self, redis_url: str) -> None:
        """Spans correctly track parent-child relationships."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_spans_parent",
        )
        await processor.initialize()

        trace_id = f"trace_{int(time.time())}"
        parent_span_id = f"span_parent_{int(time.time())}"
        child_span_id = f"span_child_{int(time.time())}"

        parent_span = MockSpan(
            trace_id=trace_id,
            span_id=parent_span_id,
            name="agent_run",
            span_type="agent",
        )
        child_span = MockSpan(
            trace_id=trace_id,
            span_id=child_span_id,
            parent_id=parent_span_id,
            name="tool_call",
            span_type="function",
        )

        processor.on_span_start(parent_span)
        processor.on_span_start(child_span)
        await processor.aforce_flush()

        spans = await processor.get_spans(trace_id)
        child = next((s for s in spans if s["span_id"] == child_span_id), None)
        assert child is not None
        assert child.get("parent_id") == parent_span_id

        await processor.close()


class TestRedisTracingProcessorQueries:
    """Test trace querying functionality."""

    async def test_list_recent_traces(self, redis_url: str) -> None:
        """Can list recent traces."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_traces_list",
        )
        await processor.initialize()

        # Create multiple traces
        for i in range(3):
            trace = MockTrace(trace_id=f"trace_list_{i}_{int(time.time())}", name=f"run_{i}")
            processor.on_trace_start(trace)
            processor.on_trace_end(trace)

        await processor.aforce_flush()

        # List recent traces
        traces = await processor.list_traces(limit=10)
        assert len(traces) >= 3

        await processor.close()

    async def test_filter_traces_by_name(self, redis_url: str) -> None:
        """Can filter traces by agent name."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_traces_filter",
        )
        await processor.initialize()

        # Create traces with different names
        trace1 = MockTrace(trace_id=f"trace_filter_1_{int(time.time())}", name="research_agent")
        trace2 = MockTrace(trace_id=f"trace_filter_2_{int(time.time())}", name="analysis_agent")

        processor.on_trace_start(trace1)
        processor.on_trace_end(trace1)
        processor.on_trace_start(trace2)
        processor.on_trace_end(trace2)
        await processor.aforce_flush()

        # Filter by name
        filtered = await processor.list_traces(name_filter="research")
        assert all("research" in t.get("name", "") for t in filtered)

        await processor.close()


class TestRedisTracingProcessorLifecycle:
    """Test processor lifecycle management."""

    async def test_shutdown_flushes_buffer(self, redis_url: str) -> None:
        """Shutdown flushes any buffered data."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_shutdown",
        )
        await processor.initialize()

        trace = MockTrace(trace_id=f"trace_shutdown_{int(time.time())}", name="test")
        processor.on_trace_start(trace)

        # Shutdown without manual flush
        processor.shutdown()

        # Verify data was flushed
        stored = await processor.get_trace(trace.trace_id)
        assert stored is not None

        await processor.close()

    async def test_force_flush_immediate(self, redis_url: str) -> None:
        """Force flush immediately writes buffered data."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_force_flush",
        )
        await processor.initialize()

        trace = MockTrace(trace_id=f"trace_flush_{int(time.time())}", name="test")
        processor.on_trace_start(trace)

        # Force flush
        await processor.aforce_flush()

        # Should be immediately available
        stored = await processor.get_trace(trace.trace_id)
        assert stored is not None

        await processor.close()


class TestRedisTracingProcessorBuffering:
    """Test buffering behavior for performance."""

    async def test_buffer_batches_writes(self, redis_url: str) -> None:
        """Processor buffers writes for performance."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_buffer",
            buffer_size=5,  # Flush every 5 events
        )
        await processor.initialize()

        # Add traces without reaching buffer limit
        for i in range(3):
            trace = MockTrace(trace_id=f"trace_buf_{i}_{int(time.time())}", name="test")
            processor.on_trace_start(trace)

        # Buffer not full, manual flush needed
        await processor.aforce_flush()

        await processor.close()

    async def test_auto_flush_on_buffer_full(self, redis_url: str) -> None:
        """Buffer automatically flushes when full."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_auto_flush",
            buffer_size=3,
        )
        await processor.initialize()

        # Add enough events to trigger auto-flush
        trace_ids = []
        for i in range(5):
            trace_id = f"trace_auto_{i}_{int(time.time())}"
            trace_ids.append(trace_id)
            trace = MockTrace(trace_id=trace_id, name="test")
            processor.on_trace_start(trace)

        # Should have auto-flushed at least the first 3
        # Final flush to ensure all written
        await processor.aforce_flush()

        for trace_id in trace_ids[:3]:
            stored = await processor.get_trace(trace_id)
            assert stored is not None

        await processor.close()


class TestRedisTracingProcessorStreamStorage:
    """Test Redis Stream-based storage for replay capability."""

    async def test_traces_stored_in_stream(self, redis_url: str) -> None:
        """Traces are stored in Redis Stream for replay."""
        from redis_openai_agents import RedisTracingProcessor

        processor = RedisTracingProcessor(
            redis_url=redis_url,
            stream_name="test_stream_store",
        )
        await processor.initialize()

        trace = MockTrace(trace_id=f"trace_stream_{int(time.time())}", name="test")
        processor.on_trace_start(trace)
        processor.on_trace_end(trace)
        await processor.aforce_flush()

        # Verify stored in stream (can be replayed)
        stream_length = await processor.get_stream_length()
        assert stream_length >= 1

        await processor.close()
