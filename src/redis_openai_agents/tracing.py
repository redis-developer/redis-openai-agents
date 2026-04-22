"""RedisTracingProcessor - Store OpenAI Agents SDK traces in Redis.

This module provides a tracing processor that stores trace and span data
in Redis for debugging, analysis, and replay capabilities.

Features:
- Buffered writes for performance
- Redis Streams for replay capability
- Redis Hash for quick trace lookup
- Parent-child span relationship tracking

Example:
    >>> from redis_openai_agents import RedisTracingProcessor
    >>> from agents.tracing import setup_tracing
    >>>
    >>> # Create processor
    >>> processor = RedisTracingProcessor(redis_url="redis://localhost:6379")
    >>> await processor.initialize()
    >>>
    >>> # Register with OpenAI Agents SDK
    >>> setup_tracing(processors=[processor])
    >>>
    >>> # Query traces later
    >>> traces = await processor.list_traces(limit=10)
    >>> spans = await processor.get_spans(trace_id="trace_123")
"""

import asyncio
import json
import logging
import time
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisTracingProcessor:
    """Stores OpenAI Agents SDK traces in Redis for observability.

    This processor implements the TracingProcessor interface from the
    OpenAI Agents SDK to capture trace and span lifecycle events.

    Traces are stored in:
    - Redis Streams for replay capability and time-series access
    - Redis Hashes for quick trace/span lookup by ID

    Attributes:
        redis_url: Redis connection URL.
        stream_name: Name of the Redis Stream for storing events.
        buffer_size: Number of events to buffer before flushing.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_name: str = "agent_traces",
        buffer_size: int = 100,
        trace_ttl: int = 86400 * 7,  # 7 days default TTL
    ) -> None:
        """Initialize the tracing processor.

        Args:
            redis_url: Redis connection URL.
            stream_name: Name of the Redis Stream for events.
            buffer_size: Number of events to buffer before auto-flush.
            trace_ttl: TTL in seconds for trace data (default 7 days).
        """
        self._redis_url = redis_url
        self._stream_name = stream_name
        self._buffer_size = buffer_size
        self._trace_ttl = trace_ttl

        self._client: redis.Redis | None = None
        self._buffer: list[dict[str, Any]] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Redis connection.

        Must be called before using the processor.
        """
        self._client = redis.from_url(self._redis_url, decode_responses=True)
        self._initialized = True

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    def on_trace_start(self, trace: Any) -> None:
        """Called when a trace begins.

        Args:
            trace: The Trace object from OpenAI Agents SDK.
        """
        event = {
            "event_type": "trace_start",
            "trace_id": getattr(trace, "trace_id", str(id(trace))),
            "name": getattr(trace, "name", "unknown"),
            "started_at": getattr(trace, "started_at", time.time()),
            "timestamp": time.time(),
        }
        self._buffer.append(event)
        self._maybe_flush()

    def on_trace_end(self, trace: Any) -> None:
        """Called when a trace completes.

        Args:
            trace: The Trace object from OpenAI Agents SDK.
        """
        event = {
            "event_type": "trace_end",
            "trace_id": getattr(trace, "trace_id", str(id(trace))),
            "name": getattr(trace, "name", "unknown"),
            "completed_at": getattr(trace, "completed_at", time.time()),
            "error": getattr(trace, "error", None),
            "timestamp": time.time(),
        }
        self._buffer.append(event)
        self._maybe_flush()

    def on_span_start(self, span: Any) -> None:
        """Called when a span begins.

        Args:
            span: The Span object from OpenAI Agents SDK.
        """
        span_data = getattr(span, "span_data", None)
        span_type = getattr(span_data, "type", "unknown") if span_data else "unknown"

        event = {
            "event_type": "span_start",
            "trace_id": getattr(span, "trace_id", "unknown"),
            "span_id": getattr(span, "span_id", str(id(span))),
            "parent_id": getattr(span, "parent_id", None),
            "name": getattr(span, "name", "unknown"),
            "span_type": span_type,
            "started_at": getattr(span, "started_at", time.time()),
            "timestamp": time.time(),
        }
        self._buffer.append(event)
        self._maybe_flush()

    def on_span_end(self, span: Any) -> None:
        """Called when a span completes.

        Args:
            span: The Span object from OpenAI Agents SDK.
        """
        span_data = getattr(span, "span_data", None)

        # Export span data if available
        exported_data = {}
        if span_data and hasattr(span_data, "export"):
            try:
                exported_data = span_data.export()
            except Exception as exc:
                logger.debug("span_data.export() failed: %s", exc)

        event = {
            "event_type": "span_end",
            "trace_id": getattr(span, "trace_id", "unknown"),
            "span_id": getattr(span, "span_id", str(id(span))),
            "parent_id": getattr(span, "parent_id", None),
            "name": getattr(span, "name", "unknown"),
            "span_type": exported_data.get("type", "unknown"),
            "finished_at": getattr(span, "finished_at", time.time()),
            "error": getattr(span, "error", None),
            "span_data": exported_data,
            "timestamp": time.time(),
        }
        self._buffer.append(event)
        self._maybe_flush()

    def shutdown(self) -> None:
        """Called on application shutdown.

        Flushes any buffered events before shutdown.
        """
        self.force_flush()

    def force_flush(self) -> None:
        """Force immediate flushing of buffered events.

        This method blocks until all buffered events are written.
        For use in async contexts, use aforce_flush() instead.
        """
        if not self._buffer or not self._client:
            return

        # Use asyncio to run the async flush
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context - this won't work well
                # The sync flush in async context is best-effort
                # Use aforce_flush() for reliable async flushing
                import threading

                # Create a new event loop in a thread for sync behavior
                result_event = threading.Event()
                exception_holder: list = []

                def run_in_thread() -> None:
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            new_loop.run_until_complete(self._flush_async_direct())
                        finally:
                            new_loop.close()
                    except Exception as e:
                        exception_holder.append(e)
                    finally:
                        result_event.set()

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                result_event.wait(timeout=5.0)
                thread.join(timeout=1.0)

                if exception_holder:
                    raise exception_holder[0]
            else:
                loop.run_until_complete(self._flush_async())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(self._flush_async())

    async def aforce_flush(self) -> None:
        """Async version of force_flush().

        Force immediate flushing of buffered events.
        Use this in async contexts for reliable flushing.
        """
        await self._flush_async()

    def _maybe_flush(self) -> None:
        """Flush buffer if it's full."""
        if len(self._buffer) >= self._buffer_size:
            # Schedule async flush - don't block
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._flush_async())
                else:
                    self.force_flush()
            except RuntimeError:
                pass

    def _build_flush_pipeline(self, pipe: Any, events: list[dict[str, Any]]) -> None:
        """Populate a Redis pipeline with flush commands for the given events."""
        for event in events:
            event_type = event.get("event_type", "unknown")
            trace_id = event.get("trace_id", "unknown")
            span_id = event.get("span_id")

            # Store in stream for replay
            stream_data = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in event.items()
            }
            pipe.xadd(self._stream_name, stream_data)

            # Store trace metadata in hash for quick lookup
            if event_type in ("trace_start", "trace_end"):
                hash_key = f"trace:{trace_id}"
                pipe.hset(
                    hash_key,
                    mapping={
                        "trace_id": trace_id,
                        "name": event.get("name", ""),
                        "started_at": str(event.get("started_at", "")),
                        "completed_at": str(event.get("completed_at", "")),
                        "error": str(event.get("error", "")),
                        "status": "completed" if event_type == "trace_end" else "running",
                    },
                )
                pipe.expire(hash_key, self._trace_ttl)

            # Store span in trace's span list
            if span_id and event_type in ("span_start", "span_end"):
                span_key = f"trace:{trace_id}:spans"
                span_data = json.dumps(
                    {
                        "span_id": span_id,
                        "parent_id": event.get("parent_id"),
                        "name": event.get("name", ""),
                        "span_type": event.get("span_type", ""),
                        "started_at": event.get("started_at"),
                        "finished_at": event.get("finished_at"),
                        "error": event.get("error"),
                        "span_data": event.get("span_data", {}),
                        "status": "completed" if event_type == "span_end" else "running",
                    }
                )
                pipe.hset(span_key, span_id, span_data)
                pipe.expire(span_key, self._trace_ttl)

    async def _flush_async_direct(self) -> None:
        """Flush using a fresh Redis connection (for threaded contexts)."""
        if not self._buffer:
            return

        events = self._buffer.copy()
        self._buffer.clear()

        client = redis.from_url(self._redis_url, decode_responses=True)
        try:
            pipe = client.pipeline()
            self._build_flush_pipeline(pipe, events)
            await pipe.execute()
        finally:
            await client.aclose()

    async def _flush_async(self) -> None:
        """Async implementation of buffer flush."""
        if not self._buffer or not self._client:
            return

        events = self._buffer.copy()
        self._buffer.clear()

        pipe = self._client.pipeline()
        self._build_flush_pipeline(pipe, events)
        await pipe.execute()

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Get trace data by ID.

        Args:
            trace_id: The trace ID to retrieve.

        Returns:
            Trace data dictionary or None if not found.
        """
        if not self._client:
            return None

        hash_key = f"trace:{trace_id}"
        data_result = await self._client.hgetall(hash_key)  # type: ignore[misc]
        data: dict[str, str] = data_result if isinstance(data_result, dict) else {}

        if not data:
            return None

        return {
            "trace_id": data.get("trace_id", trace_id),
            "name": data.get("name", ""),
            "started_at": float(data.get("started_at", 0)) if data.get("started_at") else None,
            "completed_at": float(data.get("completed_at", 0))
            if data.get("completed_at") and data.get("completed_at") != "None"
            else None,
            "error": data.get("error") if data.get("error") != "None" else None,
            "status": data.get("status", "unknown"),
        }

    async def get_spans(self, trace_id: str) -> list[dict[str, Any]]:
        """Get all spans for a trace.

        Args:
            trace_id: The trace ID to get spans for.

        Returns:
            List of span data dictionaries.
        """
        if not self._client:
            return []

        span_key = f"trace:{trace_id}:spans"
        data_result = await self._client.hgetall(span_key)  # type: ignore[misc]
        data: dict[str, str] = data_result if isinstance(data_result, dict) else {}

        if not data:
            return []

        spans = []
        for _span_id, span_json in data.items():
            try:
                span_data = json.loads(span_json)
                spans.append(span_data)
            except json.JSONDecodeError:
                continue

        return spans

    async def list_traces(
        self,
        limit: int = 100,
        name_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent traces.

        Args:
            limit: Maximum number of traces to return.
            name_filter: Optional filter by trace name substring.

        Returns:
            List of trace data dictionaries.
        """
        if not self._client:
            return []

        # Scan for trace keys
        traces = []
        cursor = 0

        while True:
            cursor, keys = await self._client.scan(
                cursor=cursor,
                match="trace:*",
                count=100,
            )

            for key in keys:
                # Skip span keys
                if ":spans" in key:
                    continue

                trace_id = key.replace("trace:", "")
                trace_data = await self.get_trace(trace_id)

                if trace_data:
                    # Apply name filter
                    if name_filter:
                        if name_filter.lower() not in trace_data.get("name", "").lower():
                            continue

                    traces.append(trace_data)

                if len(traces) >= limit:
                    break

            if cursor == 0 or len(traces) >= limit:
                break

        # Sort by started_at descending (most recent first)
        traces.sort(key=lambda x: x.get("started_at", 0) or 0, reverse=True)

        return traces[:limit]

    async def get_stream_length(self) -> int:
        """Get the length of the trace stream.

        Returns:
            Number of events in the stream.
        """
        if not self._client:
            return 0

        return int(await self._client.xlen(self._stream_name))

    async def trim_stream(self, max_length: int = 10000) -> int:
        """Trim the trace stream to a maximum length.

        Args:
            max_length: Maximum number of events to keep.

        Returns:
            Number of events trimmed.
        """
        if not self._client:
            return 0

        current_length = await self.get_stream_length()
        if current_length <= max_length:
            return 0

        # Trim using XTRIM MAXLEN
        await self._client.xtrim(self._stream_name, maxlen=max_length, approximate=True)

        return current_length - max_length
