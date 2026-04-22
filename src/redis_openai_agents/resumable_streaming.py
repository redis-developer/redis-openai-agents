"""ResumableStreamRunner - Resumable LLM streaming backed by Redis Streams.

This module provides resumable streaming for LLM responses, allowing clients
to disconnect and reconnect without losing data. Events are stored durably
in Redis Streams and can be replayed from any point.

Features:
- Durable event storage in Redis Streams
- Resumable subscriptions from any message ID
- Consumer groups for per-client progress tracking
- Automatic stream trimming with configurable max length
- Multi-client support (same stream, different progress)

Example:
    >>> from redis_openai_agents import ResumableStreamRunner
    >>>
    >>> runner = ResumableStreamRunner(redis_url="redis://localhost:6379")
    >>> await runner.initialize()
    >>>
    >>> # Producer publishes events
    >>> session_id = "chat_123"
    >>> await runner.publish_event(session_id, "text_delta", {"delta": "Hello"})
    >>> await runner.publish_event(session_id, "text_delta", {"delta": " world"})
    >>>
    >>> # Consumer subscribes (can reconnect anytime)
    >>> async for event in runner.subscribe(session_id, from_id="0"):
    ...     print(event["data"]["delta"], end="")
    Hello world
"""

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis


@dataclass
class StreamEvent:
    """A streaming event with metadata.

    Attributes:
        id: Redis Stream message ID.
        type: Event type (e.g., "text_delta", "tool_call").
        data: Event payload.
        timestamp: Unix timestamp when event was published.
    """

    id: str
    type: str
    data: dict
    timestamp: float

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class ResumableStreamRunner:
    """Resumable LLM streaming backed by Redis Streams.

    Enables durable, resumable streaming of LLM responses. Events are
    published to Redis Streams and can be consumed by multiple clients,
    each tracking their own progress.

    The key insight is separating generation from consumption:
    - Generation continues even if clients disconnect
    - Clients can reconnect and resume from where they left off
    - Multiple clients can consume the same stream independently

    Attributes:
        stream_prefix: Prefix for Redis Stream keys.
        max_stream_length: Maximum events per stream (for trimming).
        consumer_group: Default consumer group name.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_prefix: str = "llm_stream",
        max_stream_length: int | None = None,
        consumer_group: str = "consumers",
    ) -> None:
        """Initialize the resumable stream runner.

        Args:
            redis_url: Redis connection URL.
            stream_prefix: Prefix for stream keys.
            max_stream_length: Max events per stream (None = unlimited).
            consumer_group: Default consumer group name.
        """
        self._redis_url = redis_url
        self._stream_prefix = stream_prefix
        self._max_stream_length = max_stream_length
        self._consumer_group = consumer_group

        self._client: redis.Redis | None = None

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        self._client = redis.from_url(self._redis_url, decode_responses=True)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_stream_key(self, session_id: str) -> str:
        """Get Redis Stream key for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Full Redis key for the stream.
        """
        return f"{self._stream_prefix}:{session_id}"

    async def publish_event(
        self,
        session_id: str,
        event_type: str,
        data: dict,
        metadata: dict | None = None,
    ) -> str:
        """Publish a streaming event to Redis.

        Args:
            session_id: Session identifier for the stream.
            event_type: Type of event (e.g., "text_delta", "tool_call").
            data: Event payload data.
            metadata: Optional additional metadata.

        Returns:
            Redis Stream message ID.
        """
        if not self._client:
            raise RuntimeError("Runner not initialized. Call initialize() first.")

        stream_key = self._get_stream_key(session_id)
        timestamp = time.time()

        # Build event fields
        fields = {
            "type": event_type,
            "data": json.dumps(data),
            "timestamp": str(timestamp),
        }
        if metadata:
            fields["metadata"] = json.dumps(metadata)

        # Publish to stream with optional trimming
        if self._max_stream_length:
            msg_id = await self._client.xadd(
                stream_key,
                fields,  # type: ignore[arg-type]
                maxlen=self._max_stream_length,
                approximate=False,
            )
        else:
            msg_id = await self._client.xadd(stream_key, fields)  # type: ignore[arg-type]

        return str(msg_id)

    async def get_all_events(self, session_id: str) -> list[dict]:
        """Get all events from a stream.

        Args:
            session_id: Session identifier.

        Returns:
            List of events with id, type, data, and timestamp.
        """
        if not self._client:
            return []

        stream_key = self._get_stream_key(session_id)

        try:
            messages = await self._client.xrange(stream_key, "-", "+")
        except redis.ResponseError:
            return []

        events = []
        for msg_id, fields in messages:
            event = self._parse_message(msg_id, fields)
            events.append(event)

        return events

    def _parse_message(self, msg_id: str, fields: dict) -> dict:
        """Parse a Redis Stream message into an event dict.

        Args:
            msg_id: Redis Stream message ID.
            fields: Message fields from Redis.

        Returns:
            Parsed event dictionary.
        """
        data = json.loads(fields.get("data", "{}"))
        timestamp = float(fields.get("timestamp", 0))

        event = {
            "id": msg_id,
            "type": fields.get("type", "unknown"),
            "data": data,
            "timestamp": timestamp,
        }

        if "metadata" in fields:
            event["metadata"] = json.loads(fields["metadata"])

        return event

    async def subscribe(
        self,
        session_id: str,
        from_id: str = "$",
        timeout_ms: int = 5000,
        count: int = 100,
    ) -> AsyncIterator[dict]:
        """Subscribe to streaming events.

        Yields events from the stream, starting from the specified ID.
        Use from_id="0" to start from the beginning, or "$" for new events only.

        Args:
            session_id: Session identifier.
            from_id: Message ID to start from ("0" = beginning, "$" = new only).
            timeout_ms: Block timeout in milliseconds (0 = don't block).
            count: Max events per read.

        Yields:
            Event dictionaries with id, type, data, and timestamp.
        """
        if not self._client:
            return

        stream_key = self._get_stream_key(session_id)
        last_id = from_id

        # First, read any existing messages if starting from beginning or specific ID
        if from_id != "$":
            try:
                messages = await self._client.xrange(
                    stream_key,
                    "(" + from_id if from_id != "0" else "-",
                    "+",
                    count=count,
                )
                for msg_id, fields in messages:
                    event = self._parse_message(msg_id, fields)
                    yield event
                    last_id = msg_id
            except redis.ResponseError:
                pass

        # Then block for new messages
        if timeout_ms > 0:
            try:
                result = await self._client.xread(
                    {stream_key: last_id},
                    count=count,
                    block=timeout_ms,
                )
                if result:
                    for _, messages in result:
                        for msg_id, fields in messages:
                            event = self._parse_message(msg_id, fields)
                            yield event
            except redis.ResponseError:
                pass

    async def subscribe_as_consumer(
        self,
        session_id: str,
        consumer_id: str,
        timeout_ms: int = 5000,
        count: int = 100,
    ) -> AsyncIterator[dict]:
        """Subscribe as a consumer in a consumer group.

        Uses Redis consumer groups to track which messages each consumer
        has processed. Consumers must call ack() after processing each message.

        Args:
            session_id: Session identifier.
            consumer_id: Unique consumer identifier.
            timeout_ms: Block timeout in milliseconds.
            count: Max events per read.

        Yields:
            Event dictionaries with id, type, data, and timestamp.
        """
        if not self._client:
            return

        stream_key = self._get_stream_key(session_id)
        group_name = f"{self._consumer_group}:{session_id}"

        # Ensure consumer group exists
        try:
            await self._client.xgroup_create(
                stream_key,
                group_name,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        # First, read any pending messages for this consumer
        try:
            pending = await self._client.xreadgroup(
                group_name,
                consumer_id,
                {stream_key: "0"},
                count=count,
            )
            if pending:
                for _, messages in pending:
                    for msg_id, fields in messages:
                        if fields:  # Skip deleted messages
                            event = self._parse_message(msg_id, fields)
                            yield event
        except redis.ResponseError:
            pass

        # Then read new messages
        try:
            result = await self._client.xreadgroup(
                group_name,
                consumer_id,
                {stream_key: ">"},
                count=count,
                block=timeout_ms,
            )
            if result:
                for _, messages in result:
                    for msg_id, fields in messages:
                        event = self._parse_message(msg_id, fields)
                        yield event
        except redis.ResponseError:
            pass

    async def ack(
        self,
        session_id: str,
        consumer_id: str,
        message_id: str,
    ) -> int:
        """Acknowledge a message as processed.

        After acknowledging, the message won't be redelivered to this consumer.

        Args:
            session_id: Session identifier.
            consumer_id: Consumer identifier (for group name lookup).
            message_id: Message ID to acknowledge.

        Returns:
            Number of messages acknowledged (0 or 1).
        """
        if not self._client:
            return 0

        stream_key = self._get_stream_key(session_id)
        group_name = f"{self._consumer_group}:{session_id}"

        return int(await self._client.xack(stream_key, group_name, message_id))

    async def get_stream_info(self, session_id: str) -> dict:
        """Get information about a stream.

        Args:
            session_id: Session identifier.

        Returns:
            Dictionary with stream info (length, first/last entry IDs, etc.).
        """
        if not self._client:
            return {}

        stream_key = self._get_stream_key(session_id)

        try:
            info = await self._client.xinfo_stream(stream_key)
            return {
                "length": info.get("length", 0),
                "first_entry_id": info.get("first-entry", [None])[0]
                if info.get("first-entry")
                else None,
                "last_entry_id": info.get("last-entry", [None])[0]
                if info.get("last-entry")
                else None,
                "groups": info.get("groups", 0),
            }
        except redis.ResponseError:
            return {"length": 0}

    async def delete_stream(self, session_id: str) -> bool:
        """Delete a stream and all its messages.

        Args:
            session_id: Session identifier.

        Returns:
            True if stream was deleted, False otherwise.
        """
        if not self._client:
            return False

        stream_key = self._get_stream_key(session_id)
        result = await self._client.delete(stream_key)
        return int(result) > 0

    async def get_pending_count(
        self,
        session_id: str,
        consumer_id: str | None = None,
    ) -> int:
        """Get count of pending (unacknowledged) messages.

        Args:
            session_id: Session identifier.
            consumer_id: Optional consumer ID to filter by.

        Returns:
            Number of pending messages.
        """
        if not self._client:
            return 0

        stream_key = self._get_stream_key(session_id)
        group_name = f"{self._consumer_group}:{session_id}"

        try:
            info = await self._client.xpending(stream_key, group_name)
            if not info or (isinstance(info, (list, tuple)) and info[0] == 0):
                return 0
            return int(info[0]) if isinstance(info, (list, tuple)) else 0
        except redis.ResponseError:
            return 0

    async def claim_pending(
        self,
        session_id: str,
        consumer_id: str,
        min_idle_time_ms: int = 60000,
        count: int = 10,
    ) -> list[dict]:
        """Claim pending messages from dead consumers.

        Transfers ownership of messages that have been pending longer than
        min_idle_time_ms to the specified consumer.

        Args:
            session_id: Session identifier.
            consumer_id: Consumer to claim messages for.
            min_idle_time_ms: Minimum idle time to claim.
            count: Max messages to claim.

        Returns:
            List of claimed message events.
        """
        if not self._client:
            return []

        stream_key = self._get_stream_key(session_id)
        group_name = f"{self._consumer_group}:{session_id}"

        try:
            # Get pending message IDs
            pending = await self._client.xpending_range(
                stream_key,
                group_name,
                "-",
                "+",
                count,
            )
            if not pending:
                return []

            # Filter by idle time and claim
            message_ids = [
                p["message_id"]
                for p in pending
                if p.get("time_since_delivered", 0) >= min_idle_time_ms
            ]

            if not message_ids:
                return []

            claimed = await self._client.xclaim(
                stream_key,
                group_name,
                consumer_id,
                min_idle_time_ms,
                message_ids,
            )

            return [self._parse_message(msg_id, fields) for msg_id, fields in claimed if fields]
        except redis.ResponseError:
            return []


class StreamingEventPublisher:
    """High-level helper for publishing SDK streaming events.

    Provides convenient methods for publishing common event types
    from the OpenAI Agents SDK streaming interface.

    Example:
        >>> publisher = StreamingEventPublisher(runner, session_id="chat_123")
        >>>
        >>> # Publish text deltas as they arrive
        >>> async for event in sdk_stream_events:
        ...     if isinstance(event, RawResponsesStreamEvent):
        ...         await publisher.publish_raw_event(event.data)
        ...     elif isinstance(event, RunItemStreamEvent):
        ...         await publisher.publish_item_event(event.name, event.item)
    """

    def __init__(
        self,
        runner: ResumableStreamRunner,
        session_id: str,
    ) -> None:
        """Initialize the publisher.

        Args:
            runner: ResumableStreamRunner instance.
            session_id: Session identifier for this stream.
        """
        self._runner = runner
        self._session_id = session_id
        self._message_count = 0

    async def publish_text_delta(self, delta: str) -> str:
        """Publish a text delta event.

        Args:
            delta: Text content to publish.

        Returns:
            Message ID.
        """
        return await self._runner.publish_event(
            session_id=self._session_id,
            event_type="text_delta",
            data={"delta": delta},
        )

    async def publish_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        call_id: str | None = None,
    ) -> str:
        """Publish a tool call event.

        Args:
            tool_name: Name of the tool being called.
            arguments: Tool arguments.
            call_id: Optional tool call ID.

        Returns:
            Message ID.
        """
        data = {
            "tool": tool_name,
            "arguments": arguments,
        }
        if call_id:
            data["call_id"] = call_id

        return await self._runner.publish_event(
            session_id=self._session_id,
            event_type="tool_call",
            data=data,
        )

    async def publish_tool_result(
        self,
        tool_name: str,
        result: Any,
        call_id: str | None = None,
    ) -> str:
        """Publish a tool result event.

        Args:
            tool_name: Name of the tool.
            result: Tool execution result.
            call_id: Optional tool call ID.

        Returns:
            Message ID.
        """
        data = {
            "tool": tool_name,
            "result": result,
        }
        if call_id:
            data["call_id"] = call_id

        return await self._runner.publish_event(
            session_id=self._session_id,
            event_type="tool_result",
            data=data,
        )

    async def publish_stream_start(
        self,
        agent_name: str,
        metadata: dict | None = None,
    ) -> str:
        """Publish stream start event.

        Args:
            agent_name: Name of the agent starting.
            metadata: Optional additional metadata.

        Returns:
            Message ID.
        """
        return await self._runner.publish_event(
            session_id=self._session_id,
            event_type="stream_start",
            data={"agent": agent_name},
            metadata=metadata,
        )

    async def publish_stream_end(
        self,
        reason: str = "complete",
        metadata: dict | None = None,
    ) -> str:
        """Publish stream end event.

        Args:
            reason: Reason for ending (e.g., "complete", "error", "cancelled").
            metadata: Optional additional metadata.

        Returns:
            Message ID.
        """
        return await self._runner.publish_event(
            session_id=self._session_id,
            event_type="stream_end",
            data={"reason": reason},
            metadata=metadata,
        )

    async def publish_handoff(
        self,
        from_agent: str,
        to_agent: str,
    ) -> str:
        """Publish agent handoff event.

        Args:
            from_agent: Agent handing off.
            to_agent: Agent receiving handoff.

        Returns:
            Message ID.
        """
        return await self._runner.publish_event(
            session_id=self._session_id,
            event_type="handoff",
            data={
                "from_agent": from_agent,
                "to_agent": to_agent,
            },
        )

    async def publish_error(
        self,
        error_type: str,
        message: str,
        details: dict | None = None,
    ) -> str:
        """Publish error event.

        Args:
            error_type: Type of error.
            message: Error message.
            details: Optional error details.

        Returns:
            Message ID.
        """
        data: dict[str, Any] = {
            "error_type": error_type,
            "message": message,
        }
        if details:
            data["details"] = details

        return await self._runner.publish_event(
            session_id=self._session_id,
            event_type="error",
            data=data,
        )
