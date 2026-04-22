"""RedisStreamTransport - Reliable token streaming via Redis Streams.

This module provides reliable event streaming using Redis Streams,
enabling real-time token delivery with replay capability.

Features:
- Reliable event delivery via Redis Streams
- Consumer groups for multiple clients
- Event replay from any point
- Automatic stream trimming
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Optional

from redis import Redis, ResponseError
from redis import asyncio as aioredis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .pool import RedisConnectionPool


class RedisStreamTransport:
    """Redis Streams-based event transport for token streaming.

    Provides reliable, replayable event streaming using Redis Streams.
    Supports consumer groups for multiple concurrent clients.

    Example:
        >>> stream = RedisStreamTransport(stream_name="agent_output")
        >>> stream.publish(event_type="token", data={"token": "Hello"})
        >>> events = stream.read_all(count=10)

    Args:
        stream_name: Name of the Redis Stream
        redis_url: Redis connection URL
        consumer_group: Consumer group name for reading
        max_len: Maximum stream length (older entries trimmed)
    """

    def __init__(
        self,
        stream_name: str,
        redis_url: str = "redis://localhost:6379",
        consumer_group: str = "agents",
        max_len: int = 10000,
        pool: Optional["RedisConnectionPool"] = None,
    ) -> None:
        """Initialize the stream transport.

        Args:
            stream_name: Name of the Redis Stream
            redis_url: Redis connection URL
            consumer_group: Consumer group name for reading
            max_len: Maximum stream length (older entries trimmed)
            pool: Optional shared connection pool
        """
        self._stream_name = stream_name
        self._consumer_group = consumer_group
        self._max_len = max_len
        self._pool = pool

        # Use pool's client if provided
        if pool is not None:
            self._redis_url = pool.redis_url
            self._client = pool.get_sync_client()
        else:
            self._redis_url = redis_url
            self._client = Redis.from_url(redis_url, decode_responses=True)

        # Create consumer group if it doesn't exist
        try:
            self._client.xgroup_create(
                stream_name,
                consumer_group,
                id="0",
                mkstream=True,
            )
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    @staticmethod
    def _parse_event(msg_id: str, fields: dict[str, str]) -> dict[str, Any]:
        """Parse raw stream fields into a structured event dict."""
        event: dict[str, Any] = {
            "id": msg_id,
            "type": fields.get("type", "unknown"),
            "timestamp": float(fields.get("timestamp", 0)),
        }

        data_str = fields.get("data", "{}")
        try:
            event["data"] = json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            event["data"] = {}

        metadata_str = fields.get("metadata")
        if metadata_str:
            try:
                event["metadata"] = json.loads(metadata_str)
            except (json.JSONDecodeError, TypeError):
                event["metadata"] = {}

        return event

    @property
    def stream_name(self) -> str:
        """Name of the Redis Stream."""
        return self._stream_name

    @property
    def consumer_group(self) -> str:
        """Consumer group name."""
        return self._consumer_group

    def publish(
        self,
        event_type: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Publish an event to the stream.

        Args:
            event_type: Type of event (token, tool_call, tool_result, complete)
            data: Event data
            metadata: Optional metadata

        Returns:
            Message ID from Redis
        """
        event = {
            "type": event_type,
            "timestamp": str(time.time()),
            "data": json.dumps(data),
        }
        if metadata:
            event["metadata"] = json.dumps(metadata)

        # Add to stream with automatic trimming
        msg_id: str = self._client.xadd(  # type: ignore[assignment]
            self._stream_name,
            event,  # type: ignore[arg-type]
            maxlen=self._max_len,
            approximate=True,
        )
        return msg_id

    def read_all(
        self,
        count: int = 100,
        start_id: str = "0",
    ) -> list[dict[str, Any]]:
        """Read all events from the stream.

        Args:
            count: Maximum number of events to read
            start_id: Start reading from this ID (default: beginning)

        Returns:
            List of events
        """
        try:
            # Use XRANGE to read all messages
            messages = self._client.xrange(
                self._stream_name,
                min=start_id,
                max="+",
                count=count,
            )
        except ResponseError:
            return []

        if not messages:
            return []

        events: list[dict[str, Any]] = []
        for msg_id, fields in messages:  # type: ignore[union-attr]
            events.append(self._parse_event(msg_id, fields))

        return events

    def info(self) -> dict[str, Any]:
        """Get stream information.

        Returns:
            Dictionary with stream info (length, groups, etc.)
        """
        try:
            # Get stream info
            info: dict[str, Any] = self._client.xinfo_stream(  # type: ignore[assignment]
                self._stream_name
            )
            groups_info: list[Any] = self._client.xinfo_groups(  # type: ignore[assignment]
                self._stream_name
            )

            return {
                "length": info.get("length", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": len(groups_info),
            }
        except ResponseError:
            return {
                "length": 0,
                "first_entry": None,
                "last_entry": None,
                "groups": 0,
            }

    def read_group(
        self,
        consumer: str,
        count: int = 100,
        block_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read events using consumer group (XREADGROUP).

        Events read but not ACKed are tracked as pending.
        Use ack() to acknowledge processed events.

        Args:
            consumer: Consumer name within the group
            count: Maximum number of events to read
            block_ms: Block for this many milliseconds waiting for events
                      (None = non-blocking)

        Returns:
            List of events
        """
        try:
            # XREADGROUP reads only new messages for this consumer
            streams: dict[str, str] = {self._stream_name: ">"}

            result = self._client.xreadgroup(
                groupname=self._consumer_group,
                consumername=consumer,
                streams=streams,  # type: ignore[arg-type]
                count=count,
                block=block_ms,
            )

            if not result:
                return []

            # Result format: [(stream_name, [(msg_id, {fields}), ...])]
            events: list[dict[str, Any]] = []
            for _stream_name, messages in result:  # type: ignore[union-attr]
                for msg_id, fields in messages:
                    events.append(self._parse_event(msg_id, fields))

            return events

        except ResponseError:
            return []

    def ack(self, ids: list[str]) -> int:
        """Acknowledge events as processed (XACK).

        ACKed events are removed from the pending list.

        Args:
            ids: List of message IDs to acknowledge

        Returns:
            Number of messages acknowledged
        """
        if not ids:
            return 0

        try:
            acked: int = self._client.xack(  # type: ignore[assignment]
                self._stream_name,
                self._consumer_group,
                *ids,
            )
            return acked
        except ResponseError:
            return 0

    def pending(self) -> dict[str, Any]:
        """Get information about pending messages (XPENDING).

        Returns:
            Dictionary with pending info:
            - count: Number of pending messages
            - min_id: Smallest pending message ID
            - max_id: Largest pending message ID
            - consumers: Dict mapping consumer names to pending counts
        """
        try:
            result = self._client.xpending(
                self._stream_name,
                self._consumer_group,
            )

            if not result:
                return {"count": 0, "min_id": None, "max_id": None, "consumers": {}}

            # redis-py returns a dict with keys: pending, min, max, consumers
            pending_count = result.get("pending", 0)  # type: ignore[union-attr]
            if pending_count == 0:
                return {"count": 0, "min_id": None, "max_id": None, "consumers": {}}

            consumers_list = result.get("consumers", [])  # type: ignore[union-attr]
            consumers: dict[str, int] = {}
            if consumers_list:
                for consumer_info in consumers_list:
                    consumer_name = consumer_info.get("name", "")
                    consumer_count = consumer_info.get("pending", 0)
                    if consumer_name:
                        consumers[consumer_name] = consumer_count

            return {
                "count": pending_count,
                "min_id": result.get("min"),  # type: ignore[union-attr]
                "max_id": result.get("max"),  # type: ignore[union-attr]
                "consumers": consumers,
            }

        except ResponseError:
            return {"count": 0, "min_id": None, "max_id": None, "consumers": {}}

    def claim(
        self,
        consumer: str,
        min_idle_ms: int,
        ids: list[str],
    ) -> list[dict[str, Any]]:
        """Claim pending messages from another consumer (XCLAIM).

        Used to recover messages from dead/stuck consumers.

        Args:
            consumer: Consumer to assign messages to
            min_idle_ms: Only claim messages idle for at least this long
            ids: Message IDs to claim

        Returns:
            List of claimed events
        """
        if not ids:
            return []

        try:
            result = self._client.xclaim(
                self._stream_name,
                self._consumer_group,
                consumer,
                min_idle_ms,
                ids,  # type: ignore[arg-type]
            )

            if not result:
                return []

            events: list[dict[str, Any]] = []
            for msg_id, fields in result:  # type: ignore[union-attr]
                if fields is None:
                    # Message doesn't exist or already ACKed
                    continue
                events.append(self._parse_event(msg_id, fields))

            return events

        except ResponseError:
            return []

    def delete(self) -> None:
        """Delete the stream and all its data."""
        try:
            self._client.delete(self._stream_name)
        except ResponseError:
            pass

    def close(self) -> None:
        """Close the Redis connection."""
        self._client.close()

    # Async methods

    def _get_async_redis(self) -> aioredis.Redis:
        """Get or create async Redis client."""
        if not hasattr(self, "_async_client"):
            self._async_client: aioredis.Redis = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._async_client

    async def apublish(
        self,
        event_type: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Async version of publish() - publish an event to the stream.

        Args:
            event_type: Type of event (token, tool_call, tool_result, complete)
            data: Event data
            metadata: Optional metadata

        Returns:
            Message ID from Redis
        """
        client = self._get_async_redis()

        event = {
            "type": event_type,
            "timestamp": str(time.time()),
            "data": json.dumps(data),
        }
        if metadata:
            event["metadata"] = json.dumps(metadata)

        msg_id = await client.xadd(
            self._stream_name,
            event,  # type: ignore[arg-type]
            maxlen=self._max_len,
            approximate=True,
        )
        return str(msg_id)

    async def asubscribe(
        self,
        last_id: str = "$",
        block_ms: int = 1000,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async generator that yields events from the stream.

        Args:
            last_id: Start from this ID ("$" = new events only, "0" = all)
            block_ms: Block timeout in milliseconds

        Yields:
            Event dictionaries
        """
        client = self._get_async_redis()
        current_id = last_id

        while True:
            try:
                result = await client.xread(
                    streams={self._stream_name: current_id},
                    block=block_ms,
                    count=10,
                )

                if not result:
                    continue

                for _stream_name, messages in result:
                    for msg_id, fields in messages:
                        current_id = msg_id
                        yield self._parse_event(msg_id, fields)

            except RedisError as exc:
                logger.warning("Transient error in asubscribe, retrying: %s", exc)
                await asyncio.sleep(0.5)
                continue
            except Exception as exc:
                logger.error("Fatal error in asubscribe, stopping: %s", exc)
                break
