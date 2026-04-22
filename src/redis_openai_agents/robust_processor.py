"""RobustStreamProcessor - Fault-tolerant stream processing with DLQ.

This module provides enhanced stream processing with automatic recovery features:
- Automatic pending message recovery via XCLAIM
- Dead-letter queue (DLQ) for failed messages after max retries
- Processing timeout detection
- Health statistics and monitoring

Key Features:
- Crash recovery: Automatically claims pending messages from crashed consumers
- DLQ: Messages that fail repeatedly are moved to a dead-letter queue
- Replay: Failed messages can be replayed back to the main stream
- Observability: Health stats for monitoring stream processor status
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from redis import asyncio as aioredis
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.asyncio import Redis


class RobustStreamProcessor:
    """
    Fault-tolerant stream processor with DLQ support.

    Features:
    - Automatic pending message recovery (XCLAIM)
    - Dead-letter queue for failed messages
    - Processing timeout detection
    - Health statistics

    Example:
        >>> processor = RobustStreamProcessor(
        ...     redis_url="redis://localhost:6379",
        ...     stream_name="agent_events",
        ...     consumer_group="workers",
        ... )
        >>> await processor.initialize()
        >>>
        >>> async def handle_message(msg: dict) -> bool:
        ...     # Process message, return True on success
        ...     return True
        >>>
        >>> await processor.process_with_recovery(handle_message)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_name: str = "agent_events",
        consumer_group: str = "workers",
        consumer_name: str | None = None,
        dlq_stream: str | None = None,
        max_retries: int = 3,
        claim_timeout_ms: int = 300000,  # 5 minutes
    ) -> None:
        """
        Initialize RobustStreamProcessor.

        Args:
            redis_url: Redis connection URL
            stream_name: Name of the main Redis Stream
            consumer_group: Consumer group name
            consumer_name: Consumer name (auto-generated if not provided)
            dlq_stream: Dead-letter queue stream name (default: {stream_name}:dlq)
            max_retries: Maximum delivery attempts before moving to DLQ
            claim_timeout_ms: Idle time (ms) before claiming pending messages
        """
        self._redis_url = redis_url
        self._stream_name = stream_name
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name or f"consumer_{uuid4().hex[:8]}"
        self._dlq_stream = dlq_stream or f"{stream_name}:dlq"
        self._max_retries = max_retries
        self._claim_timeout_ms = claim_timeout_ms
        self._client: Redis | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the processor and create consumer group if needed.

        Must be called before using other methods.
        """
        if self._initialized:
            return

        self._client = aioredis.from_url(self._redis_url, decode_responses=True)

        # Create consumer group if not exists
        try:
            await self._client.xgroup_create(
                self._stream_name,
                self._consumer_group,
                id="0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            # Group already exists

        self._initialized = True

    async def _get_client(self) -> Redis:
        """Get Redis client, ensuring initialization."""
        if not self._initialized or self._client is None:
            await self.initialize()
        return self._client  # type: ignore[return-value]

    async def process_batch(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[bool]],
        batch_size: int = 10,
        block_ms: int = 5000,
        max_batches: int | None = None,
    ) -> int:
        """
        Process a batch of messages from the stream.

        Args:
            handler: Async function to process each message. Return True on success.
            batch_size: Number of messages to read per batch
            block_ms: Blocking timeout in milliseconds
            max_batches: Maximum batches to process (None = unlimited)

        Returns:
            Number of messages successfully processed
        """
        client = await self._get_client()
        processed = 0
        batches = 0

        while max_batches is None or batches < max_batches:
            # First, try to claim any abandoned messages
            await self.claim_pending_messages()

            # Read new messages
            messages = await client.xreadgroup(
                self._consumer_group,
                self._consumer_name,
                {self._stream_name: ">"},
                count=batch_size,
                block=block_ms,
            )

            if not messages:
                batches += 1
                continue

            for _stream_name, events in messages:
                for msg_id, data in events:
                    success = await self._process_message(msg_id, data, handler)
                    if success:
                        await client.xack(self._stream_name, self._consumer_group, msg_id)
                        processed += 1
                    # If not success, message stays pending for retry

            batches += 1

        return processed

    async def process_with_recovery(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[bool]],
        batch_size: int = 10,
        block_ms: int = 5000,
    ) -> None:
        """
        Process stream with automatic failure recovery.

        This method runs indefinitely:
        1. Claims abandoned messages from crashed consumers
        2. Processes new messages
        3. Moves failed messages to DLQ after max retries

        Args:
            handler: Async function to process each message
            batch_size: Number of messages per batch
            block_ms: Blocking timeout
        """
        await self.process_batch(handler, batch_size, block_ms, max_batches=None)

    async def _process_message(
        self,
        msg_id: str,
        data: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[bool]],
    ) -> bool:
        """Process a single message with error handling."""
        try:
            return await handler(data)
        except Exception as e:
            # Log error but don't ACK - will be retried
            logger.error("Error processing %s: %s", msg_id, e)
            return False

    async def claim_pending_messages(self) -> int:
        """
        Claim messages from crashed/slow consumers.

        Messages older than claim_timeout_ms are claimed for reprocessing.
        Messages exceeding max_retries are moved to DLQ.

        Returns:
            Number of messages claimed for reprocessing
        """
        client = await self._get_client()

        # Get pending messages
        pending = await client.xpending_range(
            self._stream_name,
            self._consumer_group,
            min="-",
            max="+",
            count=100,
        )

        claimed = 0
        for entry in pending:
            idle_time = entry.get("time_since_delivered", 0)
            times_delivered = entry.get("times_delivered", 0)
            msg_id = entry.get("message_id")

            if idle_time < self._claim_timeout_ms:
                # Message not old enough to claim
                continue

            if times_delivered >= self._max_retries:
                # Move to DLQ
                await self._move_to_dlq(
                    msg_id,
                    reason="max_retries_exceeded",
                    attempts=times_delivered,
                )
            else:
                # Claim for reprocessing
                await client.xclaim(
                    self._stream_name,
                    self._consumer_group,
                    self._consumer_name,
                    self._claim_timeout_ms,
                    [msg_id],
                )
                claimed += 1

        return claimed

    async def _move_to_dlq(
        self,
        msg_id: str,
        reason: str,
        attempts: int,
    ) -> None:
        """Move failed message to dead-letter queue."""
        client = await self._get_client()

        # Get original message
        messages = await client.xrange(self._stream_name, msg_id, msg_id)

        if messages:
            _, data = messages[0]

            # Add to DLQ with metadata
            await client.xadd(
                self._dlq_stream,
                {
                    **data,
                    "original_stream": self._stream_name,
                    "original_id": msg_id,
                    "failure_reason": reason,
                    "attempts": str(attempts),
                    "dlq_timestamp": str(time.time()),
                },
            )

        # Acknowledge original (remove from pending)
        await client.xack(self._stream_name, self._consumer_group, msg_id)

    async def get_dlq_messages(
        self,
        count: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get messages from dead-letter queue for inspection.

        Args:
            count: Maximum number of messages to retrieve

        Returns:
            List of DLQ messages with id and data
        """
        client = await self._get_client()

        messages = await client.xrange(self._dlq_stream, "-", "+", count=count)

        return [{"id": mid, **data} for mid, data in messages]

    async def replay_dlq_message(
        self,
        dlq_message_id: str,
    ) -> str:
        """
        Replay a DLQ message back to main stream.

        Call this after fixing the root cause of the failure.

        Args:
            dlq_message_id: Message ID in the DLQ

        Returns:
            New message ID in main stream

        Raises:
            ValueError: If DLQ message not found
        """
        client = await self._get_client()

        # Get DLQ message
        messages = await client.xrange(self._dlq_stream, dlq_message_id, dlq_message_id)

        if not messages:
            raise ValueError(f"DLQ message not found: {dlq_message_id}")

        _, data = messages[0]

        # Remove DLQ metadata fields
        dlq_fields = {
            "original_stream",
            "original_id",
            "failure_reason",
            "attempts",
            "dlq_timestamp",
        }
        replay_data = {k: v for k, v in data.items() if k not in dlq_fields}
        replay_data["replayed_from_dlq"] = dlq_message_id

        # Add back to main stream
        new_id = await client.xadd(self._stream_name, replay_data)

        # Remove from DLQ
        await client.xdel(self._dlq_stream, dlq_message_id)

        return str(new_id)

    async def get_health_stats(self) -> dict[str, Any]:
        """
        Get processor health statistics.

        Returns:
            Dictionary with:
            - stream_length: Total messages in stream
            - pending_messages: Messages awaiting processing
            - consumers: Number of consumers in group
            - dlq_length: Messages in dead-letter queue
            - last_delivered_id: Last delivered message ID
        """
        client = await self._get_client()

        try:
            # Stream info
            stream_info = await client.xinfo_stream(self._stream_name)

            # Group info
            groups = await client.xinfo_groups(self._stream_name)
            group_info: dict[str, Any] = next(
                (g for g in groups if g["name"] == self._consumer_group),
                {},
            )

            # DLQ count
            dlq_len = await client.xlen(self._dlq_stream)

            return {
                "stream_length": stream_info.get("length", 0),
                "pending_messages": group_info.get("pending", 0),
                "consumers": group_info.get("consumers", 0),
                "dlq_length": dlq_len,
                "last_delivered_id": group_info.get("last-delivered-id"),
            }
        except Exception as exc:
            logger.debug("get_health_stats failed (stream may not exist): %s", exc)
            return {
                "stream_length": 0,
                "pending_messages": 0,
                "consumers": 0,
                "dlq_length": 0,
                "last_delivered_id": None,
            }

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
