"""AgentCoordinator - Streams-based coordination for distributed agents.

This module provides real-time coordination between distributed agent instances
using Redis Streams with consumer groups.

Key Features:
- Real-time handoff notifications
- Tool result broadcasting to multiple consumers
- State synchronization across replicas
- Crash recovery via XCLAIM for pending messages
- Consumer group support for work distribution
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from redis import asyncio as aioredis
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.asyncio import Redis


class EventType(str, Enum):
    """Standard event types for agent coordination."""

    HANDOFF_READY = "handoff_ready"
    TOOL_RESULT = "tool_result"
    STATE_CHANGED = "state_changed"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    ERROR = "error"


class AgentCoordinator:
    """
    Streams-based coordination for distributed agents.

    Enables:
    - Real-time handoff notifications (low latency vs polling)
    - Tool result broadcasting to multiple consumers
    - State synchronization across replicas
    - Crash recovery via pending message claiming

    Example:
        >>> coordinator = AgentCoordinator(
        ...     redis_url="redis://localhost:6379",
        ...     stream_name="agent_events",
        ...     consumer_group="workers",
        ... )
        >>> await coordinator.initialize()
        >>> await coordinator.publish_handoff_ready(
        ...     from_agent="research",
        ...     to_agent="analysis",
        ...     session_id="sess_123",
        ...     context={"data": "value"},
        ... )
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_name: str = "agent_events",
        consumer_group: str | None = None,
        consumer_name: str | None = None,
    ) -> None:
        """
        Initialize AgentCoordinator.

        Args:
            redis_url: Redis connection URL
            stream_name: Name of the Redis Stream
            consumer_group: Consumer group name (required for subscribing)
            consumer_name: Consumer name within group (auto-generated if not provided)
        """
        self._redis_url = redis_url
        self._stream_name = stream_name
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name or (
            f"consumer_{uuid4().hex[:8]}" if consumer_group else None
        )
        self._client: Redis | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the coordinator and create consumer group if needed.

        Must be called before using other methods.
        """
        if self._initialized:
            return

        self._client = aioredis.from_url(self._redis_url, decode_responses=True)

        # Create consumer group if specified
        if self._consumer_group:
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

    async def _publish(self, event_data: dict[str, Any]) -> str:
        """
        Publish event to stream.

        Args:
            event_data: Event data dictionary

        Returns:
            Message ID
        """
        client = await self._get_client()

        # Serialize complex values to JSON strings. Drop keys whose value is
        # None, since XADD rejects NoneType fields.
        serialized = {}
        for key, value in event_data.items():
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value)
            elif isinstance(value, (int, float)):
                serialized[key] = str(value)
            else:
                serialized[key] = value

        result = await client.xadd(self._stream_name, serialized)  # type: ignore[arg-type]
        return str(result)

    async def publish_handoff_ready(
        self,
        from_agent: str,
        to_agent: str,
        session_id: str,
        context: dict[str, Any],
    ) -> str:
        """
        Notify target agent that handoff is ready.

        Args:
            from_agent: Name of agent initiating handoff
            to_agent: Name of target agent
            session_id: Session identifier
            context: Handoff context data

        Returns:
            Message ID for tracking
        """
        return await self._publish(
            {
                "type": EventType.HANDOFF_READY.value,
                "from_agent": from_agent,
                "to_agent": to_agent,
                "session_id": session_id,
                "context": context,
                "timestamp": time.time(),
            }
        )

    async def publish_tool_result(
        self,
        tool_name: str,
        session_id: str,
        result: Any,
        execution_time_ms: float,
    ) -> str:
        """
        Broadcast tool completion to all interested consumers.

        Args:
            tool_name: Name of the tool
            session_id: Session identifier
            result: Tool execution result
            execution_time_ms: Execution time in milliseconds

        Returns:
            Message ID
        """
        return await self._publish(
            {
                "type": EventType.TOOL_RESULT.value,
                "tool_name": tool_name,
                "session_id": session_id,
                "result": result,
                "execution_time_ms": execution_time_ms,
                "timestamp": time.time(),
            }
        )

    async def publish_state_changed(
        self,
        session_id: str,
        changes: dict[str, Any],
    ) -> str:
        """
        Notify about session state changes.

        Args:
            session_id: Session identifier
            changes: Dictionary of changed fields

        Returns:
            Message ID
        """
        return await self._publish(
            {
                "type": EventType.STATE_CHANGED.value,
                "session_id": session_id,
                "changes": changes,
                "timestamp": time.time(),
            }
        )

    async def publish_agent_started(
        self,
        agent_name: str,
        session_id: str,
        input_summary: str,
    ) -> str:
        """
        Notify that an agent started processing.

        Args:
            agent_name: Name of the agent
            session_id: Session identifier
            input_summary: Summary of input

        Returns:
            Message ID
        """
        return await self._publish(
            {
                "type": EventType.AGENT_STARTED.value,
                "agent_name": agent_name,
                "session_id": session_id,
                "input_summary": input_summary,
                "timestamp": time.time(),
            }
        )

    async def publish_agent_completed(
        self,
        agent_name: str,
        session_id: str,
        output_summary: str,
        duration_ms: float,
        tokens_used: int,
    ) -> str:
        """
        Notify that an agent completed processing.

        Args:
            agent_name: Name of the agent
            session_id: Session identifier
            output_summary: Summary of output
            duration_ms: Processing duration
            tokens_used: Number of tokens used

        Returns:
            Message ID
        """
        return await self._publish(
            {
                "type": EventType.AGENT_COMPLETED.value,
                "agent_name": agent_name,
                "session_id": session_id,
                "output_summary": output_summary,
                "duration_ms": duration_ms,
                "tokens_used": tokens_used,
                "timestamp": time.time(),
            }
        )

    async def publish_error(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
        agent_name: str | None = None,
    ) -> str:
        """
        Publish error event.

        Args:
            session_id: Session identifier
            error_type: Type of error
            error_message: Error message
            agent_name: Optional agent name if error is agent-specific

        Returns:
            Message ID
        """
        return await self._publish(
            {
                "type": EventType.ERROR.value,
                "session_id": session_id,
                "error_type": error_type,
                "error_message": error_message,
                "agent_name": agent_name,
                "timestamp": time.time(),
            }
        )

    def _parse_event(self, data: dict[str, str]) -> dict[str, Any]:
        """Parse event data, deserializing JSON fields."""
        result = {}
        for key, value in data.items():
            # Try to parse JSON strings
            if isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value
            else:
                result[key] = value
        return result

    async def subscribe(
        self,
        event_types: list[str] | None = None,
        timeout_ms: int = 5000,
        max_events: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Subscribe to coordination events.

        Args:
            event_types: Filter to specific types (None = all)
            timeout_ms: Block timeout in milliseconds
            max_events: Maximum events to yield (None = unlimited)

        Yields:
            Event dictionaries with automatic acknowledgment
        """
        if not self._consumer_group:
            raise ValueError("Consumer group required for subscription")

        client = await self._get_client()
        events_yielded = 0

        while True:
            if max_events and events_yielded >= max_events:
                break

            messages = await client.xreadgroup(
                self._consumer_group,
                self._consumer_name or "",
                {self._stream_name: ">"},
                count=10,
                block=timeout_ms,
            )

            if not messages:
                if max_events:
                    break
                continue

            for _stream_name, events in messages:
                for msg_id, data in events:
                    event = self._parse_event(data)
                    event["_msg_id"] = msg_id

                    # Filter by type if specified
                    if event_types and event.get("type") not in event_types:
                        await client.xack(self._stream_name, self._consumer_group, msg_id)
                        continue

                    yield event

                    # Acknowledge after yielding
                    await client.xack(self._stream_name, self._consumer_group, msg_id)

                    events_yielded += 1
                    if max_events and events_yielded >= max_events:
                        return

    async def claim_abandoned_messages(
        self,
        min_idle_ms: int = 300000,  # 5 minutes
    ) -> list[dict[str, Any]]:
        """
        Claim messages from crashed consumers.

        Call periodically to recover from worker failures.

        Args:
            min_idle_ms: Minimum idle time to consider message abandoned

        Returns:
            List of claimed event dictionaries
        """
        if not self._consumer_group:
            raise ValueError("Consumer group required for claiming")

        client = await self._get_client()

        # Get pending messages
        pending = await client.xpending_range(
            self._stream_name,
            self._consumer_group,
            min="-",
            max="+",
            count=100,
        )

        # Filter to old messages only
        old_messages = [p for p in pending if p.get("time_since_delivered", 0) > min_idle_ms]

        if not old_messages:
            return []

        # Claim them for this consumer
        msg_ids = [p["message_id"] for p in old_messages]
        claimed = await client.xclaim(
            self._stream_name,
            self._consumer_group,
            self._consumer_name or "",
            min_idle_time=min_idle_ms,
            message_ids=msg_ids,
        )

        return [self._parse_event(data) for _, data in claimed]

    async def get_stream_info(self) -> dict[str, Any]:
        """
        Get stream statistics.

        Returns:
            Dictionary with stream length, groups, etc.
        """
        client = await self._get_client()

        try:
            stream_info = await client.xinfo_stream(self._stream_name)
            groups = await client.xinfo_groups(self._stream_name)
        except Exception as exc:
            logger.debug("get_stream_info failed (stream may not exist): %s", exc)
            return {"length": 0, "groups": []}

        return {
            "length": stream_info.get("length", 0),
            "first_entry": stream_info.get("first-entry"),
            "last_entry": stream_info.get("last-entry"),
            "groups": groups,
        }

    async def trim_stream(
        self,
        max_length: int,
        approximate: bool = True,
    ) -> int:
        """
        Trim stream to maximum length.

        Args:
            max_length: Maximum entries to keep
            approximate: Use approximate trimming (more efficient)

        Returns:
            Number of entries trimmed
        """
        client = await self._get_client()

        result = await client.xtrim(
            self._stream_name,
            maxlen=max_length,
            approximate=approximate,
        )
        return int(result)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
