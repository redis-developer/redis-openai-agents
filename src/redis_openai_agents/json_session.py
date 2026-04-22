"""JSONSession - Native RedisJSON session storage for OpenAI Agents SDK.

This module provides session/conversation state management using native
RedisJSON operations, avoiding JSON-as-string anti-pattern.

Key Features:
- Native JSON storage (not serialized strings)
- Atomic operations (JSON.ARRAPPEND, JSON.NUMINCRBY)
- Partial updates (no read-modify-write cycles)
- Server-side queries via JSONPath
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from redis import asyncio as aioredis

if TYPE_CHECKING:
    from redis.asyncio import Redis


class JSONSession:
    """
    Native RedisJSON-based session storage for OpenAI Agents SDK.

    Uses JSON.* commands for atomic operations:
    - JSON.ARRAPPEND for message storage (no race conditions)
    - JSON.NUMINCRBY for counters (atomic increments)
    - JSON.SET for partial field updates
    - JSONPath queries for server-side filtering

    Example:
        >>> session = JSONSession(
        ...     session_id="abc123",
        ...     user_id="user_1",
        ...     redis_url="redis://localhost:6379",
        ... )
        >>> await session.create()
        >>> await session.add_message(role="user", content="Hello")
        >>> messages = await session.get_messages()
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        redis_url: str = "redis://localhost:6379",
        ttl: int | None = None,
    ) -> None:
        """
        Initialize a JSONSession.

        Args:
            session_id: Unique session identifier
            user_id: User identifier
            redis_url: Redis connection URL
            ttl: Time-to-live in seconds (None = no expiration)
        """
        self._session_id = session_id
        self._user_id = user_id
        self._redis_url = redis_url
        self._ttl = ttl
        self._key = f"session:{session_id}"
        self._client: Redis | None = None

    async def _get_client(self) -> Redis:
        """Get or create async Redis client."""
        if self._client is None:
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def create(self) -> None:
        """
        Create a new session document in Redis.

        Creates a native JSON document with proper structure:
        - session_id, user_id at root
        - messages array (initially empty)
        - metadata object with counters
        - timestamps
        """
        client = await self._get_client()
        now = time.time()

        doc = {
            "session_id": self._session_id,
            "user_id": self._user_id,
            "created_at": now,
            "updated_at": now,
            "metadata": {
                "current_agent": None,
                "agents_used": [],
                "message_count": 0,
                "total_tokens": 0,
            },
            "messages": [],
            "handoff_context": None,
        }

        await client.json().set(self._key, "$", doc)  # type: ignore[misc, arg-type]

        if self._ttl:
            await client.expire(self._key, self._ttl)

    @classmethod
    async def load(
        cls,
        session_id: str,
        redis_url: str = "redis://localhost:6379",
    ) -> JSONSession:
        """
        Load an existing session from Redis.

        Args:
            session_id: Session ID to load
            redis_url: Redis connection URL

        Returns:
            Loaded JSONSession instance

        Raises:
            ValueError: If session not found
        """
        client = aioredis.from_url(redis_url, decode_responses=True)

        # Check if session exists
        exists = await client.json().get(f"session:{session_id}", "$.user_id")  # type: ignore[misc]

        if not exists:
            await client.aclose()
            raise ValueError(f"Session not found: {session_id}")

        user_id = exists[0] if exists else "unknown"

        session = cls(
            session_id=session_id,
            user_id=user_id,
            redis_url=redis_url,
        )
        session._client = client

        return session

    async def add_message(
        self,
        role: str,
        content: str,
        agent: str | None = None,
        tokens: int | None = None,
    ) -> None:
        """
        Add a message to the session using atomic JSON operations.

        Uses JSON.ARRAPPEND (atomic, no race conditions) and
        JSON.NUMINCRBY (atomic counter increment).

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            agent: Optional agent name that produced this message
            tokens: Optional token count for this message
        """
        client = await self._get_client()

        message = {
            "id": uuid4().hex[:16],
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "agent": agent,
            "tokens": tokens,
        }

        # Atomic append - no read-modify-write!
        await client.json().arrappend(self._key, "$.messages", message)  # type: ignore[misc, arg-type]

        # Atomic counter increment
        await client.json().numincrby(self._key, "$.metadata.message_count", 1)  # type: ignore[misc]

        if tokens:
            await client.json().numincrby(self._key, "$.metadata.total_tokens", tokens)  # type: ignore[misc]

        # Update timestamp
        await client.json().set(self._key, "$.updated_at", time.time())  # type: ignore[misc]

        # Refresh TTL if configured
        if self._ttl:
            await client.expire(self._key, self._ttl)

    async def get_messages(
        self,
        limit: int | None = None,
        role: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get messages from the session with optional filtering.

        Filtering happens server-side via JSONPath - no need to
        transfer all data and filter in Python.

        Args:
            limit: Number of recent messages to retrieve (None = all)
            role: Filter by role (user, assistant, system)

        Returns:
            List of message dictionaries
        """
        client = await self._get_client()

        if role:
            # Server-side filter by role using JSONPath
            path = f'$.messages[?(@.role == "{role}")]'
        elif limit:
            # Last N messages using JSONPath slice
            path = f"$.messages[-{limit}:]"
        else:
            path = "$.messages"

        result = await client.json().get(self._key, path)  # type: ignore[misc]

        if result is None:
            return []

        # Result is nested in array from JSONPath
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                return result[0]
            return result

        return []

    async def track_agent(self, agent_name: str) -> None:
        """
        Track agent usage in this session.

        Updates current_agent and adds to agents_used list
        (deduplicating to avoid repeats).

        Args:
            agent_name: Name of the agent being used
        """
        client = await self._get_client()

        # Set current agent
        await client.json().set(  # type: ignore[misc]
            self._key,
            "$.metadata.current_agent",
            agent_name,
        )

        # Check if agent already in list
        current_agents = await client.json().get(self._key, "$.metadata.agents_used")  # type: ignore[misc]

        if current_agents and isinstance(current_agents, list):
            agents_list = current_agents[0] if current_agents else []
            if agent_name not in agents_list:
                await client.json().arrappend(  # type: ignore[misc]
                    self._key,
                    "$.metadata.agents_used",
                    agent_name,
                )

        # Update timestamp
        await client.json().set(self._key, "$.updated_at", time.time())  # type: ignore[misc]

    async def get_metadata(self) -> dict[str, Any]:
        """
        Get session metadata (not full messages).

        Efficiently retrieves only metadata fields.

        Returns:
            Dictionary with session metadata
        """
        client = await self._get_client()

        # Get multiple paths in one call
        result = await client.json().get(  # type: ignore[misc]
            self._key,
            "$.user_id",
            "$.session_id",
            "$.metadata",
            "$.created_at",
            "$.updated_at",
        )

        if not result:
            return {}

        # Parse the multi-path result
        # The result format varies based on redis-py version
        if isinstance(result, dict):
            # Multi-path returns dict keyed by path
            metadata = result.get("$.metadata", [{}])[0] if result.get("$.metadata") else {}
            return {
                "user_id": result.get("$.user_id", [None])[0],
                "session_id": result.get("$.session_id", [None])[0],
                "created_at": result.get("$.created_at", [None])[0],
                "updated_at": result.get("$.updated_at", [None])[0],
                **metadata,
            }
        elif isinstance(result, list):
            # Single path returns list
            if len(result) > 0 and isinstance(result[0], dict):
                return result[0]

        return {}

    async def to_agent_inputs(self) -> list[dict[str, str]]:
        """
        Convert session messages to OpenAI Agents SDK input format.

        Extracts only role and content fields for SDK compatibility.

        Returns:
            List of messages in format [{"role": str, "content": str}, ...]
        """
        messages = await self.get_messages()
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    async def clear(self) -> None:
        """
        Clear all messages from the session.

        Keeps session structure but resets messages and counters.
        """
        client = await self._get_client()

        # Reset messages array
        await client.json().set(self._key, "$.messages", [])  # type: ignore[misc]

        # Reset counters
        await client.json().set(self._key, "$.metadata.message_count", 0)  # type: ignore[misc]
        await client.json().set(self._key, "$.metadata.total_tokens", 0)  # type: ignore[misc]

        # Update timestamp
        await client.json().set(self._key, "$.updated_at", time.time())  # type: ignore[misc]

    async def delete(self) -> None:
        """
        Delete the entire session from Redis.
        """
        client = await self._get_client()
        await client.delete(self._key)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    @property
    def user_id(self) -> str:
        """Get the user ID."""
        return self._user_id
