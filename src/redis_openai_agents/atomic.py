"""AtomicOperations - Lua script-based atomic operations for Redis.

This module provides atomic multi-step operations using Lua scripting,
ensuring consistency without distributed locks for simple operations.

Key Features:
- Atomic message append with metadata update
- Atomic response recording (cache + session + metrics)
- Atomic agent handoff with distributed locking
- Script caching via EVALSHA for performance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis import Redis

# Directory containing Lua scripts
SCRIPTS_DIR = Path(__file__).parent / "lua_scripts"


class HandoffInProgressError(Exception):
    """Raised when a handoff is already in progress for a session."""

    pass


class AtomicOperations:
    """
    Lua script-based atomic operations for Redis.

    Uses SCRIPT LOAD + EVALSHA pattern for efficient script execution.
    Scripts are loaded lazily on first use and cached by SHA.

    Example:
        >>> from redis.asyncio import Redis
        >>> client = Redis.from_url("redis://localhost:6379")
        >>> ops = AtomicOperations(client)
        >>> await ops.atomic_message_append(
        ...     session_key="session:abc",
        ...     message={"role": "user", "content": "Hello"},
        ... )
    """

    # Class-level script content cache (loaded once from files)
    _script_contents: dict[str, str] = {}

    def __init__(self, client: Redis) -> None:
        """
        Initialize AtomicOperations with a Redis client.

        Args:
            client: Redis client (async)
        """
        self._client = client
        self._script_shas: dict[str, str] = {}
        self._scripts_loaded = False
        self._load_script_contents()

    def _load_script_contents(self) -> None:
        """Load Lua script contents from files (class-level cache)."""
        if AtomicOperations._script_contents:
            return

        script_files = [
            "atomic_message_append",
            "atomic_response_record",
            "atomic_handoff",
        ]

        for script_name in script_files:
            script_path = SCRIPTS_DIR / f"{script_name}.lua"
            if script_path.exists():
                AtomicOperations._script_contents[script_name] = script_path.read_text()

    async def _ensure_scripts_loaded(self) -> None:
        """Lazily load scripts into Redis and cache SHAs."""
        if self._scripts_loaded:
            return

        for script_name, script_content in AtomicOperations._script_contents.items():
            sha = await self._client.script_load(script_content)
            self._script_shas[script_name] = sha

        self._scripts_loaded = True

    async def atomic_message_append(
        self,
        session_key: str,
        message: dict[str, Any],
        max_messages: int | None = None,
        ttl: int | None = None,
    ) -> int:
        """
        Atomically append a message to a session.

        Performs in a single atomic operation:
        - Append message to messages array
        - Increment message_count
        - Update timestamp
        - Trim to max_messages if specified
        - Refresh TTL if specified

        Args:
            session_key: Redis key for the session
            message: Message dict with role, content, etc.
            max_messages: Maximum messages to keep (sliding window)
            ttl: TTL in seconds for the session key

        Returns:
            New message count
        """
        await self._ensure_scripts_loaded()
        sha = self._script_shas["atomic_message_append"]
        message_json = json.dumps(message)

        result = await self._client.evalsha(  # type: ignore[misc]
            sha,
            1,  # number of keys
            session_key,
            message_json,
            str(max_messages or 0),
            str(ttl or 0),
        )

        # Result from JSON.NUMINCRBY can be JSON array string like '[1]'
        if isinstance(result, str):
            parsed = json.loads(result)
            return int(parsed[0]) if isinstance(parsed, list) else int(parsed)
        return int(result[0]) if isinstance(result, list) else int(result)

    async def atomic_response_record(
        self,
        session_key: str,
        cache_key: str,
        query_hash: str,
        response: str,
        user_message: dict[str, Any],
        assistant_message: dict[str, Any],
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cache_ttl: int | None = None,
        max_messages: int | None = None,
    ) -> str:
        """
        Atomically record a response with cache, session, and metrics update.

        Performs in a single atomic operation:
        - Store response in cache hash
        - Append user message to session
        - Append assistant message to session
        - Update message count and token totals
        - Update timestamp
        - Trim messages if needed

        Args:
            session_key: Redis key for the session
            cache_key: Redis key for the cache hash
            query_hash: Hash of the query for cache lookup
            response: Response text to cache
            user_message: User message dict
            assistant_message: Assistant message dict
            latency_ms: Response latency in milliseconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_ttl: TTL for cache entry
            max_messages: Maximum messages to keep

        Returns:
            "OK" on success
        """
        await self._ensure_scripts_loaded()
        sha = self._script_shas["atomic_response_record"]

        result = await self._client.evalsha(  # type: ignore[misc]
            sha,
            2,  # number of keys
            session_key,
            cache_key,
            query_hash,
            response,
            json.dumps(user_message),
            json.dumps(assistant_message),
            str(latency_ms),
            str(input_tokens),
            str(output_tokens),
            str(cache_ttl or 0),
            str(max_messages or 0),
        )

        return str(result)

    async def atomic_handoff(
        self,
        session_key: str,
        from_agent: str,
        to_agent: str,
        context: dict[str, Any],
        lock_ttl: int | None = None,
    ) -> str:
        """
        Atomically perform an agent handoff with distributed locking.

        Performs in a single atomic operation:
        - Acquire handoff lock (prevents concurrent handoffs)
        - Update current_agent
        - Add to_agent to agents_used
        - Store handoff context
        - Update timestamp
        - Release lock

        Args:
            session_key: Redis key for the session
            from_agent: Name of the agent handing off
            to_agent: Name of the agent receiving handoff
            context: Handoff context data
            lock_ttl: Lock timeout in seconds (default 30)

        Returns:
            "OK" on success

        Raises:
            HandoffInProgressError: If another handoff is in progress
        """
        await self._ensure_scripts_loaded()
        sha = self._script_shas["atomic_handoff"]
        lock_key = f"{session_key}:handoff_lock"

        result = await self._client.evalsha(  # type: ignore[misc]
            sha,
            2,  # number of keys
            session_key,
            lock_key,
            from_agent,
            to_agent,
            json.dumps(context),
            str(lock_ttl or 30),
        )

        # Check for error response
        if isinstance(result, dict) and result.get("err") == "HANDOFF_IN_PROGRESS":
            raise HandoffInProgressError(f"Handoff already in progress for session {session_key}")

        # Handle JSON-encoded error response
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and parsed.get("err") == "HANDOFF_IN_PROGRESS":
                    raise HandoffInProgressError(
                        f"Handoff already in progress for session {session_key}"
                    )
            except json.JSONDecodeError:
                pass

        return str(result)

    async def release_handoff_lock(self, session_key: str) -> bool:
        """
        Release the handoff lock for a session.

        Call this after handoff processing is complete to allow
        subsequent handoffs.

        Args:
            session_key: Redis key for the session

        Returns:
            True if lock was released, False if no lock existed
        """
        lock_key = f"{session_key}:handoff_lock"
        result = await self._client.delete(lock_key)
        return int(result) > 0
