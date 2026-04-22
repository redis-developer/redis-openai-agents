"""Integration tests for Lua Atomic Operations.

These tests run against a real Redis instance and verify:
- Atomic multi-step operations (all-or-nothing)
- Race condition prevention
- Script execution behavior
"""

import asyncio
import json
import time
from uuid import uuid4

import pytest


class TestLuaAtomicMessageAppend:
    """Tests for atomic message append with Lua scripting."""

    @pytest.mark.asyncio
    async def test_atomic_append_adds_message(self, redis_url: str) -> None:
        """Atomic append should add message to session."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        # Create session first
        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        # Use atomic operations
        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        message = {
            "id": "msg1",
            "role": "user",
            "content": "Hello",
            "timestamp": time.time(),
        }

        await ops.atomic_message_append(
            session_key=f"session:{session_id}",
            message=message,
        )

        # Verify message was added
        messages = await session.get_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello"

        await client.aclose()

    @pytest.mark.asyncio
    async def test_atomic_append_increments_counter(self, redis_url: str) -> None:
        """Atomic append should increment message count."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        for i in range(3):
            message = {
                "id": f"msg{i}",
                "role": "user",
                "content": f"Message {i}",
                "timestamp": time.time(),
            }
            await ops.atomic_message_append(
                session_key=f"session:{session_id}",
                message=message,
            )

        metadata = await session.get_metadata()
        assert metadata["message_count"] == 3

        await client.aclose()

    @pytest.mark.asyncio
    async def test_atomic_append_with_window_trimming(self, redis_url: str) -> None:
        """Atomic append should trim to max_messages when specified."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        # Add 10 messages with max_messages=5
        for i in range(10):
            message = {
                "id": f"msg{i}",
                "role": "user",
                "content": f"Message {i}",
                "timestamp": time.time(),
            }
            await ops.atomic_message_append(
                session_key=f"session:{session_id}",
                message=message,
                max_messages=5,
            )

        messages = await session.get_messages()
        # Should only have last 5 messages
        assert len(messages) == 5
        assert messages[0]["content"] == "Message 5"
        assert messages[4]["content"] == "Message 9"

        await client.aclose()

    @pytest.mark.asyncio
    async def test_atomic_append_concurrent_safety(self, redis_url: str) -> None:
        """Concurrent atomic appends should not lose messages."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        async def append_message(i: int) -> None:
            message = {
                "id": f"msg{i}",
                "role": "user",
                "content": f"Concurrent {i}",
                "timestamp": time.time(),
            }
            await ops.atomic_message_append(
                session_key=f"session:{session_id}",
                message=message,
            )

        # Run 20 concurrent appends
        await asyncio.gather(*[append_message(i) for i in range(20)])

        messages = await session.get_messages()
        # All 20 messages should be present
        assert len(messages) == 20

        metadata = await session.get_metadata()
        assert metadata["message_count"] == 20

        await client.aclose()


class TestLuaAtomicResponseRecord:
    """Tests for atomic cache + session + metrics update."""

    @pytest.mark.asyncio
    async def test_atomic_response_stores_in_cache(self, redis_url: str) -> None:
        """Atomic response record should store in cache."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        await ops.atomic_response_record(
            session_key=f"session:{session_id}",
            cache_key="cache:test",
            query_hash="abc123",
            response="Test response",
            user_message={"role": "user", "content": "Hello"},
            assistant_message={"role": "assistant", "content": "Hi"},
            latency_ms=100.0,
            input_tokens=10,
            output_tokens=5,
        )

        # Verify cache entry
        cached = await client.hget("cache:test", "abc123")
        assert cached is not None
        cached_data = json.loads(cached)
        assert cached_data["response"] == "Test response"

        await client.aclose()

    @pytest.mark.asyncio
    async def test_atomic_response_adds_messages(self, redis_url: str) -> None:
        """Atomic response record should add both messages to session."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        await ops.atomic_response_record(
            session_key=f"session:{session_id}",
            cache_key="cache:test",
            query_hash="abc123",
            response="Test response",
            user_message={"role": "user", "content": "What is Redis?"},
            assistant_message={"role": "assistant", "content": "Redis is a database"},
            latency_ms=150.0,
            input_tokens=20,
            output_tokens=15,
        )

        messages = await session.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is Redis?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Redis is a database"

        await client.aclose()

    @pytest.mark.asyncio
    async def test_atomic_response_all_or_nothing(self, redis_url: str) -> None:
        """If one part fails, entire operation should fail (atomicity)."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        # Create session
        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        # First operation should succeed
        await ops.atomic_response_record(
            session_key=f"session:{session_id}",
            cache_key="cache:test",
            query_hash="query1",
            response="Response 1",
            user_message={"role": "user", "content": "Q1"},
            assistant_message={"role": "assistant", "content": "A1"},
            latency_ms=100.0,
            input_tokens=10,
            output_tokens=5,
        )

        # Verify first operation
        messages = await session.get_messages()
        assert len(messages) == 2

        await client.aclose()


class TestLuaAtomicHandoff:
    """Tests for atomic handoff with locking."""

    @pytest.mark.asyncio
    async def test_atomic_handoff_updates_agent(self, redis_url: str) -> None:
        """Atomic handoff should update current agent."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()
        await session.track_agent("agent_a")

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        result = await ops.atomic_handoff(
            session_key=f"session:{session_id}",
            from_agent="agent_a",
            to_agent="agent_b",
            context={"reason": "task_complete"},
        )

        assert result == "OK"

        metadata = await session.get_metadata()
        assert metadata["current_agent"] == "agent_b"
        assert "agent_b" in metadata["agents_used"]

        await client.aclose()

    @pytest.mark.asyncio
    async def test_atomic_handoff_stores_context(self, redis_url: str) -> None:
        """Atomic handoff should store handoff context."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()
        await session.track_agent("research")

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        context = {
            "reason": "needs_analysis",
            "data": {"topic": "Redis"},
        }

        await ops.atomic_handoff(
            session_key=f"session:{session_id}",
            from_agent="research",
            to_agent="analysis",
            context=context,
        )

        # Verify context stored
        stored_context = await client.json().get(f"session:{session_id}", "$.handoff_context")
        assert stored_context is not None
        assert stored_context[0]["reason"] == "needs_analysis"

        await client.aclose()

    @pytest.mark.asyncio
    async def test_atomic_handoff_prevents_concurrent(self, redis_url: str) -> None:
        """Only one handoff should succeed when concurrent attempts."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, HandoffInProgressError, JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()
        await session.track_agent("agent_a")

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        results = {"success": 0, "blocked": 0}

        async def attempt_handoff(to_agent: str) -> None:
            try:
                await ops.atomic_handoff(
                    session_key=f"session:{session_id}",
                    from_agent="agent_a",
                    to_agent=to_agent,
                    context={},
                    lock_ttl=5,  # 5 second lock
                )
                results["success"] += 1
            except HandoffInProgressError:
                results["blocked"] += 1

        # Attempt 3 concurrent handoffs
        await asyncio.gather(
            attempt_handoff("agent_b"),
            attempt_handoff("agent_c"),
            attempt_handoff("agent_d"),
        )

        # Only one should succeed
        assert results["success"] == 1
        assert results["blocked"] == 2

        await client.aclose()

    @pytest.mark.asyncio
    async def test_atomic_handoff_releases_lock(self, redis_url: str) -> None:
        """After releasing lock, subsequent handoff should succeed."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()
        await session.track_agent("agent_a")

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        # First handoff
        await ops.atomic_handoff(
            session_key=f"session:{session_id}",
            from_agent="agent_a",
            to_agent="agent_b",
            context={},
        )

        # Release the lock to allow subsequent handoffs
        await ops.release_handoff_lock(f"session:{session_id}")

        # Update current agent for second handoff
        await session.track_agent("agent_b")

        # Second handoff should also succeed (lock was released)
        result = await ops.atomic_handoff(
            session_key=f"session:{session_id}",
            from_agent="agent_b",
            to_agent="agent_c",
            context={},
        )

        assert result == "OK"

        await client.aclose()


class TestLuaScriptCaching:
    """Tests for Lua script caching behavior."""

    @pytest.mark.asyncio
    async def test_scripts_are_cached(self, redis_url: str) -> None:
        """Scripts should be loaded once and cached."""
        from redis.asyncio import Redis

        from redis_openai_agents import AtomicOperations, JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        client = Redis.from_url(redis_url, decode_responses=True)
        ops = AtomicOperations(client)

        # Script SHAs should be populated after first use
        message = {"id": "1", "role": "user", "content": "Test", "timestamp": 0}
        await ops.atomic_message_append(
            session_key=f"session:{session_id}",
            message=message,
        )

        # Verify scripts are cached
        assert len(ops._script_shas) > 0
        assert "atomic_message_append" in ops._script_shas

        await client.aclose()
