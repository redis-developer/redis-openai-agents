"""Integration tests for Native JSON Session Storage.

These tests run against a real Redis instance and verify:
- Native RedisJSON storage (not JSON-as-string)
- Atomic operations (no race conditions)
- Partial updates (no read-modify-write)
- Server-side queries on nested fields
"""

import asyncio
from uuid import uuid4

import pytest


class TestJSONSessionCreation:
    """Tests for session creation with native JSON storage."""

    @pytest.mark.asyncio
    async def test_create_session_stores_json_document(self, redis_url: str) -> None:
        """Session should be stored as native JSON, not serialized string."""
        from redis import Redis

        from redis_openai_agents import JSONSession

        session_id = uuid4().hex[:16]
        user_id = "test_user"

        session = JSONSession(
            session_id=session_id,
            user_id=user_id,
            redis_url=redis_url,
        )
        await session.create()

        # Verify storage is native JSON (not string)
        client = Redis.from_url(redis_url, decode_responses=True)
        key = f"session:{session_id}"

        # JSON.TYPE should return 'object' for root
        doc_type = client.json().type(key, "$")
        assert doc_type is not None, "Document should exist as JSON"
        assert doc_type[0] == "object", "Root should be JSON object"

        # Should be able to query nested paths
        user = client.json().get(key, "$.user_id")
        assert user == [user_id], "Should retrieve user_id via JSONPath"

        client.close()

    @pytest.mark.asyncio
    async def test_create_session_initializes_structure(self, redis_url: str) -> None:
        """Session should have proper initial structure."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        metadata = await session.get_metadata()

        assert metadata["user_id"] == "test_user"
        assert metadata["message_count"] == 0
        assert metadata["current_agent"] is None
        assert metadata["agents_used"] == []
        assert "created_at" in metadata
        assert "updated_at" in metadata

    @pytest.mark.asyncio
    async def test_create_session_with_ttl(self, redis_url: str) -> None:
        """Session with TTL should set expiration on Redis key."""
        from redis import Redis

        from redis_openai_agents import JSONSession

        session_id = uuid4().hex[:16]

        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
            ttl=60,  # 60 seconds
        )
        await session.create()

        client = Redis.from_url(redis_url, decode_responses=True)
        ttl = client.ttl(f"session:{session_id}")
        assert ttl > 0, "TTL should be set"
        assert ttl <= 60, "TTL should be <= 60 seconds"

        client.close()


class TestJSONSessionMessages:
    """Tests for message operations with native JSON."""

    @pytest.mark.asyncio
    async def test_add_message_uses_atomic_append(self, redis_url: str) -> None:
        """Adding message should use JSON.ARRAPPEND (atomic, not read-modify-write)."""
        from redis import Redis

        from redis_openai_agents import JSONSession

        session_id = uuid4().hex[:16]
        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        await session.add_message(role="user", content="Hello")

        # Verify message was added
        client = Redis.from_url(redis_url, decode_responses=True)
        messages = client.json().get(f"session:{session_id}", "$.messages")

        assert len(messages[0]) == 1
        assert messages[0][0]["role"] == "user"
        assert messages[0][0]["content"] == "Hello"

        client.close()

    @pytest.mark.asyncio
    async def test_add_message_increments_counter_atomically(self, redis_url: str) -> None:
        """Message count should be incremented atomically with JSON.NUMINCRBY."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        await session.add_message(role="user", content="Message 1")
        await session.add_message(role="assistant", content="Response 1")
        await session.add_message(role="user", content="Message 2")

        metadata = await session.get_metadata()
        assert metadata["message_count"] == 3

    @pytest.mark.asyncio
    async def test_add_message_tracks_tokens(self, redis_url: str) -> None:
        """Token counts should be tracked when provided."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        await session.add_message(role="user", content="Hello", tokens=10)
        await session.add_message(role="assistant", content="Hi there", tokens=5)

        metadata = await session.get_metadata()
        assert metadata["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_get_messages_returns_all(self, redis_url: str) -> None:
        """Should retrieve all messages from session."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        await session.add_message(role="user", content="First")
        await session.add_message(role="assistant", content="Second")
        await session.add_message(role="user", content="Third")

        messages = await session.get_messages()

        assert len(messages) == 3
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"
        assert messages[2]["content"] == "Third"

    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self, redis_url: str) -> None:
        """Should retrieve only the last N messages."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        for i in range(10):
            await session.add_message(role="user", content=f"Message {i}")

        messages = await session.get_messages(limit=3)

        assert len(messages) == 3
        # Should be last 3 messages
        assert messages[0]["content"] == "Message 7"
        assert messages[1]["content"] == "Message 8"
        assert messages[2]["content"] == "Message 9"

    @pytest.mark.asyncio
    async def test_get_messages_filtered_by_role(self, redis_url: str) -> None:
        """Should filter messages by role using server-side JSONPath."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        await session.add_message(role="user", content="User 1")
        await session.add_message(role="assistant", content="Assistant 1")
        await session.add_message(role="user", content="User 2")
        await session.add_message(role="assistant", content="Assistant 2")

        user_messages = await session.get_messages(role="user")

        assert len(user_messages) == 2
        assert all(m["role"] == "user" for m in user_messages)


class TestJSONSessionAgentTracking:
    """Tests for agent tracking with native JSON."""

    @pytest.mark.asyncio
    async def test_track_agent_updates_current(self, redis_url: str) -> None:
        """Tracking agent should update current_agent field."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        await session.track_agent("research_agent")

        metadata = await session.get_metadata()
        assert metadata["current_agent"] == "research_agent"

    @pytest.mark.asyncio
    async def test_track_agent_adds_to_used_list(self, redis_url: str) -> None:
        """Tracking agent should add to agents_used list."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        await session.track_agent("agent_a")
        await session.track_agent("agent_b")
        await session.track_agent("agent_a")  # Duplicate

        metadata = await session.get_metadata()
        # Should deduplicate
        assert set(metadata["agents_used"]) == {"agent_a", "agent_b"}


class TestJSONSessionConcurrency:
    """Tests for concurrent access safety."""

    @pytest.mark.asyncio
    async def test_concurrent_message_adds_no_race_condition(self, redis_url: str) -> None:
        """Multiple concurrent message adds should not lose messages."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        # Simulate 10 concurrent message adds
        async def add_msg(i: int) -> None:
            await session.add_message(role="user", content=f"Message {i}")

        await asyncio.gather(*[add_msg(i) for i in range(10)])

        # All messages should be present (no lost due to race)
        messages = await session.get_messages()
        assert len(messages) == 10

        metadata = await session.get_metadata()
        assert metadata["message_count"] == 10

    @pytest.mark.asyncio
    async def test_concurrent_token_counting(self, redis_url: str) -> None:
        """Concurrent token additions should sum correctly."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()

        # 10 concurrent adds of 100 tokens each
        async def add_msg_with_tokens(i: int) -> None:
            await session.add_message(role="user", content=f"Msg {i}", tokens=100)

        await asyncio.gather(*[add_msg_with_tokens(i) for i in range(10)])

        metadata = await session.get_metadata()
        assert metadata["total_tokens"] == 1000  # 10 * 100


class TestJSONSessionLoad:
    """Tests for loading existing sessions."""

    @pytest.mark.asyncio
    async def test_load_existing_session(self, redis_url: str) -> None:
        """Should load an existing session with all data intact."""
        from redis_openai_agents import JSONSession

        session_id = uuid4().hex[:16]

        # Create and populate
        session1 = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session1.create()
        await session1.add_message(role="user", content="Hello")
        await session1.add_message(role="assistant", content="Hi")
        await session1.track_agent("my_agent")

        # Load in new instance
        session2 = await JSONSession.load(
            session_id=session_id,
            redis_url=redis_url,
        )

        messages = await session2.get_messages()
        metadata = await session2.get_metadata()

        assert len(messages) == 2
        assert metadata["user_id"] == "test_user"
        assert metadata["current_agent"] == "my_agent"

    @pytest.mark.asyncio
    async def test_load_nonexistent_session_raises(self, redis_url: str) -> None:
        """Loading non-existent session should raise error."""
        from redis_openai_agents import JSONSession

        with pytest.raises(ValueError, match="not found"):
            await JSONSession.load(
                session_id="nonexistent_session",
                redis_url=redis_url,
            )


class TestJSONSessionClear:
    """Tests for session clearing and deletion."""

    @pytest.mark.asyncio
    async def test_clear_removes_messages_keeps_metadata(self, redis_url: str) -> None:
        """Clear should remove messages but keep session structure."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()
        await session.add_message(role="user", content="Hello")
        await session.add_message(role="assistant", content="Hi")

        await session.clear()

        messages = await session.get_messages()
        assert len(messages) == 0

        metadata = await session.get_metadata()
        assert metadata["user_id"] == "test_user"
        assert metadata["message_count"] == 0

    @pytest.mark.asyncio
    async def test_delete_removes_entire_session(self, redis_url: str) -> None:
        """Delete should remove the entire session from Redis."""
        from redis import Redis

        from redis_openai_agents import JSONSession

        session_id = uuid4().hex[:16]

        session = JSONSession(
            session_id=session_id,
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()
        await session.add_message(role="user", content="Hello")

        await session.delete()

        # Key should not exist
        client = Redis.from_url(redis_url, decode_responses=True)
        exists = client.exists(f"session:{session_id}")
        assert exists == 0

        client.close()


class TestJSONSessionAgentInputs:
    """Tests for converting to OpenAI Agents SDK input format."""

    @pytest.mark.asyncio
    async def test_to_agent_inputs_format(self, redis_url: str) -> None:
        """Should convert messages to SDK-compatible format."""
        from redis_openai_agents import JSONSession

        session = JSONSession(
            session_id=uuid4().hex[:16],
            user_id="test_user",
            redis_url=redis_url,
        )
        await session.create()
        await session.add_message(role="user", content="Hello")
        await session.add_message(role="assistant", content="Hi there")

        inputs = await session.to_agent_inputs()

        assert len(inputs) == 2
        assert inputs[0] == {"role": "user", "content": "Hello"}
        assert inputs[1] == {"role": "assistant", "content": "Hi there"}
