"""Unit tests for Native JSON Session Storage.

These tests use mocks to verify behavior without Redis.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def create_mock_redis_client():
    """Create a properly configured mock Redis client.

    The redis-py async client has:
    - .json() - synchronous method returning JSON commands object
    - .json().get(), .json().set(), etc. - async methods
    """
    mock_client = AsyncMock()
    mock_json = MagicMock()

    # JSON methods are async
    mock_json.get = AsyncMock()
    mock_json.set = AsyncMock()
    mock_json.arrappend = AsyncMock()
    mock_json.numincrby = AsyncMock()

    # .json() is synchronous, returns the JSON commands object
    mock_client.json = MagicMock(return_value=mock_json)

    # Other async methods
    mock_client.expire = AsyncMock()
    mock_client.delete = AsyncMock()
    mock_client.aclose = AsyncMock()

    return mock_client, mock_json


class TestJSONSessionInit:
    """Tests for JSONSession initialization."""

    def test_init_sets_key_with_prefix(self) -> None:
        """Session key should use 'session:' prefix."""
        from redis_openai_agents.json_session import JSONSession

        session = JSONSession(
            session_id="abc123",
            user_id="user1",
            redis_url="redis://localhost:6379",
        )

        assert session._key == "session:abc123"

    def test_init_stores_user_id(self) -> None:
        """Should store user_id for document creation."""
        from redis_openai_agents.json_session import JSONSession

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )

        assert session._user_id == "test_user"

    def test_init_stores_ttl(self) -> None:
        """Should store TTL for expiration."""
        from redis_openai_agents.json_session import JSONSession

        session = JSONSession(
            session_id="abc123",
            user_id="user1",
            redis_url="redis://localhost:6379",
            ttl=3600,
        )

        assert session._ttl == 3600


class TestJSONSessionCreate:
    """Tests for session creation."""

    @pytest.mark.asyncio
    async def test_create_sets_json_document(self) -> None:
        """Create should set complete JSON document structure."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()

        with patch("redis_openai_agents.json_session.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            session = JSONSession(
                session_id="abc123",
                user_id="test_user",
                redis_url="redis://localhost:6379",
            )
            await session.create()

            # Verify JSON.SET was called with correct structure
            mock_json.set.assert_called_once()
            call_args = mock_json.set.call_args
            assert call_args[0][0] == "session:abc123"  # key
            assert call_args[0][1] == "$"  # path

            doc = call_args[0][2]
            assert doc["session_id"] == "abc123"
            assert doc["user_id"] == "test_user"
            assert doc["messages"] == []
            assert doc["metadata"]["message_count"] == 0
            assert doc["metadata"]["current_agent"] is None
            assert doc["metadata"]["agents_used"] == []
            assert doc["metadata"]["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_create_sets_ttl_when_provided(self) -> None:
        """Create should set expiration when TTL is provided."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()

        with patch("redis_openai_agents.json_session.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            session = JSONSession(
                session_id="abc123",
                user_id="test_user",
                redis_url="redis://localhost:6379",
                ttl=3600,
            )
            await session.create()

            mock_client.expire.assert_called_once_with("session:abc123", 3600)


class TestJSONSessionAddMessage:
    """Tests for adding messages."""

    @pytest.mark.asyncio
    async def test_add_message_uses_arrappend(self) -> None:
        """Should use JSON.ARRAPPEND for atomic append."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        await session.add_message(role="user", content="Hello")

        # Verify ARRAPPEND was called
        mock_json.arrappend.assert_called()
        call_args = mock_json.arrappend.call_args
        assert call_args[0][0] == "session:abc123"
        assert call_args[0][1] == "$.messages"

    @pytest.mark.asyncio
    async def test_add_message_increments_count(self) -> None:
        """Should use JSON.NUMINCRBY to increment message count."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        await session.add_message(role="user", content="Hello")

        # Verify NUMINCRBY was called for message_count
        calls = mock_json.numincrby.call_args_list
        count_call = [c for c in calls if "message_count" in str(c)]
        assert len(count_call) >= 1

    @pytest.mark.asyncio
    async def test_add_message_with_tokens_increments_total(self) -> None:
        """Should increment total_tokens when tokens provided."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        await session.add_message(role="user", content="Hello", tokens=50)

        # Verify NUMINCRBY was called for total_tokens
        calls = mock_json.numincrby.call_args_list
        token_call = [c for c in calls if "total_tokens" in str(c)]
        assert len(token_call) >= 1

    @pytest.mark.asyncio
    async def test_add_message_generates_id(self) -> None:
        """Each message should have a unique ID."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        await session.add_message(role="user", content="Hello")

        # Get the message that was appended
        call_args = mock_json.arrappend.call_args
        message = call_args[0][2]  # Third arg is the message

        assert "id" in message
        assert len(message["id"]) > 0


class TestJSONSessionGetMessages:
    """Tests for retrieving messages."""

    @pytest.mark.asyncio
    async def test_get_messages_uses_json_get(self) -> None:
        """Should use JSON.GET for retrieval."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()
        mock_json.get.return_value = [[{"role": "user", "content": "Hi"}]]

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        messages = await session.get_messages()

        mock_json.get.assert_called()
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_get_messages_with_limit_uses_slice(self) -> None:
        """Should use JSONPath slice for limit."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()
        mock_json.get.return_value = [[{"role": "user", "content": "Recent"}]]

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        await session.get_messages(limit=5)

        # Verify path includes slice notation
        call_args = mock_json.get.call_args
        path = call_args[0][1]
        assert "-5" in path

    @pytest.mark.asyncio
    async def test_get_messages_with_role_filter(self) -> None:
        """Should use JSONPath filter for role."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()
        mock_json.get.return_value = [[{"role": "user", "content": "Hi"}]]

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        await session.get_messages(role="user")

        # Verify path includes role filter
        call_args = mock_json.get.call_args
        path = call_args[0][1]
        assert "user" in path


class TestJSONSessionTrackAgent:
    """Tests for agent tracking."""

    @pytest.mark.asyncio
    async def test_track_agent_sets_current(self) -> None:
        """Should set current_agent using JSON.SET."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()
        mock_json.get.return_value = [[]]  # Empty agents_used

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        await session.track_agent("my_agent")

        # Verify SET was called for current_agent
        calls = mock_json.set.call_args_list
        current_agent_calls = [c for c in calls if "current_agent" in str(c)]
        assert len(current_agent_calls) >= 1


class TestJSONSessionMetadata:
    """Tests for metadata retrieval."""

    @pytest.mark.asyncio
    async def test_get_metadata_retrieves_nested_fields(self) -> None:
        """Should retrieve metadata fields efficiently."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()
        mock_json.get.return_value = {
            "$.user_id": ["test_user"],
            "$.session_id": ["abc123"],
            "$.metadata": [
                {
                    "message_count": 5,
                    "current_agent": "agent1",
                    "agents_used": ["agent1"],
                    "total_tokens": 100,
                }
            ],
            "$.created_at": [1234567890.0],
            "$.updated_at": [1234567891.0],
        }

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        metadata = await session.get_metadata()

        assert metadata["user_id"] == "test_user"
        assert metadata["message_count"] == 5


class TestJSONSessionLoad:
    """Tests for loading existing sessions."""

    @pytest.mark.asyncio
    async def test_load_checks_existence(self) -> None:
        """Load should verify session exists."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()
        mock_json.get.return_value = None  # Session doesn't exist

        with patch("redis_openai_agents.json_session.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            with pytest.raises(ValueError, match="not found"):
                await JSONSession.load(
                    session_id="nonexistent",
                    redis_url="redis://localhost:6379",
                )


class TestJSONSessionClear:
    """Tests for clearing session."""

    @pytest.mark.asyncio
    async def test_clear_resets_messages_array(self) -> None:
        """Clear should set messages to empty array."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        await session.clear()

        # Verify messages was set to empty array
        calls = mock_json.set.call_args_list
        messages_clear = [c for c in calls if "messages" in str(c)]
        assert len(messages_clear) >= 1
        # Check the value is empty list
        for call in messages_clear:
            if "$.messages" in str(call[0]):
                assert call[0][2] == []


class TestJSONSessionDelete:
    """Tests for deleting session."""

    @pytest.mark.asyncio
    async def test_delete_removes_key(self) -> None:
        """Delete should remove the Redis key."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        await session.delete()

        mock_client.delete.assert_called_once_with("session:abc123")


class TestJSONSessionToAgentInputs:
    """Tests for SDK format conversion."""

    @pytest.mark.asyncio
    async def test_to_agent_inputs_extracts_role_content(self) -> None:
        """Should extract only role and content for SDK format."""
        from redis_openai_agents.json_session import JSONSession

        mock_client, mock_json = create_mock_redis_client()
        mock_json.get.return_value = [
            [
                {"id": "1", "role": "user", "content": "Hello", "timestamp": 123},
                {"id": "2", "role": "assistant", "content": "Hi", "timestamp": 124},
            ]
        ]

        session = JSONSession(
            session_id="abc123",
            user_id="test_user",
            redis_url="redis://localhost:6379",
        )
        session._client = mock_client

        inputs = await session.to_agent_inputs()

        assert len(inputs) == 2
        assert inputs[0] == {"role": "user", "content": "Hello"}
        assert inputs[1] == {"role": "assistant", "content": "Hi"}
        # Should not include id, timestamp, etc.
        assert "id" not in inputs[0]
        assert "timestamp" not in inputs[0]
