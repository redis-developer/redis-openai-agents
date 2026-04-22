"""Unit tests for AgentSession that don't require OpenAI API."""

import uuid

from redis import Redis

from redis_openai_agents import AgentSession


class TestSessionUnit:
    """Unit tests for basic session operations without API calls."""

    def test_create_session(self, redis_url: str) -> None:
        """Test creating a new session."""
        session = AgentSession.create(user_id="test_user", redis_url=redis_url)

        assert session.user_id == "test_user"
        assert session.conversation_id is not None
        assert len(session.conversation_id) > 0

    def test_session_with_custom_id(self, redis_url: str) -> None:
        """Test creating session with custom conversation ID."""
        custom_id = str(uuid.uuid4().hex[:16])
        session = AgentSession(user_id="test_user", conversation_id=custom_id, redis_url=redis_url)

        assert session.conversation_id == custom_id
        assert session.user_id == "test_user"

    def test_add_and_retrieve_messages(self, redis_url: str) -> None:
        """Test manually adding and retrieving messages."""
        session = AgentSession.create(user_id="test_user", redis_url=redis_url)

        # Initially empty
        assert session.message_count() == 0

        # Add messages
        session.add_message(role="user", content="Hello!")
        session.add_message(role="assistant", content="Hi there!")

        # Retrieve
        messages = session.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello!"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    def test_message_count(self, redis_url: str) -> None:
        """Test message counting."""
        session = AgentSession.create(user_id="test_user", redis_url=redis_url)

        assert session.message_count() == 0

        session.add_message(role="user", content="Test")
        assert session.message_count() == 1

        session.add_message(role="assistant", content="Response")
        assert session.message_count() == 2

    def test_to_agent_inputs(self, redis_url: str) -> None:
        """Test converting messages to OpenAI Agents SDK format."""
        session = AgentSession.create(user_id="test_user", redis_url=redis_url)

        session.add_message(role="user", content="Question?")
        session.add_message(role="assistant", content="Answer!")

        inputs = session.to_agent_inputs()

        assert len(inputs) == 2
        assert inputs[0] == {"content": "Question?", "role": "user"}
        assert inputs[1] == {"content": "Answer!", "role": "assistant"}

    def test_load_session(self, redis_url: str) -> None:
        """Test saving and loading a session."""
        # Create and populate session
        conv_id = str(uuid.uuid4().hex[:16])
        session1 = AgentSession(user_id="test_user", conversation_id=conv_id, redis_url=redis_url)

        session1.add_message(role="user", content="Test message")
        message_count = session1.message_count()

        # Simulate clearing state
        del session1

        # Load session
        session2 = AgentSession.load(
            conversation_id=conv_id, user_id="test_user", redis_url=redis_url
        )

        assert session2.conversation_id == conv_id
        assert session2.message_count() == message_count

        messages = session2.get_messages()
        assert messages[0]["content"] == "Test message"

    def test_session_persists_in_redis(self, redis_url: str, redis_client: Redis) -> None:
        """Test that session data actually exists in Redis."""
        conv_id = str(uuid.uuid4().hex[:16])
        session = AgentSession(user_id="test_user", conversation_id=conv_id, redis_url=redis_url)

        session.add_message(role="user", content="Persisted!")

        # Check Redis keys exist
        keys = redis_client.keys(f"*{conv_id}*")
        assert len(keys) > 0

    def test_clear_session(self, redis_url: str) -> None:
        """Test clearing a session."""
        session = AgentSession.create(user_id="test_user", redis_url=redis_url)

        session.add_message(role="user", content="Message 1")
        session.add_message(role="assistant", content="Response 1")
        assert session.message_count() == 2

        session.clear()
        assert session.message_count() == 0

    def test_delete_session(self, redis_url: str, redis_client: Redis) -> None:
        """Test deleting a session from Redis."""
        conv_id = str(uuid.uuid4().hex[:16])
        session = AgentSession(user_id="test_user", conversation_id=conv_id, redis_url=redis_url)

        session.add_message(role="user", content="Test")

        # Verify exists
        keys_before = redis_client.keys(f"*{conv_id}*")
        assert len(keys_before) > 0

        # Delete
        session.delete()

        # Verify gone
        keys_after = redis_client.keys(f"*{conv_id}*")
        assert len(keys_after) == 0

    def test_agent_tracking(self, redis_url: str) -> None:
        """Test tracking agent usage."""
        session = AgentSession.create(user_id="test_user", redis_url=redis_url)

        # Initially no agent
        assert session.current_agent is None

        # Track agent
        session.track_agent("test_agent")
        assert session.current_agent == "test_agent"

        # Check metadata
        metadata = session.get_metadata()
        assert metadata["current_agent"] == "test_agent"
        assert "test_agent" in metadata["agents_used"]

    def test_store_exchange(self, redis_url: str) -> None:
        """Test convenience method for storing user-assistant exchanges."""
        session = AgentSession.create(user_id="test_user", redis_url=redis_url)

        session.store_exchange(
            user_message="What's the weather?",
            assistant_response="It's sunny!",
            agent_name="weather_agent",
        )

        assert session.message_count() == 2
        assert session.current_agent == "weather_agent"

        messages = session.get_messages()
        assert messages[0]["content"] == "What's the weather?"
        assert messages[1]["content"] == "It's sunny!"
