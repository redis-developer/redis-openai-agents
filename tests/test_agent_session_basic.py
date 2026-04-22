"""Basic integration tests for AgentSession - core notebook demo scenarios.

This simplified test suite focuses on the key user-facing scenarios:
1. Create session, store messages, retrieve them
2. Save session, clear Python state, load and continue
3. Multiple concurrent conversations

These tests replicate exactly what the notebook will demonstrate.
"""

import uuid

import pytest
from agents import Agent, Runner, TResponseInputItem
from redis import Redis

from redis_openai_agents import AgentSession

# Mark all tests as requiring API keys
pytestmark = pytest.mark.requires_api_keys


@pytest.fixture
def language_agents() -> dict[str, Agent]:
    """Create language-specific agents for routing tests."""
    return {
        "french": Agent(name="french_agent", instructions="You only speak French"),
        "spanish": Agent(name="spanish_agent", instructions="You only speak Spanish"),
        "english": Agent(name="english_agent", instructions="You only speak English"),
    }


@pytest.fixture
def triage_agent(language_agents: dict[str, Agent]) -> Agent:
    """Create triage agent that routes to language-specific agents."""
    return Agent(
        name="triage_agent",
        instructions="Handoff to the appropriate agent based on the language of the request.",
        handoffs=list(language_agents.values()),
    )


class TestSessionBasics:
    """Test basic session creation and message storage."""

    def test_create_and_store_messages(self, redis_url: str) -> None:
        """Test creating a session and storing messages manually."""
        session = AgentSession.create(user_id="demo_user", redis_url=redis_url)

        # Manually store some messages
        session.add_message(role="user", content="Hello!")
        session.add_message(role="assistant", content="Hi there!")

        # Retrieve and verify
        messages = session.get_messages()
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello!"
        assert messages[1]["content"] == "Hi there!"

    def test_message_count(self, redis_url: str) -> None:
        """Test message counting."""
        session = AgentSession.create(user_id="demo_user", redis_url=redis_url)

        assert session.message_count() == 0

        session.add_message(role="user", content="Test")
        assert session.message_count() == 1

        session.add_message(role="assistant", content="Response")
        assert session.message_count() == 2


class TestNotebookDemoScenario:
    """
    Test the exact scenario that will be demonstrated in the notebook.

    This is the critical test - it must work exactly as shown in the notebook.
    """

    async def test_save_clear_load_continue(
        self, redis_url: str, triage_agent: Agent, language_agents: dict[str, Agent]
    ) -> None:
        """
        Replicate the notebook demo:
        1. Create session and run conversation
        2. Store result
        3. Simulate "clearing Python state" (del variables)
        4. Load session from Redis
        5. Continue conversation with restored context
        """
        # PART 1: Initial conversation
        session1 = AgentSession.create(user_id="demo_user", redis_url=redis_url)
        conversation_id = session1.conversation_id

        # First message - in French
        user_msg1 = "Bonjour! Comment allez-vous?"
        inputs1: list[TResponseInputItem] = [{"content": user_msg1, "role": "user"}]

        result1 = Runner.run_streamed(triage_agent, input=inputs1)

        # Consume the stream to get the final result
        async for _ in result1.stream_events():
            pass  # We don't need the streaming events for this test

        # Store the conversation (result now contains all messages via to_input_list())
        session1.store_agent_result(result1)

        # Verify routing worked
        assert result1.current_agent.name == "french_agent"

        initial_count = session1.message_count()
        assert initial_count >= 2  # User + assistant

        # PART 2: Simulate clearing Python state (kernel restart simulation)
        del session1, result1, inputs1

        # PART 3: Load session from Redis
        session2 = AgentSession.load(
            conversation_id=conversation_id, user_id="demo_user", redis_url=redis_url
        )

        # Verify session was restored
        assert session2.conversation_id == conversation_id
        assert session2.message_count() == initial_count

        # Get history in OpenAI Agents SDK format
        history = session2.to_agent_inputs()
        assert len(history) == initial_count
        assert history[0]["content"] == user_msg1

        # PART 4: Continue conversation
        user_msg2 = "Parlez-moi de Paris"
        continued_inputs = history + [{"content": user_msg2, "role": "user"}]

        # Continue with the french agent directly (we know from history which agent to use)
        result2 = Runner.run_streamed(language_agents["french"], input=continued_inputs)

        # Consume the stream
        async for _ in result2.stream_events():
            pass

        # Store the new exchange (all messages are in result via to_input_list())
        session2.store_agent_result(result2)

        # Verify continuation
        assert session2.message_count() > initial_count

        final_messages = session2.get_messages()
        user_msgs = [m for m in final_messages if m["role"] == "user"]
        assert len(user_msgs) >= 2
        assert user_msgs[-1]["content"] == user_msg2


class TestMultipleConversations:
    """Test managing multiple concurrent conversations."""

    async def test_three_parallel_conversations(self, redis_url: str, triage_agent: Agent) -> None:
        """
        Create three conversations in different languages.

        This demonstrates the multi-conversation management feature.
        """
        conversations = {
            "french": ("Bonjour!", "french_agent"),
            "spanish": ("¡Hola!", "spanish_agent"),
            "english": ("Hello!", "english_agent"),
        }

        sessions = {}

        # Create three separate conversations
        for lang, (message, expected_agent) in conversations.items():
            session = AgentSession.create(user_id="demo_user", redis_url=redis_url)

            inputs: list[TResponseInputItem] = [{"content": message, "role": "user"}]
            result = Runner.run_streamed(triage_agent, input=inputs)

            # Consume the stream
            async for _ in result.stream_events():
                pass

            session.store_agent_result(result)

            # Verify routing
            assert result.current_agent.name == expected_agent

            sessions[lang] = session

        # Verify all sessions are independent
        conv_ids = [s.conversation_id for s in sessions.values()]
        assert len(set(conv_ids)) == 3  # All unique

        # Verify each has its own messages
        for lang, session in sessions.items():
            messages = session.get_messages()
            assert messages[0]["content"] == conversations[lang][0]


class TestSessionPersistence:
    """Test that data actually persists in Redis."""

    def test_data_persists_after_python_object_deleted(
        self, redis_url: str, redis_client: Redis
    ) -> None:
        """Verify session data exists in Redis after Python object is destroyed."""
        conv_id = str(uuid.uuid4().hex[:16])

        # Create session and add data
        session = AgentSession(user_id="test_user", conversation_id=conv_id, redis_url=redis_url)
        session.add_message(role="user", content="Test message")

        message_count = session.message_count()
        assert message_count == 1

        # Delete Python object
        del session

        # Data should still exist in Redis
        # RedisVL stores messages with a key pattern like "message_history:{name}:messages"
        keys = redis_client.keys(f"*{conv_id}*")
        assert len(keys) > 0  # Some keys exist

        # Load in new session
        new_session = AgentSession.load(
            conversation_id=conv_id, user_id="test_user", redis_url=redis_url
        )
        assert new_session.message_count() == message_count


class TestCleanup:
    """Test session cleanup operations."""

    def test_clear_session(self, redis_url: str) -> None:
        """Test clearing a session's messages."""
        session = AgentSession.create(user_id="test_user", redis_url=redis_url)

        session.add_message(role="user", content="Message 1")
        session.add_message(role="assistant", content="Response 1")

        assert session.message_count() == 2

        session.clear()

        assert session.message_count() == 0

    def test_delete_session(self, redis_url: str, redis_client: Redis) -> None:
        """Test deleting a session entirely."""
        conv_id = str(uuid.uuid4().hex[:16])
        session = AgentSession(user_id="test_user", conversation_id=conv_id, redis_url=redis_url)

        session.add_message(role="user", content="Test")

        # Verify data exists
        keys_before = redis_client.keys(f"*{conv_id}*")
        assert len(keys_before) > 0

        # Delete
        session.delete()

        # Verify data is gone
        keys_after = redis_client.keys(f"*{conv_id}*")
        assert len(keys_after) == 0
