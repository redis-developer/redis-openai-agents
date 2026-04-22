"""Simple integration test without agent handoffs - proves core concept works."""

import pytest
from agents import Agent, Runner

from redis_openai_agents import AgentSession

pytestmark = pytest.mark.requires_api_keys


class TestSimpleSessionIntegration:
    """Test session persistence with a simple agent (no handoffs)."""

    async def test_simple_save_and_continue(self, redis_url: str) -> None:
        """
        Test the core save/load/continue flow without agent handoffs.

        This proves the fundamental concept works.
        """
        # Create a simple agent (no handoffs = no message wrapping issues)
        simple_agent = Agent(
            name="helpful_agent",
            instructions="You are a helpful assistant. Be concise.",
        )

        # PART 1: Initial conversation
        session1 = AgentSession.create(user_id="demo_user", redis_url=redis_url)
        conversation_id = session1.conversation_id

        # Run first exchange
        result1 = Runner.run_streamed(
            simple_agent, input=[{"content": "What is 2+2?", "role": "user"}]
        )

        # Consume stream
        async for _ in result1.stream_events():
            pass

        # Store
        session1.store_agent_result(result1)

        # Verify we have messages
        initial_count = session1.message_count()
        assert initial_count >= 2  # user + assistant

        # PART 2: Simulate restart - delete Python state
        del session1, result1

        # PART 3: Load session
        session2 = AgentSession.load(
            conversation_id=conversation_id, user_id="demo_user", redis_url=redis_url
        )

        # Verify messages loaded
        assert session2.message_count() == initial_count

        # Get history
        history = session2.to_agent_inputs()
        assert len(history) == initial_count
        assert history[0]["content"] == "What is 2+2?"

        # PART 4: Continue conversation
        continued_inputs = history + [{"content": "And what is 3+3?", "role": "user"}]

        result2 = Runner.run_streamed(simple_agent, input=continued_inputs)

        # Consume stream
        async for _ in result2.stream_events():
            pass

        # Store continuation
        session2.store_agent_result(result2)

        # Verify we have more messages now
        assert session2.message_count() > initial_count

        print("\n✅ Session persistence works!")
        print(f"   Started with: {initial_count} messages")
        print(f"   Ended with: {session2.message_count()} messages")
        print(f"   Conversation ID: {conversation_id}")
