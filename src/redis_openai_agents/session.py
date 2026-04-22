"""AgentSession - Persistent session management for OpenAI Agents SDK.

This module provides session/conversation state management using Redis,
built on top of RedisVL's MessageHistory.
"""

import asyncio
import re
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field
from redisvl.extensions.message_history import MessageHistory  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from .pool import RedisConnectionPool


class SessionMetadata(BaseModel):
    """Metadata for an agent session."""

    user_id: str
    conversation_id: str
    current_agent: str | None = None
    agents_used: list[str] = Field(default_factory=list)
    created_at: float | None = None
    updated_at: float | None = None


class AgentSession:
    """
    Persistent session storage for OpenAI Agents SDK conversations.

    Built on top of RedisVL's MessageHistory, this class provides:
    - Message persistence across Python sessions
    - Multi-conversation management per user
    - Agent handoff tracking
    - Conversation metadata

    Example:
        >>> # Create a new session
        >>> session = AgentSession.create(user_id="user_123")
        >>>
        >>> # Run agent with session
        >>> result = Runner.run(agent, input=inputs, session=session)
        >>>
        >>> # Later, load and continue
        >>> session = AgentSession.load(conversation_id="conv_abc")
        >>> history = session.to_agent_inputs()
        >>> result = Runner.run(agent, input=history + new_inputs, session=session)
    """

    def __init__(
        self,
        user_id: str,
        conversation_id: str | None = None,
        redis_url: str = "redis://localhost:6379",
        pool: Optional["RedisConnectionPool"] = None,
    ):
        """
        Initialize an AgentSession.

        Args:
            user_id: User identifier
            conversation_id: Optional conversation ID (generates one if not provided)
            redis_url: Redis connection URL
            pool: Optional shared connection pool
        """
        self.user_id = user_id
        self.conversation_id = conversation_id or str(uuid4().hex[:16])
        self._pool = pool

        # Use pool's URL if provided
        if pool is not None:
            self.redis_url = pool.redis_url
        else:
            self.redis_url = redis_url

        # Create RedisVL MessageHistory instance
        # CRITICAL: Use conversation_id as session_tag to enable loading existing conversations
        # The name is the index name (shared across all sessions)
        # The session_tag is what identifies this specific conversation
        self._history = MessageHistory(
            name="agent_sessions",
            session_tag=self.conversation_id,  # This makes messages retrievable!
            redis_url=self.redis_url,
        )

        # Track current agent
        self._current_agent: str | None = None
        self._agents_used: set[str] = set()

    @classmethod
    def create(
        cls,
        user_id: str,
        redis_url: str = "redis://localhost:6379",
    ) -> "AgentSession":
        """
        Create a new session with a generated conversation ID.

        Args:
            user_id: User identifier
            redis_url: Redis connection URL

        Returns:
            New AgentSession instance
        """
        return cls(user_id=user_id, redis_url=redis_url)

    @classmethod
    def load(
        cls,
        conversation_id: str,
        user_id: str | None = None,
        redis_url: str = "redis://localhost:6379",
    ) -> "AgentSession":
        """
        Load an existing session from Redis.

        Args:
            conversation_id: Conversation ID to load
            user_id: Optional user ID (will be extracted from session if not provided)
            redis_url: Redis connection URL

        Returns:
            Loaded AgentSession instance

        Raises:
            ValueError: If session not found and user_id not provided
        """
        if not user_id:
            # TODO: Extract user_id from session metadata in Redis
            raise ValueError("user_id must be provided when loading session")

        session = cls(user_id=user_id, conversation_id=conversation_id, redis_url=redis_url)

        # Load metadata to restore agent tracking
        metadata = session.get_metadata()
        session._current_agent = metadata.get("current_agent")
        session._agents_used = set(metadata.get("agents_used", []))

        return session

    @classmethod
    def list_conversations(
        cls,
        user_id: str,
        redis_url: str = "redis://localhost:6379",
    ) -> list["AgentSession"]:
        """
        List all conversations for a user.

        Args:
            user_id: User identifier
            redis_url: Redis connection URL

        Returns:
            List of AgentSession instances for this user
        """
        # TODO: Implement by querying Redis for all sessions with this user_id prefix
        # For now, return empty list - this requires maintaining a user index
        return []

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """
        Add a message to the session.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **metadata: Additional metadata to store with message
        """
        self._history.add_message({"role": role, "content": content, **metadata})

    def store_exchange(
        self, user_message: str, assistant_response: str, agent_name: str | None = None
    ) -> None:
        """
        Store a user-assistant message exchange.

        This is a convenience method for the common pattern of storing
        a user message and the assistant's response together.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            agent_name: Optional name of the agent that responded
        """
        self.add_message(role="user", content=user_message)
        self.add_message(role="assistant", content=assistant_response)
        if agent_name:
            self.track_agent(agent_name)

    def store_agent_result(self, result: Any) -> None:
        """
        Store messages from OpenAI Agents SDK Runner result.

        This works with both `Runner.run()` and `Runner.run_streamed()` results.
        After streaming completes, call `result.to_input_list()` to get all messages
        including the assistant's response.

        Args:
            result: The result object from Runner.run() or Runner.run_streamed()
                   (after consuming the stream)
        """
        # Track the current agent
        if hasattr(result, "current_agent") and hasattr(result.current_agent, "name"):
            self.track_agent(result.current_agent.name)

        # Get all messages from the result using to_input_list()
        # This includes both the input messages and the assistant's response
        if hasattr(result, "to_input_list"):
            all_messages = result.to_input_list()

            # Track what we've stored to avoid duplicates from handoff wrapping
            stored_user_messages = set()

            # Store all messages, but filter out internal/function messages
            for msg in all_messages:
                # Handle different message formats
                # OpenAI Agents SDK returns messages as objects with .role and .content
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    role = msg.role if isinstance(msg.role, str) else str(msg.role)

                    # Skip system and function messages - only store user/assistant
                    if role not in ["user", "assistant"]:
                        continue

                    # Content can be a string or a list of content parts
                    content_raw = msg.content

                    if isinstance(content_raw, str):
                        # Check if this is a wrapped handoff context message
                        # These start with "For context, here is the conversation"
                        if "For context, here is the conversation" in content_raw:
                            # Extract the original user message from the history
                            # Format: "1. user: <original message>\n2. function_call:..."
                            match = re.search(r"1\. user: (.+?)(?:\n2\.|$)", content_raw, re.DOTALL)
                            if match:
                                content = match.group(1).strip()
                            else:
                                # Fallback: skip this wrapped message entirely
                                continue
                        else:
                            content = content_raw
                    elif isinstance(content_raw, list):
                        # Extract text from content parts
                        text_parts = []
                        for part in content_raw:
                            part_text = None
                            if hasattr(part, "text"):
                                part_text = part.text
                            elif isinstance(part, dict) and "text" in part:
                                part_text = part["text"]

                            if part_text:
                                # Check if this text part is a wrapped handoff message
                                if "For context, here is the conversation" in part_text:
                                    # Extract original from wrapped
                                    match = re.search(
                                        r"1\. user: (.+?)(?:\n2\.|$)", part_text, re.DOTALL
                                    )
                                    if match:
                                        text_parts.append(match.group(1).strip())
                                else:
                                    text_parts.append(part_text)
                        content = " ".join(text_parts)
                    else:
                        content = str(content_raw)

                    if content:  # Only store non-empty messages
                        # Deduplicate: skip if this exact user message was already stored
                        if role == "user" and content in stored_user_messages:
                            continue
                        if role == "user":
                            stored_user_messages.add(content)

                        self.add_message(role=role, content=content)
                elif isinstance(msg, dict):
                    role = msg.get("role", "unknown")

                    # Skip non-user/assistant messages
                    if role not in ["user", "assistant"]:
                        continue

                    content_raw = msg.get("content", "")

                    # Handle list format in dict - extract text from content parts
                    if isinstance(content_raw, list):
                        text_parts = []
                        for part in content_raw:
                            if isinstance(part, dict) and "text" in part:
                                text_parts.append(part["text"])
                        content = " ".join(text_parts)
                    else:
                        content = content_raw

                    # Unwrap handoff context messages
                    # These come from agent handoffs and wrap the original user message
                    # They have role="assistant" but contain "For context, here is the conversation"
                    if (
                        isinstance(content, str)
                        and "For context, here is the conversation" in content
                    ):
                        match = re.search(r"1\. user: (.+?)(?:\n2\.|$)", content, re.DOTALL)
                        if match:
                            # Extract the original user message and correct the role
                            content = match.group(1).strip()
                            role = "user"  # Correct the role to user
                        else:
                            # Skip this wrapped message if we can't extract the original
                            continue

                    if content:
                        # Deduplicate for dict format too
                        if role == "user" and content in stored_user_messages:
                            continue
                        if role == "user":
                            stored_user_messages.add(content)

                        self.add_message(role=role, content=content)

    def get_messages(self, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Get recent messages from the session.

        Args:
            top_k: Number of recent messages to retrieve (None for all)

        Returns:
            List of message dictionaries
        """
        if top_k is None:
            # Get all messages
            result: list[dict[str, Any]] = self._history.get_recent(top_k=1000)
            return result
        result = self._history.get_recent(top_k=top_k)
        return list(result)

    def to_agent_inputs(self) -> list[dict[str, str]]:
        """
        Convert session messages to OpenAI Agents SDK input format.

        Returns:
            List of messages in format [{" content": str, "role": str}, ...]
        """
        messages = self.get_messages()
        # Ensure format matches TResponseInputItem
        return [{"content": msg["content"], "role": msg["role"]} for msg in messages]

    def message_count(self) -> int:
        """
        Get the number of messages in the session.

        Returns:
            Count of messages
        """
        return len(self.get_messages())

    def track_agent(self, agent_name: str) -> None:
        """
        Track agent usage in this session.

        Args:
            agent_name: Name of the agent being used
        """
        self._current_agent = agent_name
        self._agents_used.add(agent_name)

    @property
    def current_agent(self) -> str | None:
        """Get the current agent name."""
        return self._current_agent

    def get_metadata(self) -> dict[str, Any]:
        """
        Get session metadata.

        Returns:
            Dictionary containing session metadata
        """
        return {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "current_agent": self._current_agent,
            "agents_used": list(self._agents_used),
            "message_count": self.message_count(),
        }

    def clear(self) -> None:
        """Clear all messages from the session."""
        self._history.clear()
        self._current_agent = None
        self._agents_used.clear()

    def delete(self) -> None:
        """Delete the session and all its data from Redis."""
        self._history.clear()
        self._current_agent = None
        self._agents_used.clear()

    # Async methods

    async def aadd_message(self, role: str, content: str, **metadata: Any) -> None:
        """Async version of add_message() - add a message to the session.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **metadata: Additional metadata to store with message
        """
        await asyncio.to_thread(
            self._history.add_message, {"role": role, "content": content, **metadata}
        )

    async def aget_messages(self, top_k: int | None = None) -> list[dict[str, Any]]:
        """Async version of get_messages() - get recent messages from the session.

        Args:
            top_k: Number of recent messages to retrieve (None for all)

        Returns:
            List of message dictionaries
        """
        if top_k is None:
            result: list[dict[str, Any]] = await asyncio.to_thread(
                self._history.get_recent, top_k=1000
            )
            return result
        result = await asyncio.to_thread(self._history.get_recent, top_k=top_k)
        return list(result)

    async def astore_agent_result(self, result: Any) -> None:
        """Async version of store_agent_result() - store messages from Runner result.

        Args:
            result: The result object from Runner.run()
        """
        await asyncio.to_thread(self.store_agent_result, result)
