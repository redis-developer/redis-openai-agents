"""Unit tests for Lua Atomic Operations.

These tests use mocks to verify behavior without Redis.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestAtomicOperationsInit:
    """Tests for AtomicOperations initialization."""

    def test_init_loads_script_contents(self) -> None:
        """Should load Lua script contents on initialization."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()

        AtomicOperations(mock_client)

        # Script contents should be loaded (class-level cache)
        assert "atomic_message_append" in AtomicOperations._script_contents
        assert "atomic_response_record" in AtomicOperations._script_contents
        assert "atomic_handoff" in AtomicOperations._script_contents

    @pytest.mark.asyncio
    async def test_lazy_load_caches_script_shas(self) -> None:
        """Should cache script SHAs from script_load on first use."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="test_sha")
        mock_client.evalsha = AsyncMock(return_value=1)

        ops = AtomicOperations(mock_client)

        # SHAs should not be loaded yet
        assert len(ops._script_shas) == 0

        # Trigger lazy load
        await ops.atomic_message_append(
            session_key="session:test",
            message={"role": "user", "content": "test"},
        )

        # All cached SHAs should be populated
        assert len(ops._script_shas) > 0
        for sha in ops._script_shas.values():
            assert sha == "test_sha"


class TestAtomicMessageAppend:
    """Tests for atomic_message_append operation."""

    @pytest.mark.asyncio
    async def test_calls_evalsha_with_correct_keys(self) -> None:
        """Should call evalsha with session key."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value=1)

        ops = AtomicOperations(mock_client)

        await ops.atomic_message_append(
            session_key="session:abc",
            message={"role": "user", "content": "Hi"},
        )

        mock_client.evalsha.assert_called_once()
        call_args = mock_client.evalsha.call_args

        # First arg is SHA
        assert call_args[0][0] == "sha123"
        # Second arg is number of keys
        assert call_args[0][1] == 1
        # Third arg is the key
        assert call_args[0][2] == "session:abc"

    @pytest.mark.asyncio
    async def test_passes_message_as_json(self) -> None:
        """Should serialize message to JSON."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value=1)

        ops = AtomicOperations(mock_client)

        message = {"id": "1", "role": "user", "content": "Hello"}
        await ops.atomic_message_append(
            session_key="session:abc",
            message=message,
        )

        call_args = mock_client.evalsha.call_args
        # Message should be JSON string in args
        message_arg = call_args[0][3]  # Fourth positional arg
        parsed = json.loads(message_arg)
        assert parsed["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_passes_max_messages_param(self) -> None:
        """Should pass max_messages for window trimming."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value=1)

        ops = AtomicOperations(mock_client)

        await ops.atomic_message_append(
            session_key="session:abc",
            message={"role": "user", "content": "Hi"},
            max_messages=50,
        )

        call_args = mock_client.evalsha.call_args
        # max_messages should be in args
        assert "50" in str(call_args)

    @pytest.mark.asyncio
    async def test_passes_ttl_param(self) -> None:
        """Should pass TTL for expiration refresh."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value=1)

        ops = AtomicOperations(mock_client)

        await ops.atomic_message_append(
            session_key="session:abc",
            message={"role": "user", "content": "Hi"},
            ttl=3600,
        )

        call_args = mock_client.evalsha.call_args
        # TTL should be in args
        assert "3600" in str(call_args)


class TestAtomicResponseRecord:
    """Tests for atomic_response_record operation."""

    @pytest.mark.asyncio
    async def test_calls_evalsha_with_multiple_keys(self) -> None:
        """Should call evalsha with cache and session keys."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value="OK")

        ops = AtomicOperations(mock_client)

        await ops.atomic_response_record(
            session_key="session:abc",
            cache_key="cache:test",
            query_hash="hash123",
            response="Response text",
            user_message={"role": "user", "content": "Q"},
            assistant_message={"role": "assistant", "content": "A"},
            latency_ms=100.0,
            input_tokens=10,
            output_tokens=5,
        )

        mock_client.evalsha.assert_called_once()
        call_args = mock_client.evalsha.call_args

        # Should have 2 keys (session, cache)
        assert call_args[0][1] == 2

    @pytest.mark.asyncio
    async def test_passes_all_required_params(self) -> None:
        """Should pass all required parameters."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value="OK")

        ops = AtomicOperations(mock_client)

        await ops.atomic_response_record(
            session_key="session:abc",
            cache_key="cache:test",
            query_hash="query_hash_123",
            response="Test response",
            user_message={"role": "user", "content": "Question"},
            assistant_message={"role": "assistant", "content": "Answer"},
            latency_ms=150.5,
            input_tokens=20,
            output_tokens=15,
        )

        call_args = mock_client.evalsha.call_args
        args_str = str(call_args)

        assert "query_hash_123" in args_str
        assert "Test response" in args_str
        assert "Question" in args_str
        assert "Answer" in args_str


class TestAtomicHandoff:
    """Tests for atomic_handoff operation."""

    @pytest.mark.asyncio
    async def test_calls_evalsha_with_session_and_lock_keys(self) -> None:
        """Should call evalsha with session key and lock key."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value="OK")

        ops = AtomicOperations(mock_client)

        await ops.atomic_handoff(
            session_key="session:abc",
            from_agent="agent_a",
            to_agent="agent_b",
            context={"reason": "test"},
        )

        mock_client.evalsha.assert_called_once()
        call_args = mock_client.evalsha.call_args

        # Should have 2 keys (session, lock)
        assert call_args[0][1] == 2
        # Lock key should be derived from session key
        assert "session:abc" in str(call_args)
        assert "handoff_lock" in str(call_args)

    @pytest.mark.asyncio
    async def test_passes_agent_names(self) -> None:
        """Should pass from_agent and to_agent."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value="OK")

        ops = AtomicOperations(mock_client)

        await ops.atomic_handoff(
            session_key="session:abc",
            from_agent="research_agent",
            to_agent="analysis_agent",
            context={},
        )

        call_args = mock_client.evalsha.call_args
        args_str = str(call_args)

        assert "research_agent" in args_str
        assert "analysis_agent" in args_str

    @pytest.mark.asyncio
    async def test_passes_context_as_json(self) -> None:
        """Should serialize context to JSON."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value="OK")

        ops = AtomicOperations(mock_client)

        context = {"reason": "task_complete", "data": {"key": "value"}}
        await ops.atomic_handoff(
            session_key="session:abc",
            from_agent="a",
            to_agent="b",
            context=context,
        )

        call_args = mock_client.evalsha.call_args
        args_str = str(call_args)

        assert "task_complete" in args_str

    @pytest.mark.asyncio
    async def test_raises_handoff_in_progress_error(self) -> None:
        """Should raise HandoffInProgressError when lock held."""
        from redis_openai_agents.atomic import AtomicOperations, HandoffInProgressError

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value={"err": "HANDOFF_IN_PROGRESS"})

        ops = AtomicOperations(mock_client)

        with pytest.raises(HandoffInProgressError):
            await ops.atomic_handoff(
                session_key="session:abc",
                from_agent="a",
                to_agent="b",
                context={},
            )

    @pytest.mark.asyncio
    async def test_passes_lock_ttl(self) -> None:
        """Should pass lock TTL for expiration."""
        from redis_openai_agents.atomic import AtomicOperations

        mock_client = MagicMock()
        mock_client.script_load = AsyncMock(return_value="sha123")
        mock_client.evalsha = AsyncMock(return_value="OK")

        ops = AtomicOperations(mock_client)

        await ops.atomic_handoff(
            session_key="session:abc",
            from_agent="a",
            to_agent="b",
            context={},
            lock_ttl=60,
        )

        call_args = mock_client.evalsha.call_args
        assert "60" in str(call_args)


class TestLuaScripts:
    """Tests for Lua script loading and structure."""

    def test_scripts_directory_exists(self) -> None:
        """Lua scripts directory should exist."""
        from redis_openai_agents.atomic import SCRIPTS_DIR

        assert SCRIPTS_DIR.exists()
        assert SCRIPTS_DIR.is_dir()

    def test_required_scripts_exist(self) -> None:
        """All required Lua scripts should exist."""
        from redis_openai_agents.atomic import SCRIPTS_DIR

        required = [
            "atomic_message_append.lua",
            "atomic_response_record.lua",
            "atomic_handoff.lua",
        ]

        for script_name in required:
            script_path = SCRIPTS_DIR / script_name
            assert script_path.exists(), f"Missing script: {script_name}"

    def test_scripts_contain_keys_argv(self) -> None:
        """Scripts should use KEYS and ARGV (cluster-safe)."""
        from redis_openai_agents.atomic import SCRIPTS_DIR

        for script_file in SCRIPTS_DIR.glob("*.lua"):
            content = script_file.read_text()
            assert "KEYS[" in content, f"{script_file.name} should use KEYS"
            assert "ARGV[" in content, f"{script_file.name} should use ARGV"
