"""Unit tests for ResumableStreamRunner."""

import json
from unittest.mock import AsyncMock

import pytest

from redis_openai_agents.resumable_streaming import (
    ResumableStreamRunner,
    StreamEvent,
    StreamingEventPublisher,
)


class TestStreamEvent:
    """Test StreamEvent dataclass."""

    def test_create_stream_event(self) -> None:
        """Can create StreamEvent."""
        event = StreamEvent(
            id="12345-0",
            type="text_delta",
            data={"delta": "Hello"},
            timestamp=1234567890.0,
        )

        assert event.id == "12345-0"
        assert event.type == "text_delta"
        assert event.data == {"delta": "Hello"}
        assert event.timestamp == 1234567890.0

    def test_stream_event_to_dict(self) -> None:
        """StreamEvent converts to dict."""
        event = StreamEvent(
            id="12345-0",
            type="text_delta",
            data={"delta": "Hello"},
            timestamp=1234567890.0,
        )

        result = event.to_dict()

        assert result["id"] == "12345-0"
        assert result["type"] == "text_delta"
        assert result["data"] == {"delta": "Hello"}
        assert result["timestamp"] == 1234567890.0


class TestResumableStreamRunnerInit:
    """Test ResumableStreamRunner initialization."""

    def test_default_values(self) -> None:
        """Runner has correct default values."""
        runner = ResumableStreamRunner()

        assert runner._redis_url == "redis://localhost:6379"
        assert runner._stream_prefix == "llm_stream"
        assert runner._max_stream_length is None
        assert runner._consumer_group == "consumers"
        assert runner._client is None

    def test_custom_values(self) -> None:
        """Runner accepts custom configuration."""
        runner = ResumableStreamRunner(
            redis_url="redis://custom:6380",
            stream_prefix="custom_stream",
            max_stream_length=1000,
            consumer_group="custom_group",
        )

        assert runner._redis_url == "redis://custom:6380"
        assert runner._stream_prefix == "custom_stream"
        assert runner._max_stream_length == 1000
        assert runner._consumer_group == "custom_group"


class TestResumableStreamRunnerStreamKey:
    """Test stream key generation."""

    def test_get_stream_key(self) -> None:
        """Stream key generated correctly."""
        runner = ResumableStreamRunner(stream_prefix="test")

        key = runner._get_stream_key("session_123")

        assert key == "test:session_123"

    def test_get_stream_key_with_special_chars(self) -> None:
        """Stream key handles special characters."""
        runner = ResumableStreamRunner(stream_prefix="llm_stream")

        key = runner._get_stream_key("user:123:chat:456")

        assert key == "llm_stream:user:123:chat:456"


class TestResumableStreamRunnerParseMessage:
    """Test message parsing."""

    def test_parse_message_basic(self) -> None:
        """Parses basic message correctly."""
        runner = ResumableStreamRunner()

        fields = {
            "type": "text_delta",
            "data": '{"delta": "Hello"}',
            "timestamp": "1234567890.0",
        }

        result = runner._parse_message("12345-0", fields)

        assert result["id"] == "12345-0"
        assert result["type"] == "text_delta"
        assert result["data"] == {"delta": "Hello"}
        assert result["timestamp"] == 1234567890.0

    def test_parse_message_with_metadata(self) -> None:
        """Parses message with metadata."""
        runner = ResumableStreamRunner()

        fields = {
            "type": "text_delta",
            "data": '{"delta": "Hello"}',
            "timestamp": "1234567890.0",
            "metadata": '{"agent": "assistant"}',
        }

        result = runner._parse_message("12345-0", fields)

        assert result["metadata"] == {"agent": "assistant"}

    def test_parse_message_missing_fields(self) -> None:
        """Handles missing fields gracefully."""
        runner = ResumableStreamRunner()

        fields = {}

        result = runner._parse_message("12345-0", fields)

        assert result["id"] == "12345-0"
        assert result["type"] == "unknown"
        assert result["data"] == {}
        assert result["timestamp"] == 0


class TestResumableStreamRunnerPublish:
    """Test event publishing."""

    async def test_publish_event_not_initialized(self) -> None:
        """Raises error when not initialized."""
        runner = ResumableStreamRunner()

        with pytest.raises(RuntimeError, match="not initialized"):
            await runner.publish_event(
                session_id="test",
                event_type="text_delta",
                data={"delta": "test"},
            )

    async def test_publish_event_calls_xadd(self) -> None:
        """Publish calls Redis XADD."""
        runner = ResumableStreamRunner(stream_prefix="test")

        mock_client = AsyncMock()
        mock_client.xadd.return_value = "12345-0"
        runner._client = mock_client

        msg_id = await runner.publish_event(
            session_id="session_1",
            event_type="text_delta",
            data={"delta": "Hello"},
        )

        assert msg_id == "12345-0"
        mock_client.xadd.assert_called_once()

        # Check call arguments
        call_args = mock_client.xadd.call_args
        assert call_args[0][0] == "test:session_1"  # Stream key
        fields = call_args[0][1]
        assert fields["type"] == "text_delta"
        assert json.loads(fields["data"]) == {"delta": "Hello"}

    async def test_publish_event_with_max_length(self) -> None:
        """Publish uses maxlen when configured."""
        runner = ResumableStreamRunner(
            stream_prefix="test",
            max_stream_length=100,
        )

        mock_client = AsyncMock()
        mock_client.xadd.return_value = "12345-0"
        runner._client = mock_client

        await runner.publish_event(
            session_id="session_1",
            event_type="text_delta",
            data={"delta": "test"},
        )

        # Check maxlen was passed
        call_kwargs = mock_client.xadd.call_args[1]
        assert call_kwargs["maxlen"] == 100
        assert call_kwargs["approximate"] is False

    async def test_publish_event_with_metadata(self) -> None:
        """Publish includes metadata when provided."""
        runner = ResumableStreamRunner(stream_prefix="test")

        mock_client = AsyncMock()
        mock_client.xadd.return_value = "12345-0"
        runner._client = mock_client

        await runner.publish_event(
            session_id="session_1",
            event_type="text_delta",
            data={"delta": "test"},
            metadata={"agent": "assistant"},
        )

        fields = mock_client.xadd.call_args[0][1]
        assert "metadata" in fields
        assert json.loads(fields["metadata"]) == {"agent": "assistant"}


class TestResumableStreamRunnerGetAllEvents:
    """Test get_all_events method."""

    async def test_get_all_events_no_client(self) -> None:
        """Returns empty list when not initialized."""
        runner = ResumableStreamRunner()

        events = await runner.get_all_events("session_1")

        assert events == []

    async def test_get_all_events_returns_parsed(self) -> None:
        """Returns parsed events."""
        runner = ResumableStreamRunner(stream_prefix="test")

        mock_client = AsyncMock()
        mock_client.xrange.return_value = [
            (
                "12345-0",
                {"type": "text_delta", "data": '{"delta": "Hello"}', "timestamp": "1234567890.0"},
            ),
            (
                "12345-1",
                {"type": "text_delta", "data": '{"delta": " world"}', "timestamp": "1234567891.0"},
            ),
        ]
        runner._client = mock_client

        events = await runner.get_all_events("session_1")

        assert len(events) == 2
        assert events[0]["data"]["delta"] == "Hello"
        assert events[1]["data"]["delta"] == " world"


class TestResumableStreamRunnerStreamInfo:
    """Test get_stream_info method."""

    async def test_get_stream_info_no_client(self) -> None:
        """Returns empty dict when not initialized."""
        runner = ResumableStreamRunner()

        info = await runner.get_stream_info("session_1")

        assert info == {}

    async def test_get_stream_info_returns_data(self) -> None:
        """Returns stream information."""
        runner = ResumableStreamRunner(stream_prefix="test")

        mock_client = AsyncMock()
        mock_client.xinfo_stream.return_value = {
            "length": 10,
            "first-entry": ("12345-0", {}),
            "last-entry": ("12345-9", {}),
            "groups": 2,
        }
        runner._client = mock_client

        info = await runner.get_stream_info("session_1")

        assert info["length"] == 10
        assert info["first_entry_id"] == "12345-0"
        assert info["last_entry_id"] == "12345-9"


class TestResumableStreamRunnerDelete:
    """Test delete_stream method."""

    async def test_delete_stream_no_client(self) -> None:
        """Returns False when not initialized."""
        runner = ResumableStreamRunner()

        result = await runner.delete_stream("session_1")

        assert result is False

    async def test_delete_stream_success(self) -> None:
        """Returns True when deleted."""
        runner = ResumableStreamRunner(stream_prefix="test")

        mock_client = AsyncMock()
        mock_client.delete.return_value = 1
        runner._client = mock_client

        result = await runner.delete_stream("session_1")

        assert result is True
        mock_client.delete.assert_called_once_with("test:session_1")


class TestResumableStreamRunnerAck:
    """Test ack method."""

    async def test_ack_no_client(self) -> None:
        """Returns 0 when not initialized."""
        runner = ResumableStreamRunner()

        result = await runner.ack("session_1", "consumer_1", "12345-0")

        assert result == 0

    async def test_ack_calls_xack(self) -> None:
        """Ack calls Redis XACK."""
        runner = ResumableStreamRunner(stream_prefix="test")

        mock_client = AsyncMock()
        mock_client.xack.return_value = 1
        runner._client = mock_client

        result = await runner.ack("session_1", "consumer_1", "12345-0")

        assert result == 1
        mock_client.xack.assert_called_once_with(
            "test:session_1",
            "consumers:session_1",
            "12345-0",
        )


class TestResumableStreamRunnerLifecycle:
    """Test lifecycle methods."""

    async def test_close_clears_client(self) -> None:
        """Close clears client reference."""
        runner = ResumableStreamRunner()

        mock_client = AsyncMock()
        runner._client = mock_client

        await runner.close()

        assert runner._client is None
        mock_client.aclose.assert_called_once()

    async def test_close_when_no_client(self) -> None:
        """Close handles no client gracefully."""
        runner = ResumableStreamRunner()

        # Should not raise
        await runner.close()


class TestStreamingEventPublisher:
    """Test StreamingEventPublisher helper."""

    async def test_publish_text_delta(self) -> None:
        """Publishes text delta correctly."""
        mock_runner = AsyncMock()
        mock_runner.publish_event.return_value = "12345-0"

        publisher = StreamingEventPublisher(mock_runner, "session_1")

        msg_id = await publisher.publish_text_delta("Hello")

        assert msg_id == "12345-0"
        mock_runner.publish_event.assert_called_once_with(
            session_id="session_1",
            event_type="text_delta",
            data={"delta": "Hello"},
        )

    async def test_publish_tool_call(self) -> None:
        """Publishes tool call correctly."""
        mock_runner = AsyncMock()
        mock_runner.publish_event.return_value = "12345-0"

        publisher = StreamingEventPublisher(mock_runner, "session_1")

        msg_id = await publisher.publish_tool_call(
            tool_name="search",
            arguments={"query": "test"},
            call_id="call_123",
        )

        assert msg_id == "12345-0"
        call_args = mock_runner.publish_event.call_args
        assert call_args[1]["event_type"] == "tool_call"
        assert call_args[1]["data"]["tool"] == "search"
        assert call_args[1]["data"]["call_id"] == "call_123"

    async def test_publish_tool_result(self) -> None:
        """Publishes tool result correctly."""
        mock_runner = AsyncMock()
        mock_runner.publish_event.return_value = "12345-0"

        publisher = StreamingEventPublisher(mock_runner, "session_1")

        msg_id = await publisher.publish_tool_result(
            tool_name="search",
            result={"results": ["item1", "item2"]},
        )

        assert msg_id == "12345-0"
        call_args = mock_runner.publish_event.call_args
        assert call_args[1]["event_type"] == "tool_result"
        assert call_args[1]["data"]["result"] == {"results": ["item1", "item2"]}

    async def test_publish_stream_start(self) -> None:
        """Publishes stream start correctly."""
        mock_runner = AsyncMock()
        mock_runner.publish_event.return_value = "12345-0"

        publisher = StreamingEventPublisher(mock_runner, "session_1")

        msg_id = await publisher.publish_stream_start("assistant")

        assert msg_id == "12345-0"
        call_args = mock_runner.publish_event.call_args
        assert call_args[1]["event_type"] == "stream_start"
        assert call_args[1]["data"]["agent"] == "assistant"

    async def test_publish_stream_end(self) -> None:
        """Publishes stream end correctly."""
        mock_runner = AsyncMock()
        mock_runner.publish_event.return_value = "12345-0"

        publisher = StreamingEventPublisher(mock_runner, "session_1")

        msg_id = await publisher.publish_stream_end("complete")

        assert msg_id == "12345-0"
        call_args = mock_runner.publish_event.call_args
        assert call_args[1]["event_type"] == "stream_end"
        assert call_args[1]["data"]["reason"] == "complete"

    async def test_publish_handoff(self) -> None:
        """Publishes handoff correctly."""
        mock_runner = AsyncMock()
        mock_runner.publish_event.return_value = "12345-0"

        publisher = StreamingEventPublisher(mock_runner, "session_1")

        msg_id = await publisher.publish_handoff("triage", "support")

        assert msg_id == "12345-0"
        call_args = mock_runner.publish_event.call_args
        assert call_args[1]["event_type"] == "handoff"
        assert call_args[1]["data"]["from_agent"] == "triage"
        assert call_args[1]["data"]["to_agent"] == "support"

    async def test_publish_error(self) -> None:
        """Publishes error correctly."""
        mock_runner = AsyncMock()
        mock_runner.publish_event.return_value = "12345-0"

        publisher = StreamingEventPublisher(mock_runner, "session_1")

        msg_id = await publisher.publish_error(
            error_type="rate_limit",
            message="Too many requests",
            details={"retry_after": 60},
        )

        assert msg_id == "12345-0"
        call_args = mock_runner.publish_event.call_args
        assert call_args[1]["event_type"] == "error"
        assert call_args[1]["data"]["error_type"] == "rate_limit"
        assert call_args[1]["data"]["details"]["retry_after"] == 60
