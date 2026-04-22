"""Unit tests for RedisStreamTransport - TDD RED phase."""

from redis_openai_agents import RedisStreamTransport


class TestStreamTransportBasics:
    """Test basic stream transport operations."""

    def test_create_stream_transport(self, redis_url: str) -> None:
        """Can create a stream transport."""
        stream = RedisStreamTransport(
            stream_name="test_stream",
            redis_url=redis_url,
        )
        assert stream.stream_name == "test_stream"

    def test_create_with_consumer_group(self, redis_url: str) -> None:
        """Can create with custom consumer group."""
        stream = RedisStreamTransport(
            stream_name="test_stream_cg",
            redis_url=redis_url,
            consumer_group="my_group",
        )
        assert stream.consumer_group == "my_group"


class TestEventPublishing:
    """Test event publishing."""

    def test_publish_event(self, redis_url: str) -> None:
        """Can publish an event."""
        stream = RedisStreamTransport(
            stream_name="test_publish",
            redis_url=redis_url,
        )

        msg_id = stream.publish(event_type="token", data={"token": "hello"})

        assert msg_id is not None
        assert "-" in msg_id  # Redis stream ID format

    def test_publish_multiple_events(self, redis_url: str) -> None:
        """Can publish multiple events."""
        stream = RedisStreamTransport(
            stream_name="test_multi_publish",
            redis_url=redis_url,
        )

        ids = []
        for i in range(5):
            msg_id = stream.publish(event_type="token", data={"token": f"word{i}"})
            ids.append(msg_id)

        assert len(ids) == 5
        assert len(set(ids)) == 5  # All unique

    def test_publish_with_metadata(self, redis_url: str) -> None:
        """Can publish event with metadata."""
        stream = RedisStreamTransport(
            stream_name="test_metadata",
            redis_url=redis_url,
        )

        msg_id = stream.publish(
            event_type="tool_call",
            data={"tool": "search", "args": {"query": "test"}},
            metadata={"agent": "assistant", "conversation_id": "123"},
        )

        assert msg_id is not None


class TestEventConsumption:
    """Test event consumption."""

    def test_read_all_events(self, redis_url: str) -> None:
        """Can read all events from stream."""
        stream = RedisStreamTransport(
            stream_name="test_read_all",
            redis_url=redis_url,
        )

        # Publish some events
        stream.publish(event_type="token", data={"token": "a"})
        stream.publish(event_type="token", data={"token": "b"})
        stream.publish(event_type="complete", data={"reason": "done"})

        events = stream.read_all(count=10)

        assert len(events) == 3
        assert events[0]["type"] == "token"
        assert events[2]["type"] == "complete"

    def test_read_events_have_id(self, redis_url: str) -> None:
        """Read events include message ID."""
        stream = RedisStreamTransport(
            stream_name="test_read_id",
            redis_url=redis_url,
        )

        stream.publish(event_type="token", data={"token": "test"})

        events = stream.read_all(count=1)

        assert len(events) == 1
        assert "id" in events[0]
        assert "-" in events[0]["id"]

    def test_read_events_have_data(self, redis_url: str) -> None:
        """Read events include parsed data."""
        stream = RedisStreamTransport(
            stream_name="test_read_data",
            redis_url=redis_url,
        )

        stream.publish(event_type="token", data={"token": "hello", "index": 0})

        events = stream.read_all(count=1)

        assert events[0]["data"]["token"] == "hello"
        assert events[0]["data"]["index"] == 0

    def test_read_empty_stream(self, redis_url: str) -> None:
        """Reading empty stream returns empty list."""
        stream = RedisStreamTransport(
            stream_name="test_empty_read",
            redis_url=redis_url,
        )

        events = stream.read_all(count=10)

        assert events == []


class TestStreamInfo:
    """Test stream info operations."""

    def test_get_stream_info(self, redis_url: str) -> None:
        """Can get stream info."""
        stream = RedisStreamTransport(
            stream_name="test_info",
            redis_url=redis_url,
        )

        stream.publish(event_type="token", data={"token": "test"})

        info = stream.info()

        assert "length" in info
        assert info["length"] >= 1

    def test_info_includes_groups(self, redis_url: str) -> None:
        """Stream info includes consumer groups."""
        stream = RedisStreamTransport(
            stream_name="test_info_groups",
            redis_url=redis_url,
            consumer_group="test_group",
        )

        stream.publish(event_type="token", data={"token": "test"})

        info = stream.info()

        assert "groups" in info


class TestStreamDeletion:
    """Test stream deletion."""

    def test_delete_stream(self, redis_url: str) -> None:
        """Can delete stream."""
        stream = RedisStreamTransport(
            stream_name="test_delete",
            redis_url=redis_url,
        )

        stream.publish(event_type="token", data={"token": "test"})
        info_before = stream.info()
        assert info_before["length"] >= 1

        stream.delete()

        info_after = stream.info()
        assert info_after["length"] == 0


class TestConsumerGroups:
    """Test consumer group functionality (XREADGROUP/XACK pattern)."""

    def test_read_group_returns_events(self, redis_url: str) -> None:
        """Can read events using consumer group."""
        stream = RedisStreamTransport(
            stream_name="test_read_group",
            redis_url=redis_url,
            consumer_group="test_cg",
        )

        # Publish events
        stream.publish(event_type="token", data={"token": "hello"})
        stream.publish(event_type="token", data={"token": "world"})

        # Read using consumer group
        events = stream.read_group(consumer="consumer1", count=10)

        assert len(events) == 2
        assert events[0]["data"]["token"] == "hello"
        assert events[1]["data"]["token"] == "world"

    def test_read_group_events_are_pending(self, redis_url: str) -> None:
        """Read but unacknowledged events are pending."""
        stream = RedisStreamTransport(
            stream_name="test_pending",
            redis_url=redis_url,
            consumer_group="test_cg_pending",
        )

        stream.publish(event_type="token", data={"token": "test"})

        # Read but don't ACK
        events = stream.read_group(consumer="consumer1", count=10)
        assert len(events) == 1

        # Check pending
        pending = stream.pending()
        assert pending["count"] >= 1

    def test_ack_removes_from_pending(self, redis_url: str) -> None:
        """ACK removes events from pending."""
        stream = RedisStreamTransport(
            stream_name="test_ack",
            redis_url=redis_url,
            consumer_group="test_cg_ack",
        )

        msg_id = stream.publish(event_type="token", data={"token": "ack_me"})

        # Read using consumer group
        events = stream.read_group(consumer="consumer1", count=10)
        assert len(events) == 1

        # ACK the message
        acked = stream.ack([msg_id])
        assert acked == 1

        # Verify not pending anymore
        pending = stream.pending()
        assert pending["count"] == 0

    def test_ack_multiple_messages(self, redis_url: str) -> None:
        """Can ACK multiple messages at once."""
        stream = RedisStreamTransport(
            stream_name="test_multi_ack",
            redis_url=redis_url,
            consumer_group="test_cg_multi",
        )

        # Publish and read
        ids = []
        for i in range(3):
            ids.append(stream.publish(event_type="token", data={"i": i}))

        stream.read_group(consumer="consumer1", count=10)

        # ACK all
        acked = stream.ack(ids)
        assert acked == 3

    def test_read_group_blocks_when_empty(self, redis_url: str) -> None:
        """read_group with block=0 returns immediately if empty."""
        stream = RedisStreamTransport(
            stream_name="test_block_empty",
            redis_url=redis_url,
            consumer_group="test_cg_block",
        )

        # Clean start - read any existing
        import time

        start = time.time()
        stream.read_group(consumer="consumer1", count=10, block_ms=100)
        elapsed = time.time() - start

        # Should have blocked for ~100ms
        assert elapsed >= 0.05  # Lower bound: at least ~50ms of blocking
        assert elapsed < 1.0  # Should not take too long

    def test_multiple_consumers_receive_different_events(self, redis_url: str) -> None:
        """Different consumers in same group get different messages."""
        stream = RedisStreamTransport(
            stream_name="test_multi_consumer",
            redis_url=redis_url,
            consumer_group="test_cg_multi_consumer",
        )

        # Publish events
        for i in range(4):
            stream.publish(event_type="token", data={"i": i})

        # Consumer 1 reads first 2
        events1 = stream.read_group(consumer="consumer1", count=2)

        # Consumer 2 reads next 2
        events2 = stream.read_group(consumer="consumer2", count=2)

        assert len(events1) == 2
        assert len(events2) == 2

        # Different events
        ids1 = {e["id"] for e in events1}
        ids2 = {e["id"] for e in events2}
        assert ids1.isdisjoint(ids2)

    def test_claim_pending_from_dead_consumer(self, redis_url: str) -> None:
        """Can claim pending messages from a dead consumer."""
        stream = RedisStreamTransport(
            stream_name="test_claim",
            redis_url=redis_url,
            consumer_group="test_cg_claim",
        )

        msg_id = stream.publish(event_type="token", data={"token": "claim_me"})

        # Dead consumer reads but doesn't ACK
        stream.read_group(consumer="dead_consumer", count=10)

        # New consumer claims the message
        claimed = stream.claim(
            consumer="new_consumer",
            min_idle_ms=0,  # Claim immediately
            ids=[msg_id],
        )

        assert len(claimed) == 1
        assert claimed[0]["id"] == msg_id
