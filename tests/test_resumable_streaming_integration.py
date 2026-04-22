"""Integration tests for ResumableStreamRunner - TDD RED phase.

Tests resumable LLM streaming backed by Redis Streams.
"""

import time


class TestResumableStreamRunnerPublishing:
    """Test event publishing to Redis Streams."""

    async def test_publish_text_delta_event(self, redis_url: str) -> None:
        """Can publish text delta events."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_publish",
        )
        await runner.initialize()

        session_id = "session_publish_1"
        await runner.publish_event(
            session_id=session_id,
            event_type="text_delta",
            data={"delta": "Hello"},
        )

        # Verify event was published
        events = await runner.get_all_events(session_id)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["data"]["delta"] == "Hello"

        await runner.close()

    async def test_publish_multiple_events(self, redis_url: str) -> None:
        """Can publish multiple events in sequence."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_multi",
        )
        await runner.initialize()

        session_id = "session_multi_1"

        # Publish multiple text deltas
        for word in ["Hello", " ", "world", "!"]:
            await runner.publish_event(
                session_id=session_id,
                event_type="text_delta",
                data={"delta": word},
            )

        events = await runner.get_all_events(session_id)
        assert len(events) == 4

        # Reconstruct the message
        full_text = "".join(e["data"]["delta"] for e in events)
        assert full_text == "Hello world!"

        await runner.close()

    async def test_publish_different_event_types(self, redis_url: str) -> None:
        """Can publish different event types."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_types",
        )
        await runner.initialize()

        session_id = "session_types_1"

        await runner.publish_event(
            session_id=session_id,
            event_type="stream_start",
            data={"agent": "assistant"},
        )
        await runner.publish_event(
            session_id=session_id,
            event_type="text_delta",
            data={"delta": "Hi"},
        )
        await runner.publish_event(
            session_id=session_id,
            event_type="tool_call",
            data={"tool": "search", "args": {"query": "test"}},
        )
        await runner.publish_event(
            session_id=session_id,
            event_type="stream_end",
            data={"reason": "complete"},
        )

        events = await runner.get_all_events(session_id)
        assert len(events) == 4

        types = [e["type"] for e in events]
        assert types == ["stream_start", "text_delta", "tool_call", "stream_end"]

        await runner.close()


class TestResumableStreamRunnerSubscription:
    """Test subscribing to streaming events."""

    async def test_subscribe_receives_events(self, redis_url: str) -> None:
        """Subscriber receives published events."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_sub",
        )
        await runner.initialize()

        session_id = "session_sub_1"

        # Publish some events
        await runner.publish_event(
            session_id=session_id,
            event_type="text_delta",
            data={"delta": "Test"},
        )

        # Subscribe from beginning to receive existing events
        received = []
        async for event in runner.subscribe(session_id, from_id="0", timeout_ms=100):
            received.append(event)
            if len(received) >= 1:
                break

        assert len(received) == 1
        assert received[0]["data"]["delta"] == "Test"

        await runner.close()

    async def test_subscribe_from_beginning(self, redis_url: str) -> None:
        """Can subscribe from the beginning of the stream."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_begin",
        )
        await runner.initialize()

        session_id = "session_begin_1"

        # Publish events before subscribing
        for i in range(3):
            await runner.publish_event(
                session_id=session_id,
                event_type="text_delta",
                data={"delta": str(i)},
            )

        # Subscribe from beginning
        received = []
        async for event in runner.subscribe(session_id, from_id="0", timeout_ms=100):
            received.append(event)
            if len(received) >= 3:
                break

        assert len(received) == 3
        deltas = [e["data"]["delta"] for e in received]
        assert deltas == ["0", "1", "2"]

        await runner.close()

    async def test_subscribe_from_specific_id(self, redis_url: str) -> None:
        """Can resume from a specific message ID."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_resume",
        )
        await runner.initialize()

        session_id = "session_resume_1"

        # Publish events and capture IDs
        message_ids = []
        for i in range(5):
            msg_id = await runner.publish_event(
                session_id=session_id,
                event_type="text_delta",
                data={"delta": str(i)},
            )
            message_ids.append(msg_id)

        # Subscribe from middle (after message 2)
        received = []
        async for event in runner.subscribe(session_id, from_id=message_ids[1], timeout_ms=100):
            received.append(event)
            if len(received) >= 3:
                break

        # Should get messages 2, 3, 4 (after message_ids[1])
        assert len(received) == 3
        deltas = [e["data"]["delta"] for e in received]
        assert deltas == ["2", "3", "4"]

        await runner.close()


class TestResumableStreamRunnerConsumerGroups:
    """Test consumer group functionality for tracking progress."""

    async def test_consumer_tracks_progress(self, redis_url: str) -> None:
        """Consumer group tracks which messages have been consumed."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_consumer",
        )
        await runner.initialize()

        session_id = "session_consumer_1"
        consumer_id = "client_1"

        # Publish events
        for i in range(3):
            await runner.publish_event(
                session_id=session_id,
                event_type="text_delta",
                data={"delta": str(i)},
            )

        # First consumer reads all
        received = []
        async for event in runner.subscribe_as_consumer(
            session_id, consumer_id=consumer_id, timeout_ms=100
        ):
            received.append(event)
            await runner.ack(session_id, consumer_id, event["id"])
            if len(received) >= 3:
                break

        assert len(received) == 3

        # Second read should get no new messages (all acknowledged)
        more_events = []
        async for event in runner.subscribe_as_consumer(
            session_id, consumer_id=consumer_id, timeout_ms=100
        ):
            more_events.append(event)

        assert len(more_events) == 0

        await runner.close()

    async def test_consumers_share_messages_for_load_balancing(self, redis_url: str) -> None:
        """Consumer group distributes messages between consumers (load balancing)."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_load_balance",
        )
        await runner.initialize()

        session_id = "session_load_balance_1"

        # Publish events
        for i in range(3):
            await runner.publish_event(
                session_id=session_id,
                event_type="text_delta",
                data={"delta": str(i)},
            )

        # Consumer A reads all (gets all since B hasn't started)
        received_a = []
        async for event in runner.subscribe_as_consumer(
            session_id, consumer_id="client_a", timeout_ms=100
        ):
            received_a.append(event)
            await runner.ack(session_id, "client_a", event["id"])
            if len(received_a) >= 3:
                break

        # Consumer B won't get any - messages already consumed by A
        # This is expected behavior for consumer groups (load balancing)
        received_b = []
        async for event in runner.subscribe_as_consumer(
            session_id, consumer_id="client_b", timeout_ms=100
        ):
            received_b.append(event)
            await runner.ack(session_id, "client_b", event["id"])

        assert len(received_a) == 3
        # Consumer B gets nothing - messages already distributed to A
        assert len(received_b) == 0

        await runner.close()

    async def test_multiple_independent_subscribers(self, redis_url: str) -> None:
        """Multiple subscribers can each see all events using regular subscribe."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_multi_sub",
        )
        await runner.initialize()

        session_id = "session_multi_sub_1"

        # Publish events
        for i in range(3):
            await runner.publish_event(
                session_id=session_id,
                event_type="text_delta",
                data={"delta": str(i)},
            )

        # Subscriber A reads all from beginning
        received_a = []
        async for event in runner.subscribe(session_id, from_id="0", timeout_ms=100):
            received_a.append(event)
            if len(received_a) >= 3:
                break

        # Subscriber B also reads all from beginning (independent)
        received_b = []
        async for event in runner.subscribe(session_id, from_id="0", timeout_ms=100):
            received_b.append(event)
            if len(received_b) >= 3:
                break

        # Both get all messages
        assert len(received_a) == 3
        assert len(received_b) == 3

        await runner.close()


class TestResumableStreamRunnerReconnection:
    """Test reconnection and resumption scenarios."""

    async def test_reconnect_receives_missed_events(self, redis_url: str) -> None:
        """Reconnecting consumer receives events missed during disconnect."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_reconnect",
        )
        await runner.initialize()

        session_id = "session_reconnect_1"
        consumer_id = "reconnecting_client"

        # Publish first batch
        for i in range(2):
            await runner.publish_event(
                session_id=session_id,
                event_type="text_delta",
                data={"delta": f"batch1_{i}"},
            )

        # Consumer reads first batch
        async for event in runner.subscribe_as_consumer(
            session_id, consumer_id=consumer_id, timeout_ms=100
        ):
            event["id"]
            await runner.ack(session_id, consumer_id, event["id"])
            break  # Simulate disconnect after first message

        # More events arrive while disconnected
        for i in range(3):
            await runner.publish_event(
                session_id=session_id,
                event_type="text_delta",
                data={"delta": f"batch2_{i}"},
            )

        # Reconnect - should get remaining events
        received = []
        async for event in runner.subscribe_as_consumer(
            session_id, consumer_id=consumer_id, timeout_ms=100
        ):
            received.append(event)
            await runner.ack(session_id, consumer_id, event["id"])
            if len(received) >= 4:  # 1 from batch1 + 3 from batch2
                break

        # Should receive the unacked message from batch1 and all batch2
        assert len(received) >= 4

        await runner.close()


class TestResumableStreamRunnerMetadata:
    """Test metadata and stream info."""

    async def test_events_have_timestamps(self, redis_url: str) -> None:
        """Events include timestamp metadata."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_timestamp",
        )
        await runner.initialize()

        session_id = "session_ts_1"

        before = time.time()
        await runner.publish_event(
            session_id=session_id,
            event_type="text_delta",
            data={"delta": "test"},
        )
        after = time.time()

        events = await runner.get_all_events(session_id)
        assert len(events) == 1
        assert "timestamp" in events[0]
        assert before <= events[0]["timestamp"] <= after

        await runner.close()

    async def test_events_have_message_ids(self, redis_url: str) -> None:
        """Events include Redis Stream message IDs."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_msgid",
        )
        await runner.initialize()

        session_id = "session_msgid_1"

        await runner.publish_event(
            session_id=session_id,
            event_type="text_delta",
            data={"delta": "test"},
        )

        events = await runner.get_all_events(session_id)
        assert len(events) == 1
        assert "id" in events[0]
        # Redis Stream IDs have format: timestamp-sequence
        assert "-" in events[0]["id"]

        await runner.close()

    async def test_get_stream_info(self, redis_url: str) -> None:
        """Can get stream metadata."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_info",
        )
        await runner.initialize()

        session_id = "session_info_1"

        # Publish some events
        for i in range(5):
            await runner.publish_event(
                session_id=session_id,
                event_type="text_delta",
                data={"delta": str(i)},
            )

        info = await runner.get_stream_info(session_id)

        assert info["length"] == 5
        assert "first_entry_id" in info
        assert "last_entry_id" in info

        await runner.close()


class TestResumableStreamRunnerCleanup:
    """Test stream cleanup and expiration."""

    async def test_delete_stream(self, redis_url: str) -> None:
        """Can delete a stream."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_delete",
        )
        await runner.initialize()

        session_id = "session_delete_1"

        await runner.publish_event(
            session_id=session_id,
            event_type="text_delta",
            data={"delta": "test"},
        )

        # Verify stream exists
        events = await runner.get_all_events(session_id)
        assert len(events) == 1

        # Delete stream
        await runner.delete_stream(session_id)

        # Verify stream is gone
        events = await runner.get_all_events(session_id)
        assert len(events) == 0

        await runner.close()

    async def test_stream_with_max_length(self, redis_url: str) -> None:
        """Stream respects max length for automatic trimming."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_maxlen",
            max_stream_length=5,
        )
        await runner.initialize()

        session_id = "session_maxlen_1"

        # Publish more than max length
        for i in range(10):
            await runner.publish_event(
                session_id=session_id,
                event_type="text_delta",
                data={"delta": str(i)},
            )

        # Should only have max_length events
        events = await runner.get_all_events(session_id)
        assert len(events) <= 5

        await runner.close()


class TestResumableStreamRunnerIntegration:
    """Test integration with SDK streaming patterns."""

    async def test_wrap_stream_events(self, redis_url: str) -> None:
        """Can wrap SDK stream events for publishing."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_wrap",
        )
        await runner.initialize()

        session_id = "session_wrap_1"

        # Simulate SDK streaming events
        sdk_events = [
            {"type": "raw_response", "data": {"delta": "Hello"}},
            {"type": "raw_response", "data": {"delta": " world"}},
            {"type": "message_output_created", "data": {"content": "Hello world"}},
            {"type": "stream_complete", "data": {}},
        ]

        for event in sdk_events:
            await runner.publish_event(
                session_id=session_id,
                event_type=event["type"],
                data=event["data"],
            )

        # Verify all events stored
        stored_events = await runner.get_all_events(session_id)
        assert len(stored_events) == 4

        await runner.close()

    async def test_publish_returns_message_id(self, redis_url: str) -> None:
        """Publish returns the message ID for tracking."""
        from redis_openai_agents import ResumableStreamRunner

        runner = ResumableStreamRunner(
            redis_url=redis_url,
            stream_prefix="test_return_id",
        )
        await runner.initialize()

        session_id = "session_return_id_1"

        msg_id = await runner.publish_event(
            session_id=session_id,
            event_type="text_delta",
            data={"delta": "test"},
        )

        assert msg_id is not None
        assert isinstance(msg_id, str)
        assert "-" in msg_id  # Redis Stream ID format

        await runner.close()
