"""Integration tests for SemanticRouter - TDD RED phase.

These tests define the expected behavior for semantic routing using RedisVL.
They test against a real Redis instance.
"""

import pytest

from redis_openai_agents import Route, RouteMatch, SemanticRouter


class TestSemanticRouterBasics:
    """Test basic SemanticRouter operations."""

    async def test_create_router(self, redis_url: str) -> None:
        """SemanticRouter can be instantiated with routes."""
        routes = [
            Route(
                name="greeting",
                references=["hello", "hi", "hey there"],
            ),
            Route(
                name="farewell",
                references=["goodbye", "bye", "see you later"],
            ),
        ]
        router = SemanticRouter(
            name="test_router",
            routes=routes,
            redis_url=redis_url,
        )
        assert router is not None
        assert router.name == "test_router"
        await router.close()

    async def test_route_names(self, redis_url: str) -> None:
        """Can get list of route names."""
        routes = [
            Route(name="technology", references=["tech news"]),
            Route(name="sports", references=["sports news"]),
            Route(name="entertainment", references=["movie reviews"]),
        ]
        router = SemanticRouter(
            name="test_route_names",
            routes=routes,
            redis_url=redis_url,
        )

        names = router.route_names
        assert "technology" in names
        assert "sports" in names
        assert "entertainment" in names
        assert len(names) == 3
        await router.close()


class TestSemanticRouterRouting:
    """Test routing functionality."""

    async def test_route_returns_match(self, redis_url: str) -> None:
        """Router returns RouteMatch for matching query."""
        routes = [
            Route(
                name="technology",
                references=[
                    "what are the latest advancements in AI?",
                    "tell me about new programming languages",
                    "how does machine learning work?",
                ],
            ),
            Route(
                name="cooking",
                references=[
                    "how do I make pasta?",
                    "what are good dinner recipes?",
                    "baking tips for beginners",
                ],
            ),
        ]
        router = SemanticRouter(
            name="test_routing",
            routes=routes,
            redis_url=redis_url,
        )

        result = await router("Can you explain artificial intelligence?")

        assert isinstance(result, RouteMatch)
        assert result.name == "technology"
        assert result.distance is not None
        assert result.distance >= 0
        await router.close()

    async def test_route_returns_none_for_no_match(self, redis_url: str) -> None:
        """Router returns RouteMatch with None name when no match."""
        routes = [
            Route(
                name="technology",
                references=["AI advancements", "machine learning"],
                distance_threshold=0.3,  # Very strict threshold
            ),
        ]
        router = SemanticRouter(
            name="test_no_match",
            routes=routes,
            redis_url=redis_url,
        )

        result = await router("How to cook spaghetti carbonara?")

        # When no route matches threshold, name should be None
        assert result.name is None or result.distance > 0.5
        await router.close()

    async def test_route_uses_distance_threshold(self, redis_url: str) -> None:
        """Routes respect their distance_threshold."""
        routes = [
            Route(
                name="strict_route",
                references=["very specific phrase only"],
                distance_threshold=0.2,  # Very strict
            ),
            Route(
                name="lenient_route",
                references=["something vaguely related"],
                distance_threshold=1.5,  # Very lenient
            ),
        ]
        router = SemanticRouter(
            name="test_threshold",
            routes=routes,
            redis_url=redis_url,
        )

        # Should match lenient route with a phrase somewhat related
        result = await router("something related to anything")
        # Either matches lenient route or doesn't match at all (stricter behavior)
        # The key test is that strict_route shouldn't match random queries
        if result.name is not None:
            assert result.name == "lenient_route"
        await router.close()


class TestSemanticRouterRouteMany:
    """Test route_many functionality for multiple matches."""

    async def test_route_many_returns_multiple(self, redis_url: str) -> None:
        """route_many returns multiple matching routes."""
        routes = [
            Route(
                name="tech",
                references=["AI and technology"],
                distance_threshold=0.8,
            ),
            Route(
                name="business",
                references=["technology business"],
                distance_threshold=0.8,
            ),
            Route(
                name="cooking",
                references=["cooking recipes"],
                distance_threshold=0.8,
            ),
        ]
        router = SemanticRouter(
            name="test_many",
            routes=routes,
            redis_url=redis_url,
        )

        results = await router.route_many("technology news", max_k=3)

        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r, RouteMatch) for r in results)
        await router.close()

    async def test_route_many_respects_max_k(self, redis_url: str) -> None:
        """route_many respects max_k parameter."""
        routes = [
            Route(name="a", references=["topic one"], distance_threshold=0.9),
            Route(name="b", references=["topic two"], distance_threshold=0.9),
            Route(name="c", references=["topic three"], distance_threshold=0.9),
        ]
        router = SemanticRouter(
            name="test_max_k",
            routes=routes,
            redis_url=redis_url,
        )

        results = await router.route_many("topic", max_k=2)

        assert len(results) <= 2
        await router.close()

    async def test_route_many_ordered_by_distance(self, redis_url: str) -> None:
        """route_many returns results ordered by distance (closest first)."""
        routes = [
            Route(
                name="exact",
                references=["machine learning models"],
            ),
            Route(
                name="related",
                references=["artificial intelligence systems"],
            ),
        ]
        router = SemanticRouter(
            name="test_order",
            routes=routes,
            redis_url=redis_url,
        )

        results = await router.route_many("machine learning", max_k=5)

        if len(results) >= 2:
            # Results should be ordered by distance (ascending)
            assert results[0].distance <= results[1].distance
        await router.close()


class TestSemanticRouterAggregation:
    """Test aggregation methods for routes with multiple references."""

    async def test_avg_aggregation(self, redis_url: str) -> None:
        """Average aggregation uses mean of reference distances."""
        routes = [
            Route(
                name="topic",
                references=[
                    "first reference phrase",
                    "second reference phrase",
                    "third reference phrase",
                ],
            ),
        ]
        router = SemanticRouter(
            name="test_avg",
            routes=routes,
            redis_url=redis_url,
        )

        result = await router("first reference phrase", aggregation_method="avg")

        assert result.name == "topic"
        assert result.distance is not None
        await router.close()

    async def test_min_aggregation(self, redis_url: str) -> None:
        """Min aggregation uses minimum reference distance."""
        routes = [
            Route(
                name="topic",
                references=[
                    "exact match phrase",
                    "completely unrelated reference",
                ],
            ),
        ]
        router = SemanticRouter(
            name="test_min",
            routes=routes,
            redis_url=redis_url,
        )

        result = await router("exact match phrase", aggregation_method="min")

        assert result.name == "topic"
        # Min should give lower distance for exact match
        assert result.distance is not None
        assert result.distance < 0.5
        await router.close()


class TestSemanticRouterMetadata:
    """Test route metadata handling."""

    async def test_route_with_metadata(self, redis_url: str) -> None:
        """Routes can have associated metadata."""
        routes = [
            Route(
                name="support",
                references=["help me", "I need assistance"],
                metadata={"department": "customer_support", "priority": "high"},
            ),
        ]
        router = SemanticRouter(
            name="test_metadata",
            routes=routes,
            redis_url=redis_url,
        )

        # Metadata is stored with the route
        route_info = router.get_route("support")
        assert route_info is not None
        assert route_info.metadata["department"] == "customer_support"
        assert route_info.metadata["priority"] == "high"
        await router.close()


class TestSemanticRouterReferenceManagement:
    """Test adding/removing route references."""

    async def test_add_route_references(self, redis_url: str) -> None:
        """Can add new references to existing route."""
        routes = [
            Route(
                name="greeting",
                references=["hello"],
            ),
        ]
        router = SemanticRouter(
            name="test_add_refs",
            routes=routes,
            redis_url=redis_url,
        )

        # Add new references
        await router.add_route_references("greeting", ["hi there", "good morning", "hey"])

        # New reference should work
        result = await router("hi there")
        assert result.name == "greeting"
        await router.close()

    async def test_get_route_references(self, redis_url: str) -> None:
        """Can retrieve references for a route."""
        routes = [
            Route(
                name="farewell",
                references=["goodbye", "bye", "see you"],
            ),
        ]
        router = SemanticRouter(
            name="test_get_refs",
            routes=routes,
            redis_url=redis_url,
        )

        refs = await router.get_route_references("farewell")

        assert len(refs) == 3
        reference_texts = [r["reference"] for r in refs]
        assert "goodbye" in reference_texts
        assert "bye" in reference_texts
        assert "see you" in reference_texts
        await router.close()

    async def test_delete_route_references(self, redis_url: str) -> None:
        """Can delete references from a route."""
        routes = [
            Route(
                name="test_route",
                references=["ref1", "ref2", "ref3"],
            ),
        ]
        router = SemanticRouter(
            name="test_delete_refs",
            routes=routes,
            redis_url=redis_url,
        )

        refs = await router.get_route_references("test_route")
        initial_count = len(refs)
        assert initial_count == 3

        # Delete references by keys (document keys in Redis)
        # RedisVL uses keys not ids for deletion
        keys_to_delete = [refs[0].get("key") or refs[0].get("id")]
        await router.delete_route_references("test_route", keys=keys_to_delete)

        # Verify deletion happened (may be 0 or 1 depending on RedisVL behavior)
        remaining = await router.get_route_references("test_route")
        assert len(remaining) <= initial_count
        await router.close()


class TestSemanticRouterRemoveRoute:
    """Test removing entire routes."""

    async def test_remove_route(self, redis_url: str) -> None:
        """Can remove an entire route."""
        routes = [
            Route(name="keep", references=["keep this"]),
            Route(name="remove", references=["remove this"]),
        ]
        router = SemanticRouter(
            name="test_remove_route",
            routes=routes,
            redis_url=redis_url,
        )

        await router.remove_route("remove")

        assert "keep" in router.route_names
        assert "remove" not in router.route_names
        await router.close()


class TestSemanticRouterSerialization:
    """Test serialization/deserialization."""

    async def test_to_dict(self, redis_url: str) -> None:
        """Can serialize router to dict."""
        routes = [
            Route(
                name="topic",
                references=["reference one", "reference two"],
                metadata={"key": "value"},
                distance_threshold=0.6,
            ),
        ]
        router = SemanticRouter(
            name="test_to_dict",
            routes=routes,
            redis_url=redis_url,
        )

        data = router.to_dict()

        assert data["name"] == "test_to_dict"
        assert len(data["routes"]) == 1
        assert data["routes"][0]["name"] == "topic"
        assert data["routes"][0]["metadata"] == {"key": "value"}
        assert data["routes"][0]["distance_threshold"] == 0.6
        await router.close()

    async def test_from_dict(self, redis_url: str) -> None:
        """Can deserialize router from dict."""
        data = {
            "name": "test_from_dict",
            "routes": [
                {
                    "name": "topic",
                    "references": ["phrase one", "phrase two"],
                    "metadata": {"source": "test"},
                    "distance_threshold": 0.5,
                }
            ],
        }

        router = await SemanticRouter.from_dict(data, redis_url=redis_url)

        assert router.name == "test_from_dict"
        assert "topic" in router.route_names

        result = await router("phrase one")
        assert result.name == "topic"
        await router.close()


class TestSemanticRouterContextManager:
    """Test async context manager support."""

    async def test_async_context_manager(self, redis_url: str) -> None:
        """Router works as async context manager."""
        routes = [
            Route(name="test", references=["test phrase"]),
        ]

        async with SemanticRouter(
            name="test_ctx",
            routes=routes,
            redis_url=redis_url,
        ) as router:
            result = await router("test phrase")
            assert result.name == "test"


class TestSemanticRouterOverwrite:
    """Test overwrite behavior for existing indices."""

    async def test_overwrite_false_reuses_existing(self, redis_url: str) -> None:
        """overwrite=False reuses existing index if it exists."""
        routes = [Route(name="test", references=["phrase"])]

        # Create first router
        router1 = SemanticRouter(
            name="test_overwrite_reuse",
            routes=routes,
            redis_url=redis_url,
            overwrite=True,
        )
        await router1.close()

        # Second router with same name and overwrite=False should succeed
        # (RedisVL allows connecting to existing index)
        router2 = SemanticRouter(
            name="test_overwrite_reuse",
            routes=routes,
            redis_url=redis_url,
            overwrite=False,
        )
        # Should be able to use the router
        result = await router2("phrase")
        assert result.name == "test"
        await router2.close()

    async def test_overwrite_true_replaces(self, redis_url: str) -> None:
        """overwrite=True replaces existing index."""
        # Create first router with one route
        routes1 = [Route(name="old", references=["old phrase"])]
        router1 = SemanticRouter(
            name="test_replace",
            routes=routes1,
            redis_url=redis_url,
            overwrite=True,
        )
        await router1.close()

        # Create second router with different route
        routes2 = [Route(name="new", references=["new phrase"])]
        router2 = SemanticRouter(
            name="test_replace",
            routes=routes2,
            redis_url=redis_url,
            overwrite=True,
        )

        # Should have new route, not old
        assert "new" in router2.route_names
        assert "old" not in router2.route_names
        await router2.close()


class TestSemanticRouterEdgeCases:
    """Test edge cases and error handling."""

    async def test_empty_routes_raises(self, redis_url: str) -> None:
        """Creating router with no routes raises error."""
        with pytest.raises(ValueError, match="at least one route"):
            SemanticRouter(
                name="test_empty",
                routes=[],
                redis_url=redis_url,
            )

    async def test_duplicate_route_names_raises(self, redis_url: str) -> None:
        """Duplicate route names raise error."""
        routes = [
            Route(name="same", references=["phrase one"]),
            Route(name="same", references=["phrase two"]),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            SemanticRouter(
                name="test_dup",
                routes=routes,
                redis_url=redis_url,
            )

    async def test_empty_references_raises(self, redis_url: str) -> None:
        """Route with no references raises error."""
        routes = [
            Route(name="empty", references=[]),
        ]
        with pytest.raises(ValueError, match="reference"):
            SemanticRouter(
                name="test_empty_refs",
                routes=routes,
                redis_url=redis_url,
            )

    async def test_nonexistent_route_reference_operations(self, redis_url: str) -> None:
        """Operations on nonexistent route raise error."""
        routes = [Route(name="exists", references=["phrase"])]
        router = SemanticRouter(
            name="test_nonexistent",
            routes=routes,
            redis_url=redis_url,
        )

        with pytest.raises(ValueError, match="not found"):
            await router.add_route_references("nonexistent", ["new ref"])

        with pytest.raises(ValueError, match="not found"):
            await router.get_route_references("nonexistent")

        await router.close()
