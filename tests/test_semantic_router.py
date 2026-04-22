"""Unit tests for SemanticRouter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from redis_openai_agents.semantic_router import Route, RouteMatch, SemanticRouter


class TestRoute:
    """Test Route model."""

    def test_create_route(self) -> None:
        """Can create Route with required fields."""
        route = Route(
            name="test",
            references=["ref1", "ref2"],
        )

        assert route.name == "test"
        assert route.references == ["ref1", "ref2"]
        assert route.metadata == {}
        assert route.distance_threshold == 0.5

    def test_route_with_metadata(self) -> None:
        """Route accepts metadata."""
        route = Route(
            name="test",
            references=["ref"],
            metadata={"key": "value"},
        )

        assert route.metadata == {"key": "value"}

    def test_route_with_threshold(self) -> None:
        """Route accepts custom distance_threshold."""
        route = Route(
            name="test",
            references=["ref"],
            distance_threshold=0.8,
        )

        assert route.distance_threshold == 0.8

    def test_route_threshold_validation(self) -> None:
        """Route validates distance_threshold range."""
        # Valid thresholds
        Route(name="t", references=["r"], distance_threshold=0.1)
        Route(name="t", references=["r"], distance_threshold=2.0)

        # Invalid thresholds
        with pytest.raises(ValueError):
            Route(name="t", references=["r"], distance_threshold=0.0)

        with pytest.raises(ValueError):
            Route(name="t", references=["r"], distance_threshold=2.1)


class TestRouteMatch:
    """Test RouteMatch dataclass."""

    def test_create_route_match(self) -> None:
        """Can create RouteMatch."""
        match = RouteMatch(name="technology", distance=0.35)

        assert match.name == "technology"
        assert match.distance == 0.35

    def test_route_match_defaults(self) -> None:
        """RouteMatch has None defaults."""
        match = RouteMatch()

        assert match.name is None
        assert match.distance is None


class TestSemanticRouterValidation:
    """Test SemanticRouter validation."""

    def test_validate_empty_routes(self) -> None:
        """Raises error for empty routes."""
        with pytest.raises(ValueError, match="at least one route"):
            router = SemanticRouter.__new__(SemanticRouter)
            router._validate_routes([])

    def test_validate_duplicate_names(self) -> None:
        """Raises error for duplicate route names."""
        routes = [
            Route(name="dup", references=["r1"]),
            Route(name="dup", references=["r2"]),
        ]

        with pytest.raises(ValueError, match="duplicate"):
            router = SemanticRouter.__new__(SemanticRouter)
            router._validate_routes(routes)

    def test_validate_empty_references(self) -> None:
        """Raises error for route with no references."""
        routes = [Route(name="empty", references=[])]

        with pytest.raises(ValueError, match="reference"):
            router = SemanticRouter.__new__(SemanticRouter)
            router._validate_routes(routes)


class TestSemanticRouterProperties:
    """Test SemanticRouter properties."""

    def test_name_property(self) -> None:
        """Name property returns router name."""
        router = SemanticRouter.__new__(SemanticRouter)
        router._name = "test_router"

        assert router.name == "test_router"

    def test_route_names_property(self) -> None:
        """route_names returns list of route names."""
        router = SemanticRouter.__new__(SemanticRouter)
        router._routes = {
            "tech": Route(name="tech", references=["r"]),
            "sports": Route(name="sports", references=["r"]),
        }

        names = router.route_names
        assert "tech" in names
        assert "sports" in names
        assert len(names) == 2

    def test_get_route(self) -> None:
        """get_route returns Route by name."""
        route = Route(name="tech", references=["r"])
        router = SemanticRouter.__new__(SemanticRouter)
        router._routes = {"tech": route}

        assert router.get_route("tech") == route
        assert router.get_route("nonexistent") is None


class TestSemanticRouterToDict:
    """Test serialization."""

    def test_to_dict(self) -> None:
        """to_dict serializes router configuration."""
        router = SemanticRouter.__new__(SemanticRouter)
        router._name = "test"
        router._routes = {
            "topic": Route(
                name="topic",
                references=["ref1", "ref2"],
                metadata={"key": "value"},
                distance_threshold=0.7,
            )
        }

        data = router.to_dict()

        assert data["name"] == "test"
        assert len(data["routes"]) == 1
        assert data["routes"][0]["name"] == "topic"
        assert data["routes"][0]["references"] == ["ref1", "ref2"]
        assert data["routes"][0]["metadata"] == {"key": "value"}
        assert data["routes"][0]["distance_threshold"] == 0.7


class TestSemanticRouterRouteCallValidation:
    """Test validation in routing methods."""

    async def test_add_route_references_nonexistent(self) -> None:
        """add_route_references raises for nonexistent route."""
        router = SemanticRouter.__new__(SemanticRouter)
        router._routes = {}

        with pytest.raises(ValueError, match="not found"):
            await router.add_route_references("nonexistent", ["ref"])

    async def test_get_route_references_nonexistent(self) -> None:
        """get_route_references raises for nonexistent route."""
        router = SemanticRouter.__new__(SemanticRouter)
        router._routes = {}

        with pytest.raises(ValueError, match="not found"):
            await router.get_route_references("nonexistent")

    async def test_delete_route_references_nonexistent(self) -> None:
        """delete_route_references raises for nonexistent route."""
        router = SemanticRouter.__new__(SemanticRouter)
        router._routes = {}

        with pytest.raises(ValueError, match="not found"):
            await router.delete_route_references("nonexistent")


class TestSemanticRouterRemoveRoute:
    """Test remove_route updates local state."""

    async def test_remove_route_updates_routes_dict(self) -> None:
        """remove_route removes route from local dict."""
        router = SemanticRouter.__new__(SemanticRouter)
        router._routes = {
            "keep": Route(name="keep", references=["r"]),
            "remove": Route(name="remove", references=["r"]),
        }
        mock_underlying = MagicMock()
        mock_underlying.remove_route = MagicMock()
        router._router = mock_underlying

        await router.remove_route("remove")

        assert "keep" in router._routes
        assert "remove" not in router._routes


class TestSemanticRouterContextManager:
    """Test async context manager."""

    async def test_aenter_returns_self(self) -> None:
        """__aenter__ returns the router instance."""
        router = SemanticRouter.__new__(SemanticRouter)

        result = await router.__aenter__()

        assert result is router

    async def test_aexit_calls_close(self) -> None:
        """__aexit__ calls close."""
        router = SemanticRouter.__new__(SemanticRouter)
        router.close = AsyncMock()

        await router.__aexit__(None, None, None)

        router.close.assert_called_once()


class TestSemanticRouterClose:
    """Test close method."""

    async def test_close_succeeds(self) -> None:
        """close() completes without error."""
        router = SemanticRouter.__new__(SemanticRouter)

        # Should not raise
        await router.close()
