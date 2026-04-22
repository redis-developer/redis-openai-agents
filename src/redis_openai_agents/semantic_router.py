"""Semantic Router using RedisVL for intent classification.

This module provides a wrapper around RedisVL's SemanticRouter that:
- Uses vector similarity search to classify statements into predefined routes
- Supports multiple references per route for better coverage
- Provides aggregation methods for combining reference distances
- Enables route management (add/remove references)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from redisvl.extensions.router import (  # type: ignore[import-untyped]
    SemanticRouter as RedisvlSemanticRouter,
)
from redisvl.extensions.router.schema import (  # type: ignore[import-untyped]
    DistanceAggregationMethod,
)
from redisvl.extensions.router.schema import (
    Route as RedisvlRoute,
)
from redisvl.extensions.router.schema import (
    RouteMatch as RedisvlRouteMatch,
)
from redisvl.utils.vectorize import HFTextVectorizer  # type: ignore[import-untyped]


class Route(BaseModel):
    """Definition of a route for semantic classification.

    A route represents a category or intent that statements can be classified into.
    Each route has a name, a list of reference phrases that exemplify the route,
    optional metadata, and a distance threshold for matching.

    Attributes:
        name: Unique identifier for this route.
        references: List of example phrases that should match this route.
        metadata: Optional key-value pairs associated with this route.
        distance_threshold: Maximum distance for a match (0-2, lower is stricter).
    """

    name: str
    references: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance_threshold: float = Field(default=0.5, gt=0, le=2)


@dataclass
class RouteMatch:
    """Result of routing a statement.

    Attributes:
        name: Name of the matched route, or None if no match.
        distance: Distance score (lower is more similar).
    """

    name: str | None = None
    distance: float | None = None


class SemanticRouter:
    """Semantic router for classifying statements into predefined routes.

    Uses vector similarity search via RedisVL to match input statements
    to the most appropriate route based on semantic similarity to
    reference phrases.

    Example::

        routes = [
            Route(
                name="technology",
                references=["AI news", "programming tips", "tech updates"],
            ),
            Route(
                name="sports",
                references=["football scores", "basketball highlights"],
            ),
        ]

        async with SemanticRouter(
            name="topic-router",
            routes=routes,
            redis_url="redis://localhost:6379",
        ) as router:
            result = await router("What's new in machine learning?")
            print(result.name)  # "technology"
    """

    def __init__(
        self,
        name: str,
        routes: list[Route],
        redis_url: str = "redis://localhost:6379",
        vectorizer: Any | None = None,
        overwrite: bool = True,
        routing_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the semantic router.

        Args:
            name: Unique name for this router (used as Redis index prefix).
            routes: List of Route objects defining classification categories.
            redis_url: Redis connection URL.
            vectorizer: Optional vectorizer instance. Defaults to HuggingFace
                sentence-transformers/all-MiniLM-L6-v2.
            overwrite: If True, overwrite existing index with same name.
                If False, raise ValueError if index exists.
            routing_config: Optional configuration for routing behavior
                (max_k, aggregation_method).

        Raises:
            ValueError: If routes is empty, has duplicates, or index exists
                and overwrite=False.
        """
        self._validate_routes(routes)

        self._name = name
        self._routes = {r.name: r for r in routes}
        self._redis_url = redis_url
        self._overwrite = overwrite

        # Use default vectorizer if not provided
        if vectorizer is None:
            vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
        self._vectorizer = vectorizer

        # Convert to RedisVL routes
        redisvl_routes = [
            RedisvlRoute(
                name=r.name,
                references=r.references,
                metadata=r.metadata,
                distance_threshold=r.distance_threshold,
            )
            for r in routes
        ]

        # Create underlying router
        try:
            self._router = RedisvlSemanticRouter(
                name=name,
                routes=redisvl_routes,
                vectorizer=vectorizer,
                redis_url=redis_url,
                overwrite=overwrite,
                routing_config=routing_config,
            )
        except Exception as e:
            if "already exists" in str(e).lower() and not overwrite:
                raise ValueError(f"Index '{name}' already exists") from e
            raise

    def _validate_routes(self, routes: list[Route]) -> None:
        """Validate route configuration.

        Args:
            routes: List of routes to validate.

        Raises:
            ValueError: If validation fails.
        """
        if not routes:
            raise ValueError("Must provide at least one route")

        names = [r.name for r in routes]
        if len(names) != len(set(names)):
            raise ValueError("Route names must be unique, found duplicate names")

        for route in routes:
            if not route.references:
                raise ValueError(f"Route '{route.name}' must have at least one reference")

    @property
    def name(self) -> str:
        """Get router name."""
        return self._name

    @property
    def route_names(self) -> list[str]:
        """Get list of all route names."""
        return list(self._routes.keys())

    def get_route(self, name: str) -> Route | None:
        """Get route by name.

        Args:
            name: Route name to look up.

        Returns:
            Route object if found, None otherwise.
        """
        return self._routes.get(name)

    async def __call__(
        self,
        statement: str,
        aggregation_method: Literal["avg", "min", "sum"] | None = None,
    ) -> RouteMatch:
        """Route a statement to the best matching route.

        Args:
            statement: Text to classify.
            aggregation_method: How to combine distances for routes with
                multiple references ("avg", "min", or "sum").

        Returns:
            RouteMatch with the matched route name and distance.
        """
        kwargs: dict[str, Any] = {}
        if aggregation_method:
            kwargs["aggregation_method"] = DistanceAggregationMethod(aggregation_method)

        # Run sync operation in thread pool
        result: RedisvlRouteMatch = await asyncio.to_thread(self._router, statement, **kwargs)

        return RouteMatch(name=result.name, distance=result.distance)

    async def route_many(
        self,
        statement: str,
        max_k: int | None = None,
        aggregation_method: Literal["avg", "min", "sum"] | None = None,
    ) -> list[RouteMatch]:
        """Route a statement to multiple matching routes.

        Args:
            statement: Text to classify.
            max_k: Maximum number of routes to return.
            aggregation_method: How to combine distances for routes with
                multiple references.

        Returns:
            List of RouteMatch objects, ordered by distance (closest first).
        """
        kwargs: dict[str, Any] = {}
        if max_k is not None:
            kwargs["max_k"] = max_k
        if aggregation_method:
            kwargs["aggregation_method"] = DistanceAggregationMethod(aggregation_method)

        # Run sync operation in thread pool
        results: list[RedisvlRouteMatch] = await asyncio.to_thread(
            self._router.route_many, statement, **kwargs
        )

        return [RouteMatch(name=r.name, distance=r.distance) for r in results]

    async def add_route_references(
        self,
        route_name: str,
        references: list[str],
    ) -> list[str]:
        """Add new references to an existing route.

        Args:
            route_name: Name of the route to update.
            references: New reference phrases to add.

        Returns:
            List of IDs for the added references.

        Raises:
            ValueError: If route_name doesn't exist.
        """
        if route_name not in self._routes:
            raise ValueError(f"Route '{route_name}' not found")

        result: list[str] = await asyncio.to_thread(
            self._router.add_route_references,
            route_name,
            references,
        )

        return result

    async def get_route_references(
        self,
        route_name: str,
        reference_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get references for a route.

        Args:
            route_name: Name of the route.
            reference_ids: Optional list of specific reference IDs to retrieve.
                If None, returns all references.

        Returns:
            List of dicts with "id" and "reference" keys.

        Raises:
            ValueError: If route_name doesn't exist.
        """
        if route_name not in self._routes:
            raise ValueError(f"Route '{route_name}' not found")

        kwargs: dict[str, Any] = {}
        if reference_ids is not None:
            kwargs["reference_ids"] = reference_ids

        result: list[dict[str, Any]] = await asyncio.to_thread(
            self._router.get_route_references,
            route_name,
            **kwargs,
        )

        return result

    async def delete_route_references(
        self,
        route_name: str,
        reference_ids: list[str] | None = None,
        keys: list[str] | None = None,
    ) -> int:
        """Delete references from a route.

        Args:
            route_name: Name of the route.
            reference_ids: Specific reference IDs to delete.
            keys: Specific Redis keys to delete.
                At least one of reference_ids or keys should be provided.

        Returns:
            Number of references deleted.

        Raises:
            ValueError: If route_name doesn't exist.
        """
        if route_name not in self._routes:
            raise ValueError(f"Route '{route_name}' not found")

        result: int = await asyncio.to_thread(
            self._router.delete_route_references,
            route_name,
            reference_ids=reference_ids,
            keys=keys,
        )

        return result

    async def remove_route(self, route_name: str) -> None:
        """Remove an entire route and its references.

        Args:
            route_name: Name of the route to remove.
        """
        await asyncio.to_thread(self._router.remove_route, route_name)

        # Update local state
        if route_name in self._routes:
            del self._routes[route_name]

    def to_dict(self) -> dict[str, Any]:
        """Serialize router configuration to dict.

        Returns:
            Dict with router name and routes configuration.
        """
        return {
            "name": self._name,
            "routes": [
                {
                    "name": r.name,
                    "references": r.references,
                    "metadata": r.metadata,
                    "distance_threshold": r.distance_threshold,
                }
                for r in self._routes.values()
            ],
        }

    @classmethod
    async def from_dict(
        cls,
        data: dict[str, Any],
        redis_url: str = "redis://localhost:6379",
        vectorizer: Any | None = None,
        overwrite: bool = True,
    ) -> SemanticRouter:
        """Create router from dict configuration.

        Args:
            data: Dict with "name" and "routes" keys.
            redis_url: Redis connection URL.
            vectorizer: Optional vectorizer instance.
            overwrite: If True, overwrite existing index.

        Returns:
            New SemanticRouter instance.
        """
        routes = [
            Route(
                name=r["name"],
                references=r["references"],
                metadata=r.get("metadata", {}),
                distance_threshold=r.get("distance_threshold", 0.5),
            )
            for r in data["routes"]
        ]

        return cls(
            name=data["name"],
            routes=routes,
            redis_url=redis_url,
            vectorizer=vectorizer,
            overwrite=overwrite,
        )

    async def close(self) -> None:
        """Close the router and release resources."""
        # RedisVL router handles cleanup internally
        pass

    async def __aenter__(self) -> SemanticRouter:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()
