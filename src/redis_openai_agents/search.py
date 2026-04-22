"""RedisFullTextSearch - BM25-based full-text search.

This module provides full-text search using Redis 8's FT.SEARCH,
built on top of RedisVL's SearchIndex.

Features:
- BM25 lexical search (keyword matching)
- Tag and category filtering
- Field boosting
- Complementary to vector search
"""

import json
import logging
import uuid
from typing import Any

from redisvl.index import SearchIndex  # type: ignore[import-untyped]
from redisvl.query import FilterQuery  # type: ignore[import-untyped]
from redisvl.query.filter import Tag, Text  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class RedisFullTextSearch:
    """Full-text search using Redis 8's FT.SEARCH.

    Provides BM25-based keyword search, complementing vector search
    for cases where exact keyword matching is preferred.

    Example:
        >>> fts = RedisFullTextSearch(name="articles", redis_url="redis://localhost:6379")
        >>> fts.add_documents([{"title": "Redis Guide", "content": "Learn Redis"}])
        >>> results = fts.search(query="Redis", k=5)

    Args:
        name: Index name in Redis
        redis_url: Redis connection URL
    """

    def __init__(
        self,
        name: str,
        redis_url: str = "redis://localhost:6379",
    ) -> None:
        """Initialize the full-text search index.

        Args:
            name: Index name in Redis
            redis_url: Redis connection URL
        """
        self._name = name
        self._redis_url = redis_url

        # Create index schema for full-text search
        schema = {
            "index": {
                "name": name,
                "prefix": f"fts:{name}:",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "title", "type": "text", "attrs": {"weight": 2.0}},
                {"name": "content", "type": "text"},
                {"name": "category", "type": "tag"},
                {"name": "tags", "type": "tag"},
                {"name": "metadata", "type": "text"},
            ],
        }

        self._index = SearchIndex.from_dict(schema, redis_url=redis_url)

        # Create index if it doesn't exist
        try:
            self._index.create(overwrite=False)
        except Exception as exc:
            logger.debug("Index %s may already exist: %s", name, exc)

    @property
    def name(self) -> str:
        """Index name in Redis."""
        return self._name

    def add_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[str]:
        """Add documents to the search index.

        Args:
            documents: List of documents with title, content, and optional metadata

        Returns:
            List of document IDs
        """
        ids: list[str] = []
        records: list[dict[str, Any]] = []

        for doc in documents:
            doc_id = str(uuid.uuid4().hex[:16])
            ids.append(doc_id)

            # Extract fields
            title = doc.get("title", "")
            content = doc.get("content", "")
            category = doc.get("category", "")
            tags = doc.get("tags", [])

            # Convert tags list to comma-separated string for Redis TAG field
            tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)

            # Store remaining fields as metadata
            metadata = {
                k: v for k, v in doc.items() if k not in ("title", "content", "category", "tags")
            }

            record = {
                "id": doc_id,
                "title": title,
                "content": content,
                "category": category,
                "tags": tags_str,
                "metadata": json.dumps(metadata) if metadata else "{}",
            }
            records.append(record)

        if records:
            self._index.load(records, id_field="id")

        return ids

    def search(
        self,
        query: str,
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search documents using full-text search.

        Args:
            query: Search query (keywords)
            k: Number of results to return
            filter: Optional filters (category, tags)

        Returns:
            List of matching documents with score
        """
        # Build filter expression
        filter_expr = None
        if filter:
            conditions = []
            for field, value in filter.items():
                if field == "category":
                    conditions.append(Tag("category") == value)
                elif field == "tags":
                    conditions.append(Tag("tags") == value)
            if conditions:
                filter_expr = conditions[0]
                for cond in conditions[1:]:
                    filter_expr = filter_expr & cond

        # Create filter query for full-text search
        try:
            if query == "*":
                # Match all - use filter only
                fq = FilterQuery(
                    return_fields=["title", "content", "category", "tags", "metadata"],
                    filter_expression=filter_expr,
                    num_results=k,
                )
            else:
                # Full-text search with optional filter
                fq = FilterQuery(
                    return_fields=["title", "content", "category", "tags", "metadata"],
                    filter_expression=filter_expr,
                    num_results=k,
                )
                # Add text search to the query
                if filter_expr:
                    text_filter = Text("title") % query | Text("content") % query
                    fq = FilterQuery(
                        return_fields=["title", "content", "category", "tags", "metadata"],
                        filter_expression=filter_expr & text_filter,
                        num_results=k,
                    )
                else:
                    text_filter = Text("title") % query | Text("content") % query
                    fq = FilterQuery(
                        return_fields=["title", "content", "category", "tags", "metadata"],
                        filter_expression=text_filter,
                        num_results=k,
                    )

            results = self._index.query(fq)
        except Exception as exc:
            logger.debug("Full-text search query failed for index %s: %s", self._name, exc)
            return []

        # Format results
        formatted: list[dict[str, Any]] = []
        for result in results:
            # Parse tags back to list
            tags_str = result.get("tags", "")
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]

            # Parse metadata
            metadata_str = result.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            formatted.append(
                {
                    "id": result.get("id", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "category": result.get("category", ""),
                    "tags": tags,
                    "score": float(result.get("score", 0.0)),
                    **metadata,
                }
            )

        return formatted

    def count(self) -> int:
        """Count documents in the index.

        Returns:
            Number of documents
        """
        try:
            info = self._index.info()
            return int(info.get("num_docs", 0))
        except Exception as exc:
            logger.debug("Failed to get document count for index %s: %s", self._name, exc)
            return 0

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        from redis import Redis

        client = Redis.from_url(self._redis_url)
        try:
            for doc_id in ids:
                key = f"fts:{self._name}:{doc_id}"
                client.delete(key)
        finally:
            client.close()

    def delete_all(self) -> None:
        """Delete all documents and recreate the index."""
        try:
            self._index.delete(drop=True)
        except Exception as exc:
            logger.debug("Failed to drop index %s (may not exist): %s", self._name, exc)

        # Recreate empty index
        schema = {
            "index": {
                "name": self._name,
                "prefix": f"fts:{self._name}:",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "title", "type": "text", "attrs": {"weight": 2.0}},
                {"name": "content", "type": "text"},
                {"name": "category", "type": "tag"},
                {"name": "tags", "type": "tag"},
                {"name": "metadata", "type": "text"},
            ],
        }
        self._index = SearchIndex.from_dict(schema, redis_url=self._redis_url)
        try:
            self._index.create(overwrite=True)
        except Exception as exc:
            logger.error("Failed to recreate index %s after delete_all: %s", self._name, exc)
