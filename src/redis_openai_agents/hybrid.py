"""HybridSearchService - Combined BM25 + Vector search.

This module provides hybrid search combining lexical (BM25) and semantic (vector)
search using Reciprocal Rank Fusion (RRF) for score combination.

Features:
- Combined BM25 + Vector similarity search
- Configurable weights for text vs vector
- Automatic result deduplication
- Metadata filtering
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any

from .search import RedisFullTextSearch
from .vector import RedisVectorStore


@dataclass
class SearchResult:
    """A search result from hybrid search.

    Attributes:
        id: Document ID
        content: Document content
        metadata: Document metadata dictionary
        score: Combined RRF score (0-1 range)
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class HybridSearchService:
    """Combined BM25 + Vector search using RedisVL.

    Performs both lexical (BM25) and semantic (vector) search,
    then combines results using Reciprocal Rank Fusion (RRF).

    Example:
        >>> service = HybridSearchService(
        ...     redis_url="redis://localhost:6379",
        ...     index_name="documents",
        ... )
        >>> await service.index_documents([
        ...     {"content": "Redis is a fast database.", "metadata": {"source": "docs"}},
        ... ])
        >>> results = await service.search("fast database", k=5)

    Args:
        redis_url: Redis connection URL
        index_name: Name for the search indices
        default_vector_weight: Default weight for vector similarity (0-1)
        default_text_weight: Default weight for BM25 text search (0-1)
    """

    # RRF constant for smoothing rankings
    _RRF_K = 60

    def __init__(
        self,
        redis_url: str,
        index_name: str,
        default_vector_weight: float = 0.7,
        default_text_weight: float = 0.3,
    ) -> None:
        """Initialize the hybrid search service.

        Args:
            redis_url: Redis connection URL
            index_name: Name for the search indices
            default_vector_weight: Default weight for vector similarity (0-1)
            default_text_weight: Default weight for BM25 text search (0-1)

        Raises:
            ValueError: If weights are not in valid range
        """
        # Validate weights
        if not 0.0 <= default_vector_weight <= 1.0:
            raise ValueError(
                f"default_vector_weight must be between 0 and 1, got {default_vector_weight}"
            )
        if not 0.0 <= default_text_weight <= 1.0:
            raise ValueError(
                f"default_text_weight must be between 0 and 1, got {default_text_weight}"
            )

        self._redis_url = redis_url
        self._index_name = index_name
        self._default_vector_weight = default_vector_weight
        self._default_text_weight = default_text_weight

        # Initialize underlying stores
        self._vector_store = RedisVectorStore(
            name=f"{index_name}_vec",
            redis_url=redis_url,
        )
        self._text_search = RedisFullTextSearch(
            name=f"{index_name}_txt",
            redis_url=redis_url,
        )

    @property
    def index_name(self) -> str:
        """The index name for this service."""
        return self._index_name

    @property
    def default_vector_weight(self) -> float:
        """Default weight for vector similarity."""
        return self._default_vector_weight

    @property
    def default_text_weight(self) -> float:
        """Default weight for BM25 text search."""
        return self._default_text_weight

    async def index_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[str]:
        """Index documents for hybrid search.

        Documents are indexed in both vector and text stores.

        Args:
            documents: List of documents with 'content' and optional 'metadata'

        Returns:
            List of document IDs
        """
        # Index in vector store (async)
        ids = await self._vector_store.aadd_documents(documents)

        # Index in text store (sync, run in executor)
        # Transform docs to match text search schema
        text_docs = []
        for doc in documents:
            text_doc = {
                "title": "",  # Optional, empty for now
                "content": doc.get("content", ""),
                "category": doc.get("metadata", {}).get("category", ""),
                "tags": doc.get("metadata", {}).get("tags", []),
                **doc.get("metadata", {}),
            }
            text_docs.append(text_doc)

        await asyncio.to_thread(self._text_search.add_documents, text_docs)

        return ids

    async def count(self) -> int:
        """Count indexed documents.

        Returns:
            Number of documents in the index
        """
        return self._vector_store.count()

    async def search(
        self,
        query: str,
        k: int = 10,
        vector_weight: float | None = None,
        text_weight: float | None = None,
        filter_expression: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search combining semantic and lexical matching.

        Uses Reciprocal Rank Fusion (RRF) to combine results from
        vector similarity search and BM25 text search.

        Args:
            query: Search query text
            k: Number of results to return
            vector_weight: Weight for vector similarity (0-1), uses default if None
            text_weight: Weight for BM25 text search (0-1), uses default if None
            filter_expression: Optional metadata filter dict

        Returns:
            Combined and re-ranked results as SearchResult objects
        """
        # Use defaults if not specified
        if vector_weight is None:
            vector_weight = self._default_vector_weight
        if text_weight is None:
            text_weight = self._default_text_weight

        # Normalize weights to sum to 1
        total_weight = vector_weight + text_weight
        if total_weight > 0:
            vector_weight = vector_weight / total_weight
            text_weight = text_weight / total_weight
        else:
            # Both zero - equal weights
            vector_weight = 0.5
            text_weight = 0.5

        # Fetch more results for better fusion
        fetch_k = k * 2

        # Run vector search (async)
        vector_results = await self._vector_store.asearch(
            query=query,
            k=fetch_k,
            filter=filter_expression,
        )

        # Run text search (sync, run in thread)
        text_results = await asyncio.to_thread(
            self._text_search.search,
            query=query,
            k=fetch_k,
            filter=filter_expression,
        )

        # Combine using RRF
        return self._rrf_fusion(
            vector_results=vector_results,
            text_results=text_results,
            k=k,
            vector_weight=vector_weight,
            text_weight=text_weight,
        )

    def _rrf_fusion(
        self,
        vector_results: list[dict[str, Any]],
        text_results: list[dict[str, Any]],
        k: int,
        vector_weight: float,
        text_weight: float,
    ) -> list[SearchResult]:
        """Combine results using Reciprocal Rank Fusion.

        RRF score = sum of weight * (1 / (k + rank)) for each list

        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            k: Number of results to return
            vector_weight: Weight for vector results
            text_weight: Weight for text results

        Returns:
            Combined and deduplicated results
        """
        # Build score and document maps
        # Use content hash for deduplication since vector/text stores have different IDs
        combined_scores: dict[str, float] = {}
        combined_docs: dict[str, dict[str, Any]] = {}
        content_to_id: dict[str, str] = {}  # Map content hash to original ID

        # Process vector results
        for rank, doc in enumerate(vector_results):
            content = doc.get("content", "")
            content_key = self._content_hash(content)
            original_id = doc.get("id", content_key)

            rrf_score = vector_weight * (1.0 / (self._RRF_K + rank + 1))
            combined_scores[content_key] = combined_scores.get(content_key, 0.0) + rrf_score

            if content_key not in combined_docs:
                combined_docs[content_key] = doc
                content_to_id[content_key] = original_id

        # Process text results
        for rank, doc in enumerate(text_results):
            content = doc.get("content", "")
            content_key = self._content_hash(content)
            original_id = doc.get("id", content_key)

            rrf_score = text_weight * (1.0 / (self._RRF_K + rank + 1))
            combined_scores[content_key] = combined_scores.get(content_key, 0.0) + rrf_score

            if content_key not in combined_docs:
                combined_docs[content_key] = doc
                content_to_id[content_key] = original_id

        # Sort by combined score
        sorted_keys = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )

        # Build results
        results: list[SearchResult] = []
        for content_key in sorted_keys[:k]:
            doc = combined_docs[content_key]
            results.append(
                SearchResult(
                    id=content_to_id[content_key],
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=combined_scores[content_key],
                )
            )

        return results

    def _content_hash(self, content: str) -> str:
        """Generate a hash for content as fallback ID.

        Args:
            content: Document content

        Returns:
            Hash string for the content
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def delete_all(self) -> None:
        """Delete all documents from both stores."""
        self._vector_store.delete_all()
        self._text_search.delete_all()
