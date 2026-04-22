"""RedisVectorStore - Vector storage and search for RAG.

This module provides vector storage and similarity search using Redis,
built on top of RedisVL's SearchIndex.

Features:
- Document storage with automatic embedding generation
- Semantic similarity search
- Metadata filtering
- HNSW algorithm for fast approximate nearest neighbor search
"""

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
from redisvl.index import SearchIndex  # type: ignore[import-untyped]
from redisvl.query import VectorQuery  # type: ignore[import-untyped]
from redisvl.utils.vectorize import HFTextVectorizer  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redisvl.extensions.cache.embeddings import (  # type: ignore[import-untyped]
        EmbeddingsCache,
    )

DEFAULT_VECTORIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RedisVectorStore:
    """Vector store for document storage and semantic search.

    Uses Redis with RedisVL for high-performance vector similarity search.

    Example:
        >>> store = RedisVectorStore(name="docs", redis_url="redis://localhost:6379")
        >>> store.add_documents([{"content": "Hello world", "metadata": {"source": "test"}}])
        >>> results = store.search(query="greeting", k=5)

    Args:
        name: Index name in Redis
        redis_url: Redis connection URL
        vector_dims: Dimension of embedding vectors (default 384 for all-MiniLM-L6-v2)
        distance_metric: Distance metric (COSINE, L2, IP)
    """

    def __init__(
        self,
        name: str,
        redis_url: str = "redis://localhost:6379",
        vector_dims: int = 384,
        distance_metric: str = "COSINE",
        embeddings_cache: "EmbeddingsCache | None" = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            name: Index name in Redis
            redis_url: Redis connection URL
            vector_dims: Dimension of embedding vectors (384 for all-MiniLM-L6-v2)
            distance_metric: Distance metric (COSINE, L2, IP)
            embeddings_cache: Optional RedisVL EmbeddingsCache. When provided,
                repeated embeddings of identical content are served from the
                cache rather than re-invoking the vectorizer.
        """
        self._name = name
        self._redis_url = redis_url
        self._vector_dims = vector_dims
        self._distance_metric = distance_metric
        self._vectorizer_model = DEFAULT_VECTORIZER_MODEL

        # Initialize vectorizer for embedding generation
        # Use a general-purpose embedding model for document retrieval
        # (redis/langcache-embed-v1 is optimized for caching, not retrieval)
        self._vectorizer = HFTextVectorizer(
            model=self._vectorizer_model,
            cache=embeddings_cache,
        )

        # Create index schema
        schema = {
            "index": {
                "name": name,
                "prefix": f"doc:{name}:",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "metadata", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": vector_dims,
                        "distance_metric": distance_metric,
                        "algorithm": "HNSW",
                        "datatype": "FLOAT32",
                    },
                },
            ],
        }

        self._index = SearchIndex.from_dict(schema, redis_url=redis_url)

        # Create index if it doesn't exist
        try:
            self._index.create(overwrite=False)
        except Exception as exc:
            # Index might already exist
            logger.debug("Index '%s' creation skipped (may already exist): %s", name, exc)

    @property
    def name(self) -> str:
        """Index name in Redis."""
        return self._name

    @property
    def vectorizer_model(self) -> str:
        """Name of the underlying embedding model."""
        return self._vectorizer_model

    def add_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[str]:
        """Add documents to the vector store.

        Documents are automatically embedded using the vectorizer.

        Args:
            documents: List of documents with 'content' and optional 'metadata'

        Returns:
            List of document IDs
        """
        ids: list[str] = []
        records: list[dict[str, Any]] = []

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Generate ID
            doc_id = str(uuid.uuid4().hex[:16])
            ids.append(doc_id)

            # Generate embedding and convert to bytes for Redis storage
            embedding = self._vectorizer.embed(content)
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

            # Create record
            record = {
                "id": doc_id,
                "content": content,
                "metadata": json.dumps(metadata) if metadata else "{}",
                "embedding": embedding_bytes,
            }
            records.append(record)

        # Load records into Redis
        if records:
            self._index.load(records, id_field="id")

        return ids

    def search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of matching documents with content, metadata, and score
        """
        # Generate query embedding
        query_embedding = self._vectorizer.embed(query)

        # Build filter expression if provided
        filter_expr = None
        if filter:
            # For now, simple tag-based filtering
            # Format: @field:{value}
            conditions = []
            for _field, value in filter.items():
                conditions.append(f"@metadata:*{value}*")
            if conditions:
                filter_expr = " ".join(conditions)

        # Create vector query
        vector_query = VectorQuery(
            vector=query_embedding,
            vector_field_name="embedding",
            return_fields=["content", "metadata"],
            num_results=k,
            filter_expression=filter_expr,
        )

        # Execute search
        try:
            results = self._index.query(vector_query)
        except Exception as exc:
            logger.debug("Vector search query failed for index '%s': %s", self._name, exc)
            return []

        # Format results
        formatted_results: list[dict[str, Any]] = []
        for result in results:
            # Parse metadata from JSON string
            metadata_str = result.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            # Calculate similarity from distance
            distance = float(result.get("vector_distance", 0.0))
            score = 1.0 - distance  # Convert distance to similarity

            formatted_results.append(
                {
                    "content": result.get("content", ""),
                    "metadata": metadata,
                    "score": score,
                    "id": result.get("id", ""),
                }
            )

        return formatted_results

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector similarity and BM25 text search.

        This method runs both semantic (vector) and lexical (BM25) searches,
        then combines the results using weighted scoring.

        Args:
            query: Search query text
            k: Number of results to return
            text_weight: Weight for BM25 text search scores (0.0 to 1.0)
            vector_weight: Weight for vector similarity scores (0.0 to 1.0)
            filter: Optional metadata filter

        Returns:
            List of matching documents with combined scores
        """
        from redisvl.query import FilterQuery
        from redisvl.query.filter import Text  # type: ignore[import-untyped]

        # Normalize weights
        total_weight = text_weight + vector_weight
        if total_weight > 0:
            text_weight = text_weight / total_weight
            vector_weight = vector_weight / total_weight

        # Run vector search
        vector_results = self.search(query=query, k=k * 2, filter=filter)

        # Run text search using FT.SEARCH with text query
        text_results: list[dict[str, Any]] = []
        try:
            # Build text filter
            text_filter = Text("content") % query

            # Build metadata filter if provided
            filter_expr = text_filter
            if filter:
                for _field, value in filter.items():
                    filter_expr = filter_expr & (Text("metadata") % f"*{value}*")

            fq = FilterQuery(
                return_fields=["content", "metadata"],
                filter_expression=filter_expr,
                num_results=k * 2,
            )

            results = self._index.query(fq)
            for result in results:
                metadata_str = result.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                text_results.append(
                    {
                        "content": result.get("content", ""),
                        "metadata": metadata,
                        "score": 1.0,  # Text matches get score 1.0
                        "id": result.get("id", ""),
                    }
                )
        except Exception as exc:
            # Text search failed, continue with vector results only
            logger.debug(
                "Text search failed for index '%s', using vector results only: %s", self._name, exc
            )

        # Combine results using Reciprocal Rank Fusion (RRF)
        # Score = sum of 1/(rank + k) for each result list
        rrf_k = 60  # RRF constant

        # Build score maps
        combined_scores: dict[str, float] = {}
        combined_docs: dict[str, dict[str, Any]] = {}

        # Add vector results
        for rank, doc in enumerate(vector_results):
            doc_id = doc.get("id") or doc.get("content", "")[:50]
            rrf_score = vector_weight * (1.0 / (rank + rrf_k))
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score
            combined_docs[doc_id] = doc

        # Add text results
        for rank, doc in enumerate(text_results):
            doc_id = doc.get("id") or doc.get("content", "")[:50]
            rrf_score = text_weight * (1.0 / (rank + rrf_k))
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score
            if doc_id not in combined_docs:
                combined_docs[doc_id] = doc

        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Return top k results
        final_results: list[dict[str, Any]] = []
        for doc_id in sorted_ids[:k]:
            doc = combined_docs[doc_id]
            doc["score"] = combined_scores[doc_id]
            final_results.append(doc)

        return final_results

    def count(self) -> int:
        """Count documents in the store.

        Returns:
            Number of documents
        """
        try:
            info = self._index.info()
            return int(info.get("num_docs", 0))
        except Exception as exc:
            logger.debug("Failed to get document count for index '%s': %s", self._name, exc)
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
                key = f"doc:{self._name}:{doc_id}"
                client.delete(key)
        finally:
            client.close()

    def delete_all(self) -> None:
        """Delete all documents and the index."""
        try:
            self._index.delete(drop=True)
        except Exception as exc:
            logger.debug("Failed to drop index '%s' during delete_all: %s", self._name, exc)

        # Recreate the empty index
        schema = {
            "index": {
                "name": self._name,
                "prefix": f"doc:{self._name}:",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "metadata", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": self._vector_dims,
                        "distance_metric": self._distance_metric,
                        "algorithm": "HNSW",
                        "datatype": "FLOAT32",
                    },
                },
            ],
        }
        self._index = SearchIndex.from_dict(schema, redis_url=self._redis_url)
        try:
            self._index.create(overwrite=True)
        except Exception as exc:
            logger.error(
                "Failed to recreate index '%s' after drop; store is in a broken state: %s",
                self._name,
                exc,
            )

    # Async methods

    async def aadd_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[str]:
        """Async version of add_documents().

        Args:
            documents: List of documents with 'content' and optional 'metadata'

        Returns:
            List of document IDs
        """
        return await asyncio.to_thread(self.add_documents, documents)

    async def asearch(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Async version of search().

        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of matching documents
        """
        return await asyncio.to_thread(self.search, query=query, k=k, filter=filter)

    async def ahybrid_search(
        self,
        query: str,
        k: int = 10,
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Async version of hybrid_search().

        Args:
            query: Search query text
            k: Number of results to return
            text_weight: Weight for BM25 text search scores
            vector_weight: Weight for vector similarity scores
            filter: Optional metadata filter

        Returns:
            List of matching documents with combined scores
        """
        return await asyncio.to_thread(
            self.hybrid_search,
            query=query,
            k=k,
            text_weight=text_weight,
            vector_weight=vector_weight,
            filter=filter,
        )
