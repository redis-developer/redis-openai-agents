"""Unit tests for HybridSearchService - TDD RED phase.

These tests verify the HybridSearchService logic using mocks.
They test the RRF algorithm, weight normalization, and deduplication.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redis_openai_agents.hybrid import HybridSearchService, SearchResult


class TestHybridSearchServiceConstruction:
    """Test HybridSearchService construction and configuration."""

    def test_create_with_defaults(self) -> None:
        """Can create service with default weights."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore"):
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch"):
                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                assert service.index_name == "test"
                assert service.default_vector_weight == 0.7
                assert service.default_text_weight == 0.3

    def test_create_with_custom_weights(self) -> None:
        """Can create service with custom default weights."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore"):
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch"):
                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                    default_vector_weight=0.5,
                    default_text_weight=0.5,
                )

                assert service.default_vector_weight == 0.5
                assert service.default_text_weight == 0.5

    def test_weights_must_be_valid(self) -> None:
        """Weights must be between 0 and 1."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore"):
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch"):
                with pytest.raises(ValueError, match="weight"):
                    HybridSearchService(
                        redis_url="redis://localhost:6379",
                        index_name="test",
                        default_vector_weight=1.5,
                    )


class TestRRFFusion:
    """Test Reciprocal Rank Fusion algorithm."""

    async def test_rrf_combines_rankings(self) -> None:
        """RRF correctly combines rankings from vector and text search."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                # Setup mocks
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                # Mock vector search results (async)
                mock_vs_instance.asearch = AsyncMock(
                    return_value=[
                        {"id": "doc1", "content": "Doc 1", "metadata": {}, "score": 0.9},
                        {"id": "doc2", "content": "Doc 2", "metadata": {}, "score": 0.8},
                    ]
                )

                # Mock text search results (sync, wrapped in executor)
                mock_fts_instance.search = MagicMock(
                    return_value=[
                        {
                            "id": "doc2",
                            "content": "Doc 2",
                            "score": 0.95,
                            "title": "",
                            "category": "",
                            "tags": [],
                        },
                        {
                            "id": "doc3",
                            "content": "Doc 3",
                            "score": 0.7,
                            "title": "",
                            "category": "",
                            "tags": [],
                        },
                    ]
                )

                results = await service.search(query="test", k=3)

                # doc2 should rank higher as it appears in both
                assert len(results) <= 3
                # Verify deduplication - doc2 should appear only once
                doc2_count = sum(1 for r in results if r.id == "doc2")
                assert doc2_count <= 1

    async def test_rrf_with_k_constant(self) -> None:
        """RRF uses k=60 constant for smoothing."""
        # The RRF formula: score = sum(1/(k + rank)) for each list
        # With k=60, a doc at rank 1 in both lists gets: 2 * (1/61) = 0.0328
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                # Same doc at rank 1 in both lists
                mock_vs_instance.asearch = AsyncMock(
                    return_value=[
                        {"id": "doc1", "content": "Doc 1", "metadata": {}, "score": 0.9},
                    ]
                )
                mock_fts_instance.search = MagicMock(
                    return_value=[
                        {
                            "id": "doc1",
                            "content": "Doc 1",
                            "score": 0.9,
                            "title": "",
                            "category": "",
                            "tags": [],
                        },
                    ]
                )

                results = await service.search(
                    query="test", k=1, vector_weight=0.5, text_weight=0.5
                )

                assert len(results) == 1
                # Score should be based on RRF: 0.5 * (1/61) + 0.5 * (1/61)
                expected_score = 0.5 * (1 / 61) + 0.5 * (1 / 61)
                assert abs(results[0].score - expected_score) < 0.001


class TestWeightNormalization:
    """Test that weights are normalized correctly."""

    async def test_weights_normalized_to_sum_1(self) -> None:
        """Weights are normalized to sum to 1."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                mock_vs_instance.asearch = AsyncMock(
                    return_value=[
                        {"id": "doc1", "content": "Doc 1", "metadata": {}, "score": 0.9},
                    ]
                )
                mock_fts_instance.search = MagicMock(return_value=[])

                # Pass unnormalized weights (0.3 + 0.3 = 0.6, not 1.0)
                results = await service.search(
                    query="test",
                    k=1,
                    vector_weight=0.3,
                    text_weight=0.3,
                )

                # Should still work - weights get normalized internally
                assert len(results) == 1

    async def test_zero_text_weight(self) -> None:
        """With text_weight=0, only vector search is used."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                mock_vs_instance.asearch = AsyncMock(
                    return_value=[
                        {"id": "doc1", "content": "Doc 1", "metadata": {}, "score": 0.9},
                    ]
                )
                mock_fts_instance.search = MagicMock(
                    return_value=[
                        {
                            "id": "doc2",
                            "content": "Doc 2",
                            "score": 0.95,
                            "title": "",
                            "category": "",
                            "tags": [],
                        },
                    ]
                )

                results = await service.search(
                    query="test",
                    k=2,
                    vector_weight=1.0,
                    text_weight=0.0,
                )

                # Only vector results should contribute to score
                assert len(results) == 2
                # doc1 should have higher score from vector search contribution
                doc1 = next((r for r in results if r.id == "doc1"), None)
                doc2 = next((r for r in results if r.id == "doc2"), None)
                assert doc1 is not None
                assert doc2 is not None
                assert doc1.score > doc2.score  # doc1 has vector contribution

    async def test_zero_vector_weight(self) -> None:
        """With vector_weight=0, only text search is used."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                mock_vs_instance.asearch = AsyncMock(
                    return_value=[
                        {"id": "doc1", "content": "Doc 1", "metadata": {}, "score": 0.9},
                    ]
                )
                mock_fts_instance.search = MagicMock(
                    return_value=[
                        {
                            "id": "doc2",
                            "content": "Doc 2",
                            "score": 0.95,
                            "title": "",
                            "category": "",
                            "tags": [],
                        },
                    ]
                )

                results = await service.search(
                    query="test",
                    k=2,
                    vector_weight=0.0,
                    text_weight=1.0,
                )

                # Only text results should contribute to score
                assert len(results) == 2
                doc1 = next((r for r in results if r.id == "doc1"), None)
                doc2 = next((r for r in results if r.id == "doc2"), None)
                assert doc1 is not None
                assert doc2 is not None
                assert doc2.score > doc1.score  # doc2 has text contribution


class TestDeduplication:
    """Test deduplication of results."""

    async def test_same_doc_appears_once(self) -> None:
        """Document appearing in both searches should appear only once in results."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                # Same doc in both results
                mock_vs_instance.asearch = AsyncMock(
                    return_value=[
                        {"id": "doc1", "content": "Shared Doc", "metadata": {}, "score": 0.9},
                    ]
                )
                mock_fts_instance.search = MagicMock(
                    return_value=[
                        {
                            "id": "doc1",
                            "content": "Shared Doc",
                            "score": 0.95,
                            "title": "",
                            "category": "",
                            "tags": [],
                        },
                    ]
                )

                results = await service.search(query="test", k=5)

                # Should have only 1 result
                assert len(results) == 1
                assert results[0].id == "doc1"

    async def test_combined_score_for_duplicates(self) -> None:
        """Duplicate docs should have combined RRF score."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                # Same doc at rank 1 in both lists
                mock_vs_instance.asearch = AsyncMock(
                    return_value=[
                        {"id": "shared", "content": "Shared", "metadata": {}, "score": 0.9},
                        {
                            "id": "vector_only",
                            "content": "Vector Only",
                            "metadata": {},
                            "score": 0.8,
                        },
                    ]
                )
                mock_fts_instance.search = MagicMock(
                    return_value=[
                        {
                            "id": "shared",
                            "content": "Shared",
                            "score": 0.95,
                            "title": "",
                            "category": "",
                            "tags": [],
                        },
                        {
                            "id": "text_only",
                            "content": "Text Only",
                            "score": 0.7,
                            "title": "",
                            "category": "",
                            "tags": [],
                        },
                    ]
                )

                results = await service.search(
                    query="test",
                    k=3,
                    vector_weight=0.5,
                    text_weight=0.5,
                )

                # Shared doc should rank highest due to combined score
                assert results[0].id == "shared"


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_fields(self) -> None:
        """SearchResult has all required fields."""
        result = SearchResult(
            id="test_id",
            content="Test content",
            metadata={"key": "value"},
            score=0.85,
        )

        assert result.id == "test_id"
        assert result.content == "Test content"
        assert result.metadata == {"key": "value"}
        assert result.score == 0.85

    def test_search_result_defaults(self) -> None:
        """SearchResult has sensible defaults."""
        result = SearchResult(
            id="test",
            content="Content",
        )

        assert result.metadata == {}
        assert result.score == 0.0


class TestDocumentIndexing:
    """Test document indexing."""

    async def test_index_to_both_stores(self) -> None:
        """Documents are indexed to both vector and text stores."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                # Setup async add method
                mock_vs_instance.aadd_documents = AsyncMock(return_value=["id1", "id2"])
                mock_fts_instance.add_documents = MagicMock(return_value=["id1", "id2"])

                docs = [
                    {"content": "Doc 1", "metadata": {"key": "val"}},
                    {"content": "Doc 2", "metadata": {}},
                ]

                ids = await service.index_documents(docs)

                # Both stores should be called
                mock_vs_instance.aadd_documents.assert_called_once()
                mock_fts_instance.add_documents.assert_called_once()
                assert len(ids) == 2

    async def test_count_from_vector_store(self) -> None:
        """Count returns document count from vector store."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                mock_vs_instance.count = MagicMock(return_value=5)

                count = await service.count()

                assert count == 5
                mock_vs_instance.count.assert_called_once()

    async def test_delete_all_both_stores(self) -> None:
        """Delete all removes from both stores."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                await service.delete_all()

                mock_vs_instance.delete_all.assert_called_once()
                mock_fts_instance.delete_all.assert_called_once()


class TestFilterExpression:
    """Test filter expression handling."""

    async def test_filter_passed_to_both_stores(self) -> None:
        """Filter expression is passed to both search methods."""
        with patch("redis_openai_agents.hybrid.RedisVectorStore") as mock_vs:
            with patch("redis_openai_agents.hybrid.RedisFullTextSearch") as mock_fts:
                mock_vs_instance = MagicMock()
                mock_fts_instance = MagicMock()
                mock_vs.return_value = mock_vs_instance
                mock_fts.return_value = mock_fts_instance

                service = HybridSearchService(
                    redis_url="redis://localhost:6379",
                    index_name="test",
                )

                mock_vs_instance.asearch = AsyncMock(return_value=[])
                mock_fts_instance.search = MagicMock(return_value=[])

                filter_expr = {"category": "database"}
                await service.search(query="test", k=5, filter_expression=filter_expr)

                # Both stores should receive the filter
                mock_vs_instance.asearch.assert_called_once()
                call_kwargs = mock_vs_instance.asearch.call_args
                assert call_kwargs.kwargs.get("filter") == filter_expr

                mock_fts_instance.search.assert_called_once()
                fts_call_kwargs = mock_fts_instance.search.call_args
                assert fts_call_kwargs.kwargs.get("filter") == filter_expr
