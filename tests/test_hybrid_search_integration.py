"""Integration tests for HybridSearchService - TDD RED phase.

These tests define the expected behavior for combining BM25 + Vector search.
They test against a real Redis instance.
"""

from redis_openai_agents import HybridSearchService


class TestHybridSearchServiceBasics:
    """Test basic HybridSearchService operations."""

    async def test_create_service(self, redis_url: str) -> None:
        """HybridSearchService can be instantiated with redis_url."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_basics",
        )
        assert service is not None
        assert service.index_name == "test_hybrid_basics"

    async def test_index_documents(self, redis_url: str) -> None:
        """Can index documents for hybrid search."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_index",
        )

        docs = [
            {"content": "Redis is an in-memory database.", "metadata": {"source": "docs"}},
            {"content": "Python is a programming language.", "metadata": {"source": "wiki"}},
        ]
        ids = await service.index_documents(docs)

        assert len(ids) == 2
        assert all(id is not None for id in ids)

    async def test_document_count(self, redis_url: str) -> None:
        """Can count indexed documents."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_count",
        )

        assert await service.count() == 0

        docs = [
            {"content": "Document one"},
            {"content": "Document two"},
        ]
        await service.index_documents(docs)

        assert await service.count() == 2


class TestHybridSearchCombinesResults:
    """Test that hybrid search combines vector and text results."""

    async def test_hybrid_search_returns_results(self, redis_url: str) -> None:
        """Hybrid search returns combined results from both methods."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_combined",
        )

        docs = [
            {"content": "Redis is a fast in-memory data structure store."},
            {"content": "Python programming language basics and tutorials."},
            {"content": "Database performance optimization techniques."},
        ]
        await service.index_documents(docs)

        results = await service.search(query="Redis database", k=2)

        assert len(results) <= 2
        assert results[0].content is not None
        assert results[0].score is not None

    async def test_hybrid_search_finds_semantic_matches(self, redis_url: str) -> None:
        """Hybrid search finds semantically similar documents."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_semantic",
        )

        docs = [
            {"content": "Redis is a fast in-memory data store."},
            {"content": "Cooking pasta requires boiling water."},
            {"content": "Database systems store information."},
        ]
        await service.index_documents(docs)

        # Search for semantic match (no exact keyword match for "data storage")
        results = await service.search(query="data storage solutions", k=2)

        # Should find Redis and Database docs due to semantic similarity
        contents = [r.content for r in results]
        assert any("Redis" in c or "Database" in c for c in contents)

    async def test_hybrid_search_finds_keyword_matches(self, redis_url: str) -> None:
        """Hybrid search finds exact keyword matches via BM25."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_keyword",
        )

        docs = [
            {"content": "The quick brown fox jumps over the lazy dog."},
            {"content": "Redis is blazingly fast for caching."},
            {"content": "Database indexing improves query speed."},
        ]
        await service.index_documents(docs)

        # Search for exact keyword "blazingly"
        results = await service.search(query="blazingly fast", k=3)

        # Should find the Redis document due to keyword match
        assert any("blazingly" in r.content for r in results)


class TestHybridSearchWeights:
    """Test that weights affect search ranking."""

    async def test_vector_weight_affects_ranking(self, redis_url: str) -> None:
        """Higher vector weight should favor semantic matches."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_vector_weight",
        )

        docs = [
            {"content": "Redis commands for data manipulation."},  # Has "data"
            {"content": "Information storage and retrieval systems."},  # Semantic match for "data"
            {"content": "Cooking recipes and kitchen tips."},  # Unrelated
        ]
        await service.index_documents(docs)

        # High vector weight - should favor semantic similarity
        results = await service.search(
            query="data management",
            k=3,
            vector_weight=0.9,
            text_weight=0.1,
        )

        # Both data-related docs should rank high
        assert len(results) >= 2

    async def test_text_weight_affects_ranking(self, redis_url: str) -> None:
        """Higher text weight should favor exact keyword matches."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_text_weight",
        )

        docs = [
            {"content": "Redis commands and operations."},
            {"content": "Memory management in systems."},
        ]
        await service.index_documents(docs)

        # High text weight - should favor exact keyword matches
        results = await service.search(
            query="Redis",
            k=2,
            text_weight=0.9,
            vector_weight=0.1,
        )

        assert len(results) >= 1
        # First result should contain "Redis" due to high text weight
        assert "Redis" in results[0].content

    async def test_equal_weights(self, redis_url: str) -> None:
        """Equal weights should give balanced results."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_equal",
        )

        docs = [
            {"content": "Redis caching strategies."},
            {"content": "Python web frameworks."},
        ]
        await service.index_documents(docs)

        results = await service.search(
            query="Redis",
            k=2,
            vector_weight=0.5,
            text_weight=0.5,
        )

        assert len(results) >= 1


class TestHybridSearchFiltering:
    """Test metadata filtering in hybrid search."""

    async def test_filter_expression_applied(self, redis_url: str) -> None:
        """Metadata filters should work with hybrid search."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_filter",
        )

        docs = [
            {"content": "Redis caching strategies.", "metadata": {"category": "database"}},
            {"content": "Python web frameworks.", "metadata": {"category": "programming"}},
            {"content": "Redis cluster setup.", "metadata": {"category": "database"}},
        ]
        await service.index_documents(docs)

        results = await service.search(
            query="Redis",
            k=10,
            filter_expression={"category": "database"},
        )

        assert len(results) == 2
        for r in results:
            assert r.metadata.get("category") == "database"

    async def test_multiple_filter_conditions(self, redis_url: str) -> None:
        """Can apply multiple filter conditions."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_multi_filter",
        )

        docs = [
            {"content": "Redis intro.", "metadata": {"category": "database", "level": "beginner"}},
            {
                "content": "Redis advanced.",
                "metadata": {"category": "database", "level": "advanced"},
            },
            {
                "content": "Python intro.",
                "metadata": {"category": "programming", "level": "beginner"},
            },
        ]
        await service.index_documents(docs)

        results = await service.search(
            query="intro",
            k=10,
            filter_expression={"category": "database", "level": "beginner"},
        )

        assert len(results) == 1
        assert "Redis intro" in results[0].content


class TestHybridSearchDeduplication:
    """Test deduplication of results from both search methods."""

    async def test_deduplication_of_results(self, redis_url: str) -> None:
        """Same doc from both searches should appear once."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_dedup",
        )

        # This doc should match both vector and text search for "Redis database"
        docs = [
            {"content": "Redis is a fast in-memory database."},
            {"content": "Python programming basics."},
        ]
        await service.index_documents(docs)

        results = await service.search(query="Redis database", k=10)

        # Count occurrences of Redis document
        redis_count = sum(1 for r in results if "Redis" in r.content)
        assert redis_count == 1  # Should appear only once

    async def test_unique_ids_in_results(self, redis_url: str) -> None:
        """Result IDs should be unique."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_unique_ids",
        )

        docs = [
            {"content": "Redis caching."},
            {"content": "Redis clustering."},
            {"content": "Redis persistence."},
        ]
        await service.index_documents(docs)

        results = await service.search(query="Redis", k=3)

        # All IDs should be unique
        ids = [r.id for r in results]
        assert len(ids) == len(set(ids))


class TestHybridSearchEdgeCases:
    """Test edge cases and error handling."""

    async def test_search_empty_index(self, redis_url: str) -> None:
        """Search on empty index returns empty list."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_empty",
        )

        results = await service.search(query="anything", k=5)

        assert results == []

    async def test_search_no_matches(self, redis_url: str) -> None:
        """Search with no matches returns empty list."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_no_match",
        )

        docs = [
            {"content": "Redis database."},
            {"content": "Python programming."},
        ]
        await service.index_documents(docs)

        # Search for completely unrelated term
        results = await service.search(query="quantum physics equations", k=5)

        # Should return empty or very low-scoring results
        # Vector search might still return something, so check scores
        assert all(r.score < 0.5 for r in results)

    async def test_search_k_greater_than_docs(self, redis_url: str) -> None:
        """Search with k > document count returns all docs."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_k_large",
        )

        docs = [
            {"content": "Doc one."},
            {"content": "Doc two."},
        ]
        await service.index_documents(docs)

        results = await service.search(query="doc", k=100)

        assert len(results) == 2

    async def test_delete_all_documents(self, redis_url: str) -> None:
        """Can delete all documents."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_hybrid_delete",
        )

        docs = [{"content": "Test doc"}]
        await service.index_documents(docs)
        assert await service.count() == 1

        await service.delete_all()
        assert await service.count() == 0


class TestSearchResult:
    """Test SearchResult dataclass attributes."""

    async def test_result_has_required_fields(self, redis_url: str) -> None:
        """SearchResult should have content, metadata, score, and id."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_result_fields",
        )

        docs = [{"content": "Test document.", "metadata": {"source": "test"}}]
        await service.index_documents(docs)

        results = await service.search(query="test", k=1)

        assert len(results) == 1
        result = results[0]

        # Check required fields
        assert hasattr(result, "content")
        assert hasattr(result, "metadata")
        assert hasattr(result, "score")
        assert hasattr(result, "id")

        assert result.content == "Test document."
        assert result.metadata["source"] == "test"
        assert isinstance(result.score, float)
        assert result.id is not None

    async def test_result_score_range(self, redis_url: str) -> None:
        """Scores should be normalized between 0 and 1."""
        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_result_score",
        )

        docs = [
            {"content": "Redis database."},
            {"content": "Python programming."},
        ]
        await service.index_documents(docs)

        results = await service.search(query="Redis", k=2)

        for result in results:
            assert 0.0 <= result.score <= 1.0
