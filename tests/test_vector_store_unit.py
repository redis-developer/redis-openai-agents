"""Unit tests for RedisVectorStore - TDD RED phase.

These tests define the expected API and behavior for the RedisVectorStore class.
They should fail initially until the implementation is complete.
"""

from redis_openai_agents import RedisVectorStore


class TestVectorStoreBasics:
    """Test basic vector store operations."""

    def test_create_vector_store(self, redis_url: str) -> None:
        """Vector store can be instantiated with redis_url."""
        store = RedisVectorStore(name="test_store", redis_url=redis_url)
        assert store.name == "test_store"

    def test_add_single_document(self, redis_url: str) -> None:
        """Can add a single document to the store."""
        store = RedisVectorStore(name="test_single", redis_url=redis_url)

        doc = {"content": "Redis is a fast database.", "metadata": {"source": "test"}}
        ids = store.add_documents([doc])

        assert len(ids) == 1
        assert ids[0] is not None

    def test_add_multiple_documents(self, redis_url: str) -> None:
        """Can add multiple documents at once."""
        store = RedisVectorStore(name="test_multi", redis_url=redis_url)

        docs = [
            {"content": "Document one", "metadata": {"id": 1}},
            {"content": "Document two", "metadata": {"id": 2}},
            {"content": "Document three", "metadata": {"id": 3}},
        ]
        ids = store.add_documents(docs)

        assert len(ids) == 3

    def test_document_count(self, redis_url: str) -> None:
        """Can count documents in the store."""
        store = RedisVectorStore(name="test_count", redis_url=redis_url)

        assert store.count() == 0

        docs = [
            {"content": "Doc 1"},
            {"content": "Doc 2"},
        ]
        store.add_documents(docs)

        assert store.count() == 2


class TestVectorSearch:
    """Test vector similarity search."""

    def test_basic_search(self, redis_url: str) -> None:
        """Can search for similar documents."""
        store = RedisVectorStore(name="test_search", redis_url=redis_url)

        docs = [
            {"content": "Python is a programming language."},
            {"content": "Redis is an in-memory database."},
            {"content": "Machine learning uses algorithms."},
        ]
        store.add_documents(docs)

        results = store.search(query="What is Python?", k=2)

        assert len(results) >= 1
        assert len(results) <= 2
        assert results[0]["content"]
        assert "score" in results[0]
        # Top result should be relevant to "Python"
        assert "Python" in results[0]["content"]

    def test_search_returns_relevant_results(self, redis_url: str) -> None:
        """Search returns semantically relevant documents."""
        store = RedisVectorStore(name="test_relevance", redis_url=redis_url)

        docs = [
            {"content": "Redis is a fast in-memory data store."},
            {"content": "Python is great for data science."},
            {"content": "Cooking pasta requires boiling water."},
        ]
        store.add_documents(docs)

        results = store.search(query="Tell me about Redis database", k=1)

        assert len(results) == 1
        assert "Redis" in results[0]["content"]

    def test_search_with_k_parameter(self, redis_url: str) -> None:
        """Search respects k parameter for result count."""
        store = RedisVectorStore(name="test_k", redis_url=redis_url)

        docs = [{"content": f"Document number {i}"} for i in range(10)]
        store.add_documents(docs)

        results_3 = store.search(query="document", k=3)
        results_5 = store.search(query="document", k=5)

        assert len(results_3) == 3
        assert len(results_5) == 5

    def test_search_empty_store(self, redis_url: str) -> None:
        """Search on empty store returns empty list."""
        store = RedisVectorStore(name="test_empty_search", redis_url=redis_url)

        results = store.search(query="anything", k=5)

        assert results == []


class TestMetadataFiltering:
    """Test metadata filtering in search."""

    def test_search_with_filter(self, redis_url: str) -> None:
        """Can filter search results by metadata."""
        store = RedisVectorStore(name="test_filter", redis_url=redis_url)

        docs = [
            {"content": "Redis overview", "metadata": {"topic": "database"}},
            {"content": "Python basics", "metadata": {"topic": "programming"}},
            {"content": "Redis clustering", "metadata": {"topic": "database"}},
        ]
        store.add_documents(docs)

        # Filter by topic=database
        results = store.search(query="overview", k=10, filter={"topic": "database"})

        assert len(results) == 2
        for r in results:
            assert r["metadata"]["topic"] == "database"

    def test_metadata_preserved_in_results(self, redis_url: str) -> None:
        """Metadata is returned with search results."""
        store = RedisVectorStore(name="test_meta_preserve", redis_url=redis_url)

        docs = [
            {
                "content": "Test document",
                "metadata": {"source": "unit_test", "version": "1.0"},
            }
        ]
        store.add_documents(docs)

        results = store.search(query="test", k=1)

        assert results[0]["metadata"]["source"] == "unit_test"
        assert results[0]["metadata"]["version"] == "1.0"


class TestDocumentDeletion:
    """Test document deletion operations."""

    def test_delete_all(self, redis_url: str) -> None:
        """Can delete all documents from the store."""
        store = RedisVectorStore(name="test_delete_all", redis_url=redis_url)

        docs = [{"content": f"Doc {i}"} for i in range(5)]
        store.add_documents(docs)
        assert store.count() == 5

        store.delete_all()

        assert store.count() == 0

    def test_delete_by_id(self, redis_url: str) -> None:
        """Can delete specific documents by ID."""
        store = RedisVectorStore(name="test_delete_id", redis_url=redis_url)

        docs = [
            {"content": "Keep this one"},
            {"content": "Delete this one"},
        ]
        ids = store.add_documents(docs)
        assert store.count() == 2

        # Delete the second document
        store.delete(ids=[ids[1]])

        assert store.count() == 1
        results = store.search(query="keep", k=10)
        assert any("Keep" in r["content"] for r in results)


class TestDocumentWithoutMetadata:
    """Test documents without metadata."""

    def test_add_document_without_metadata(self, redis_url: str) -> None:
        """Can add documents with only content."""
        store = RedisVectorStore(name="test_no_meta", redis_url=redis_url)

        docs = [{"content": "Just content, no metadata"}]
        ids = store.add_documents(docs)

        assert len(ids) == 1

    def test_search_document_without_metadata(self, redis_url: str) -> None:
        """Can search and retrieve documents without metadata."""
        store = RedisVectorStore(name="test_search_no_meta", redis_url=redis_url)

        docs = [{"content": "Document without metadata"}]
        store.add_documents(docs)

        results = store.search(query="document", k=1)

        assert len(results) == 1
        assert results[0]["content"] == "Document without metadata"


class TestHybridSearch:
    """Test hybrid search combining vector + BM25."""

    def test_hybrid_search_basic(self, redis_url: str) -> None:
        """Hybrid search returns results combining vector and text search."""
        store = RedisVectorStore(name="test_hybrid", redis_url=redis_url)

        docs = [
            {"content": "Redis is an in-memory data structure store."},
            {"content": "Python programming language basics."},
            {"content": "Database performance optimization techniques."},
        ]
        store.add_documents(docs)

        results = store.hybrid_search(query="Redis database", k=2)

        assert len(results) <= 2
        assert "content" in results[0]
        assert "score" in results[0]

    def test_hybrid_search_with_weights(self, redis_url: str) -> None:
        """Hybrid search respects text_weight and vector_weight."""
        store = RedisVectorStore(name="test_hybrid_weights", redis_url=redis_url)

        docs = [
            {"content": "Redis commands and operations."},
            {"content": "Memory management in systems."},
        ]
        store.add_documents(docs)

        # Heavy text weight - should favor exact keyword matches
        results_text = store.hybrid_search(
            query="Redis",
            k=2,
            text_weight=0.9,
            vector_weight=0.1,
        )

        # Heavy vector weight - should favor semantic similarity
        results_vector = store.hybrid_search(
            query="Redis",
            k=2,
            text_weight=0.1,
            vector_weight=0.9,
        )

        # Both should return results, and top result should contain "Redis"
        assert len(results_text) >= 1
        assert len(results_vector) >= 1
        assert "Redis" in results_text[0]["content"]
        assert "Redis" in results_vector[0]["content"]

    def test_hybrid_search_finds_keyword_matches(self, redis_url: str) -> None:
        """Hybrid search finds documents with exact keyword matches."""
        store = RedisVectorStore(name="test_hybrid_keyword", redis_url=redis_url)

        docs = [
            {"content": "The quick brown fox jumps over the lazy dog."},
            {"content": "Redis is blazingly fast for caching."},
            {"content": "Database indexing improves query speed."},
        ]
        store.add_documents(docs)

        # Search for exact keyword "blazingly"
        results = store.hybrid_search(query="blazingly fast", k=3)

        # Should find the Redis document due to keyword match
        assert any("blazingly" in r["content"] for r in results)

    def test_hybrid_search_empty_store(self, redis_url: str) -> None:
        """Hybrid search on empty store returns empty list."""
        store = RedisVectorStore(name="test_hybrid_empty", redis_url=redis_url)

        results = store.hybrid_search(query="anything", k=5)

        assert results == []

    def test_hybrid_search_with_filter(self, redis_url: str) -> None:
        """Hybrid search can filter by metadata."""
        store = RedisVectorStore(name="test_hybrid_filter", redis_url=redis_url)

        docs = [
            {"content": "Redis caching strategies.", "metadata": {"category": "database"}},
            {"content": "Python web frameworks.", "metadata": {"category": "programming"}},
            {"content": "Redis cluster setup.", "metadata": {"category": "database"}},
        ]
        store.add_documents(docs)

        results = store.hybrid_search(
            query="Redis",
            k=10,
            filter={"category": "database"},
        )

        assert len(results) == 2
        for r in results:
            assert r["metadata"]["category"] == "database"


class TestEmbeddingsCacheIntegration:
    """Test RedisVectorStore integration with RedisVL EmbeddingsCache."""

    def test_embeddings_cache_populated_after_add(self, redis_url: str) -> None:
        """Passing an EmbeddingsCache populates it as documents are embedded."""
        from redisvl.extensions.cache.embeddings import EmbeddingsCache

        cache = EmbeddingsCache(name="vec_emb_cache_test", redis_url=redis_url)
        store = RedisVectorStore(
            name="test_emb_cache",
            redis_url=redis_url,
            embeddings_cache=cache,
        )

        content = "Redis stores embeddings efficiently."
        store.add_documents([{"content": content, "metadata": {"id": 1}}])

        hit = cache.get(content=content, model_name=store.vectorizer_model)
        assert hit is not None
        assert "embedding" in hit

    def test_embeddings_cache_reused_on_repeat_query(self, redis_url: str) -> None:
        """The cache returns the same embedding for repeated identical content."""
        from redisvl.extensions.cache.embeddings import EmbeddingsCache

        cache = EmbeddingsCache(name="vec_emb_cache_reuse", redis_url=redis_url)
        store = RedisVectorStore(
            name="test_emb_reuse",
            redis_url=redis_url,
            embeddings_cache=cache,
        )

        content = "Deterministic embedding content."
        store.add_documents([{"content": content, "metadata": {"id": 1}}])
        first = cache.get(content=content, model_name=store.vectorizer_model)

        store.add_documents([{"content": content, "metadata": {"id": 2}}])
        second = cache.get(content=content, model_name=store.vectorizer_model)

        assert first is not None and second is not None
        assert first["embedding"] == second["embedding"]
