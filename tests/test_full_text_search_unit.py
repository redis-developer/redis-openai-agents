"""Unit tests for RedisFullTextSearch - TDD RED phase."""

from redis_openai_agents import RedisFullTextSearch


class TestFullTextSearchBasics:
    """Test basic full-text search operations."""

    def test_create_search_index(self, redis_url: str) -> None:
        """Can create a full-text search index."""
        fts = RedisFullTextSearch(name="test_fts", redis_url=redis_url)
        assert fts is not None
        assert fts.name == "test_fts"

    def test_add_single_document(self, redis_url: str) -> None:
        """Can add a single document."""
        fts = RedisFullTextSearch(name="test_add_single", redis_url=redis_url)

        doc = {
            "title": "Test Document",
            "content": "This is test content.",
            "category": "test",
        }
        ids = fts.add_documents([doc])

        assert len(ids) == 1

    def test_add_multiple_documents(self, redis_url: str) -> None:
        """Can add multiple documents."""
        fts = RedisFullTextSearch(name="test_add_multi", redis_url=redis_url)

        docs = [
            {"title": "Doc 1", "content": "Content one"},
            {"title": "Doc 2", "content": "Content two"},
            {"title": "Doc 3", "content": "Content three"},
        ]
        ids = fts.add_documents(docs)

        assert len(ids) == 3

    def test_document_count(self, redis_url: str) -> None:
        """Can count documents."""
        fts = RedisFullTextSearch(name="test_count", redis_url=redis_url)

        assert fts.count() == 0

        fts.add_documents([{"title": "Doc", "content": "Content"}])

        assert fts.count() == 1


class TestFullTextSearch:
    """Test full-text search functionality."""

    def test_basic_keyword_search(self, redis_url: str) -> None:
        """Can search by keyword."""
        fts = RedisFullTextSearch(name="test_keyword", redis_url=redis_url)

        fts.add_documents(
            [
                {"title": "Redis Guide", "content": "Learn about Redis database"},
                {"title": "Python Guide", "content": "Learn about Python programming"},
            ]
        )

        results = fts.search(query="Redis", k=5)

        assert len(results) >= 1
        assert any(
            "Redis" in r.get("title", "") or "Redis" in r.get("content", "") for r in results
        )

    def test_search_returns_score(self, redis_url: str) -> None:
        """Search results include relevance score."""
        fts = RedisFullTextSearch(name="test_score", redis_url=redis_url)

        fts.add_documents([{"title": "Test", "content": "Test content"}])

        results = fts.search(query="test", k=1)

        assert len(results) == 1
        assert "score" in results[0]

    def test_search_respects_k_parameter(self, redis_url: str) -> None:
        """Search respects k limit."""
        fts = RedisFullTextSearch(name="test_k_param", redis_url=redis_url)

        docs = [{"title": f"Doc {i}", "content": "common keyword"} for i in range(10)]
        fts.add_documents(docs)

        results_3 = fts.search(query="common", k=3)
        results_5 = fts.search(query="common", k=5)

        assert len(results_3) == 3
        assert len(results_5) == 5

    def test_search_empty_index(self, redis_url: str) -> None:
        """Search on empty index returns empty list."""
        fts = RedisFullTextSearch(name="test_empty", redis_url=redis_url)

        results = fts.search(query="anything", k=5)

        assert results == []

    def test_multi_word_search(self, redis_url: str) -> None:
        """Can search with multiple words."""
        fts = RedisFullTextSearch(name="test_multi_word", redis_url=redis_url)

        fts.add_documents(
            [
                {
                    "title": "Redis Data Structures",
                    "content": "Redis supports many data structures",
                },
                {"title": "Python Basics", "content": "Python is a programming language"},
            ]
        )

        results = fts.search(query="Redis data", k=5)

        assert len(results) >= 1


class TestFilteredSearch:
    """Test filtered search functionality."""

    def test_search_with_category_filter(self, redis_url: str) -> None:
        """Can filter by category."""
        fts = RedisFullTextSearch(name="test_cat_filter", redis_url=redis_url)

        fts.add_documents(
            [
                {"title": "Tutorial 1", "content": "Redis basics", "category": "tutorial"},
                {"title": "Guide 1", "content": "Redis advanced", "category": "guide"},
                {"title": "Tutorial 2", "content": "Redis more", "category": "tutorial"},
            ]
        )

        results = fts.search(query="Redis", k=10, filter={"category": "tutorial"})

        assert len(results) == 2
        for r in results:
            assert r.get("category") == "tutorial"

    def test_search_with_tag_filter(self, redis_url: str) -> None:
        """Can filter by tags."""
        fts = RedisFullTextSearch(name="test_tag_filter", redis_url=redis_url)

        fts.add_documents(
            [
                {"title": "Doc 1", "content": "Content", "tags": ["redis", "cache"]},
                {"title": "Doc 2", "content": "Content", "tags": ["python"]},
            ]
        )

        results = fts.search(query="*", k=10, filter={"tags": "redis"})

        assert len(results) >= 1


class TestDocumentDeletion:
    """Test document deletion."""

    def test_delete_all(self, redis_url: str) -> None:
        """Can delete all documents."""
        fts = RedisFullTextSearch(name="test_delete_all", redis_url=redis_url)

        fts.add_documents([{"title": "Doc", "content": "Content"}])
        assert fts.count() == 1

        fts.delete_all()

        assert fts.count() == 0

    def test_delete_by_id(self, redis_url: str) -> None:
        """Can delete specific documents."""
        fts = RedisFullTextSearch(name="test_delete_id", redis_url=redis_url)

        ids = fts.add_documents(
            [
                {"title": "Keep", "content": "Keep this"},
                {"title": "Delete", "content": "Delete this"},
            ]
        )

        fts.delete(ids=[ids[1]])

        assert fts.count() == 1
        results = fts.search(query="Keep", k=5)
        assert len(results) == 1
