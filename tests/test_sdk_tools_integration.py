"""Integration tests for OpenAI SDK tool integrations - TDD RED phase.

These tests define the expected behavior for Redis tools that integrate
with the OpenAI Agents SDK.
"""


class TestRedisFileSearchToolBasics:
    """Test basic RedisFileSearchTool operations."""

    async def test_tool_factory_creates_callable(self, redis_url: str) -> None:
        """Factory function creates a callable tool."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_factory",
        )
        tool = create_redis_file_search_tool(service)

        assert callable(tool)

    async def test_tool_has_name_attribute(self, redis_url: str) -> None:
        """Tool has a name attribute for registration with agents."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_name",
        )
        tool = create_redis_file_search_tool(service)

        assert hasattr(tool, "name")
        assert tool.name == "redis_file_search"

    async def test_tool_has_description(self, redis_url: str) -> None:
        """Tool has a description for LLM context."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_desc",
        )
        tool = create_redis_file_search_tool(service)

        assert hasattr(tool, "description")
        assert len(tool.description) > 0


class TestRedisFileSearchToolExecution:
    """Test RedisFileSearchTool execution."""

    async def test_tool_returns_search_results(self, redis_url: str) -> None:
        """Tool returns search results when executed."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_exec",
        )

        # Index some documents
        await service.index_documents(
            [
                {"content": "Redis is a fast in-memory database."},
                {"content": "Python is a programming language."},
            ]
        )

        tool = create_redis_file_search_tool(service)

        # Execute the tool
        result = await tool(query="Redis database")

        assert result is not None
        assert isinstance(result, str)
        assert "Redis" in result or "database" in result

    async def test_tool_handles_empty_results(self, redis_url: str) -> None:
        """Tool handles case when no documents match."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_empty",
        )

        tool = create_redis_file_search_tool(service)

        # Search empty index
        result = await tool(query="nonexistent topic xyz")

        assert result is not None
        assert isinstance(result, str)
        # Should return a message indicating no results
        assert "no" in result.lower() or "found" in result.lower()

    async def test_tool_accepts_k_parameter(self, redis_url: str) -> None:
        """Tool accepts k parameter to limit results."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_k",
        )

        await service.index_documents(
            [
                {"content": "Redis doc 1."},
                {"content": "Redis doc 2."},
                {"content": "Redis doc 3."},
                {"content": "Redis doc 4."},
            ]
        )

        tool = create_redis_file_search_tool(service)

        # Request only 2 results
        result = await tool(query="Redis", k=2)

        assert result is not None
        # Should contain at most 2 document references

    async def test_tool_accepts_min_score_parameter(self, redis_url: str) -> None:
        """Tool accepts min_score parameter to filter low-relevance results."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_minscore",
        )

        await service.index_documents(
            [
                {"content": "Redis database performance."},
                {"content": "Cooking recipes for pasta."},
            ]
        )

        tool = create_redis_file_search_tool(service)

        # Request high score results only
        result = await tool(query="Redis", min_score=0.5)

        assert result is not None


class TestRedisFileSearchToolMetadata:
    """Test metadata handling in search tool."""

    async def test_tool_includes_metadata_in_results(self, redis_url: str) -> None:
        """Tool includes document metadata in formatted results."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_meta",
        )

        await service.index_documents(
            [
                {
                    "content": "Redis commands guide.",
                    "metadata": {"source": "docs", "category": "database"},
                },
            ]
        )

        tool = create_redis_file_search_tool(service)
        result = await tool(query="Redis commands")

        # Metadata should be visible in formatted output
        assert result is not None
        result_lower = result.lower()
        assert "source" in result_lower or "docs" in result_lower or "database" in result_lower


class TestRedisFileSearchToolErrorHandling:
    """Test error handling in search tool."""

    async def test_tool_handles_invalid_query_gracefully(self, redis_url: str) -> None:
        """Tool handles edge case queries without crashing."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_error",
        )

        tool = create_redis_file_search_tool(service)

        # Various edge cases
        result_empty = await tool(query="")
        assert result_empty is not None

        result_long = await tool(query="a" * 1000)
        assert result_long is not None


class TestRedisFileSearchToolCustomization:
    """Test tool customization options."""

    async def test_custom_tool_name(self, redis_url: str) -> None:
        """Can create tool with custom name."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_custom_name",
        )

        tool = create_redis_file_search_tool(
            service,
            name="search_knowledge_base",
        )

        assert tool.name == "search_knowledge_base"

    async def test_custom_description(self, redis_url: str) -> None:
        """Can create tool with custom description."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_custom_desc",
        )

        custom_desc = "Search the company documentation for relevant information."
        tool = create_redis_file_search_tool(
            service,
            description=custom_desc,
        )

        assert tool.description == custom_desc

    async def test_default_k_value(self, redis_url: str) -> None:
        """Can set default k value for the tool."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_default_k",
        )

        tool = create_redis_file_search_tool(
            service,
            default_k=3,
        )

        assert hasattr(tool, "default_k")
        assert tool.default_k == 3


class TestRedisFileSearchToolSchema:
    """Test tool parameter schema for LLM function calling."""

    async def test_tool_has_parameters_schema(self, redis_url: str) -> None:
        """Tool exposes parameter schema for LLM function calling."""
        from redis_openai_agents import HybridSearchService, create_redis_file_search_tool

        service = HybridSearchService(
            redis_url=redis_url,
            index_name="test_tool_schema",
        )

        tool = create_redis_file_search_tool(service)

        # Should have parameters attribute for function calling
        assert hasattr(tool, "parameters")
        params = tool.parameters

        # Should define query as required parameter
        assert isinstance(params, dict)
        assert "query" in params["properties"]
        assert "query" in params.get("required", [])
