"""Unit tests for RedisFileSearchTool - testing with mocks."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock


@dataclass
class MockSearchResult:
    """Mock search result for testing."""

    id: str
    content: str
    metadata: dict
    score: float


class TestRedisFileSearchToolCreation:
    """Test tool creation and configuration."""

    def test_create_tool_with_defaults(self) -> None:
        """Tool can be created with default settings."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        tool = create_redis_file_search_tool(mock_service)

        assert tool.name == "redis_file_search"
        assert tool.default_k == 5
        assert tool.default_min_score == 0.0

    def test_create_tool_with_custom_name(self) -> None:
        """Tool can be created with custom name."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        tool = create_redis_file_search_tool(mock_service, name="search_docs")

        assert tool.name == "search_docs"

    def test_create_tool_with_custom_description(self) -> None:
        """Tool can be created with custom description."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        custom_desc = "Search company knowledge base."
        tool = create_redis_file_search_tool(mock_service, description=custom_desc)

        assert tool.description == custom_desc

    def test_create_tool_with_custom_k(self) -> None:
        """Tool can be created with custom default_k."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        tool = create_redis_file_search_tool(mock_service, default_k=10)

        assert tool.default_k == 10

    def test_create_tool_with_custom_min_score(self) -> None:
        """Tool can be created with custom default_min_score."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        tool = create_redis_file_search_tool(mock_service, default_min_score=0.7)

        assert tool.default_min_score == 0.7


class TestRedisFileSearchToolSchema:
    """Test tool parameter schema."""

    def test_parameters_schema_structure(self) -> None:
        """Tool has valid JSON schema for parameters."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        tool = create_redis_file_search_tool(mock_service)

        params = tool.parameters
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

    def test_query_parameter_defined(self) -> None:
        """Query parameter is defined in schema."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        tool = create_redis_file_search_tool(mock_service)

        props = tool.parameters["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"

    def test_query_is_required(self) -> None:
        """Query parameter is required."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        tool = create_redis_file_search_tool(mock_service)

        assert "query" in tool.parameters["required"]

    def test_k_parameter_defined(self) -> None:
        """K parameter is defined in schema."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        tool = create_redis_file_search_tool(mock_service)

        props = tool.parameters["properties"]
        assert "k" in props
        assert props["k"]["type"] == "integer"

    def test_min_score_parameter_defined(self) -> None:
        """Min score parameter is defined in schema."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        tool = create_redis_file_search_tool(mock_service)

        props = tool.parameters["properties"]
        assert "min_score" in props
        assert props["min_score"]["type"] == "number"


class TestRedisFileSearchToolExecution:
    """Test tool execution logic."""

    async def test_calls_service_with_query(self) -> None:
        """Tool calls service search with provided query."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=[])
        tool = create_redis_file_search_tool(mock_service)

        await tool(query="test query")

        mock_service.search.assert_called_once()
        call_args = mock_service.search.call_args
        assert call_args.kwargs["query"] == "test query"

    async def test_uses_default_k_when_not_provided(self) -> None:
        """Tool uses default_k when k not provided."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=[])
        tool = create_redis_file_search_tool(mock_service, default_k=7)

        await tool(query="test")

        call_args = mock_service.search.call_args
        assert call_args.kwargs["k"] == 7

    async def test_uses_provided_k(self) -> None:
        """Tool uses provided k value."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=[])
        tool = create_redis_file_search_tool(mock_service)

        await tool(query="test", k=15)

        call_args = mock_service.search.call_args
        assert call_args.kwargs["k"] == 15

    async def test_filters_by_min_score(self) -> None:
        """Tool filters results below min_score."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(
            return_value=[
                MockSearchResult(id="1", content="High score", metadata={}, score=0.9),
                MockSearchResult(id="2", content="Low score", metadata={}, score=0.3),
            ]
        )
        tool = create_redis_file_search_tool(mock_service)

        result = await tool(query="test", min_score=0.5)

        assert "High score" in result
        assert "Low score" not in result

    async def test_returns_no_results_message(self) -> None:
        """Tool returns message when no results found."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=[])
        tool = create_redis_file_search_tool(mock_service)

        result = await tool(query="nonexistent")

        assert "no" in result.lower() or "found" in result.lower()


class TestRedisFileSearchToolFormatting:
    """Test result formatting."""

    async def test_formats_single_result(self) -> None:
        """Tool formats single result properly."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(
            return_value=[
                MockSearchResult(id="1", content="Test content", metadata={}, score=0.8),
            ]
        )
        tool = create_redis_file_search_tool(mock_service)

        result = await tool(query="test")

        assert "Test content" in result
        assert "0.80" in result  # Score formatted

    async def test_formats_multiple_results(self) -> None:
        """Tool formats multiple results properly."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(
            return_value=[
                MockSearchResult(id="1", content="First result", metadata={}, score=0.9),
                MockSearchResult(id="2", content="Second result", metadata={}, score=0.7),
            ]
        )
        tool = create_redis_file_search_tool(mock_service)

        result = await tool(query="test")

        assert "First result" in result
        assert "Second result" in result
        assert "Result 1" in result
        assert "Result 2" in result

    async def test_includes_metadata_in_output(self) -> None:
        """Tool includes metadata in formatted output."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(
            return_value=[
                MockSearchResult(
                    id="1",
                    content="Content",
                    metadata={"source": "docs", "category": "api"},
                    score=0.8,
                ),
            ]
        )
        tool = create_redis_file_search_tool(mock_service)

        result = await tool(query="test")

        assert "source" in result or "docs" in result
        assert "category" in result or "api" in result

    async def test_shows_result_count(self) -> None:
        """Tool shows count of results found."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(
            return_value=[
                MockSearchResult(id="1", content="A", metadata={}, score=0.9),
                MockSearchResult(id="2", content="B", metadata={}, score=0.8),
                MockSearchResult(id="3", content="C", metadata={}, score=0.7),
            ]
        )
        tool = create_redis_file_search_tool(mock_service)

        result = await tool(query="test")

        assert "3" in result  # Shows count


class TestRedisFileSearchToolErrorHandling:
    """Test error handling."""

    async def test_handles_empty_query(self) -> None:
        """Tool handles empty query gracefully."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=[])
        tool = create_redis_file_search_tool(mock_service)

        result = await tool(query="")

        assert result is not None
        assert isinstance(result, str)

    async def test_handles_service_error(self) -> None:
        """Tool handles service errors gracefully."""
        from redis_openai_agents.sdk_tools import create_redis_file_search_tool

        mock_service = MagicMock()
        mock_service.search = AsyncMock(side_effect=Exception("Connection failed"))
        tool = create_redis_file_search_tool(mock_service)

        result = await tool(query="test")

        assert "error" in result.lower()
        assert "Connection failed" in result
