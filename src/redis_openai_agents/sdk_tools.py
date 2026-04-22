"""OpenAI Agents SDK tool integrations for Redis.

This module provides tools that can be used with the OpenAI Agents SDK
to integrate Redis-based search capabilities.

Example:
    >>> from redis_openai_agents import HybridSearchService, create_redis_file_search_tool
    >>>
    >>> # Create the search service
    >>> service = HybridSearchService(redis_url="redis://localhost:6379", index_name="docs")
    >>>
    >>> # Index some documents
    >>> await service.index_documents([
    ...     {"content": "Redis is a fast database.", "metadata": {"source": "docs"}},
    ... ])
    >>>
    >>> # Create the tool
    >>> search_tool = create_redis_file_search_tool(service)
    >>>
    >>> # Use with an agent
    >>> from agents import Agent
    >>> agent = Agent(name="Support", tools=[search_tool])
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hybrid import HybridSearchService


@dataclass
class RedisFileSearchTool:
    """A callable tool for searching documents using Redis hybrid search.

    This tool wraps a HybridSearchService to provide RAG capabilities
    for OpenAI Agents SDK agents.

    Attributes:
        name: Tool name for registration with agents.
        description: Tool description for LLM context.
        parameters: JSON schema for function calling parameters.
        default_k: Default number of results to return.
    """

    _service: "HybridSearchService"
    name: str = "redis_file_search"
    description: str = (
        "Search documents using semantic similarity and keyword matching. "
        "Use this tool to find relevant information from the knowledge base."
    )
    default_k: int = 5
    default_min_score: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize parameter schema after dataclass init."""
        if not self.parameters:
            self.parameters = {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents.",
                    },
                    "k": {
                        "type": "integer",
                        "description": f"Maximum number of results to return (default: {self.default_k}).",
                        "default": self.default_k,
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum similarity score (0-1) for results.",
                        "default": self.default_min_score,
                    },
                },
                "required": ["query"],
            }

    async def __call__(
        self,
        query: str,
        k: int | None = None,
        min_score: float | None = None,
    ) -> str:
        """Execute the search and return formatted results.

        Args:
            query: The search query.
            k: Maximum number of results (uses default_k if not provided).
            min_score: Minimum similarity score for filtering results.

        Returns:
            Formatted search results as a string for LLM consumption.
        """
        k = k if k is not None else self.default_k
        min_score = min_score if min_score is not None else self.default_min_score

        try:
            # Handle empty query
            if not query or not query.strip():
                return "Please provide a search query."

            # Execute search
            results = await self._service.search(
                query=query,
                k=k,
            )

            # Filter by minimum score
            if min_score > 0:
                results = [r for r in results if r.score >= min_score]

            # Format results for LLM
            if not results:
                return "No documents found matching your query."

            output_parts = [f"Found {len(results)} relevant document(s):\n"]

            for i, result in enumerate(results, 1):
                output_parts.append(f"\n--- Result {i} (score: {result.score:.2f}) ---")
                output_parts.append(f"Content: {result.content}")

                if result.metadata:
                    metadata_str = ", ".join(f"{k}: {v}" for k, v in result.metadata.items())
                    output_parts.append(f"Metadata: {metadata_str}")

            return "\n".join(output_parts)

        except Exception as e:
            # Return error as string (non-fatal, sent to LLM)
            return f"Search error: {str(e)}. Please try a different query."


def create_redis_file_search_tool(
    service: "HybridSearchService",
    name: str = "redis_file_search",
    description: str | None = None,
    default_k: int = 5,
    default_min_score: float = 0.0,
) -> RedisFileSearchTool:
    """Factory function to create a Redis file search tool.

    This function creates a callable tool that wraps a HybridSearchService
    for use with the OpenAI Agents SDK.

    Args:
        service: The HybridSearchService instance to use for searching.
        name: Custom tool name (default: "redis_file_search").
        description: Custom tool description for LLM context.
        default_k: Default number of results to return.
        default_min_score: Default minimum similarity score for filtering.

    Returns:
        A RedisFileSearchTool instance that can be used with OpenAI agents.

    Example:
        >>> service = HybridSearchService(redis_url="redis://localhost:6379", index_name="docs")
        >>> tool = create_redis_file_search_tool(service, name="search_docs", default_k=10)
        >>> result = await tool(query="Redis performance")
    """
    default_description = (
        "Search documents using semantic similarity and keyword matching. "
        "Use this tool to find relevant information from the knowledge base."
    )

    return RedisFileSearchTool(
        _service=service,
        name=name,
        description=description or default_description,
        default_k=default_k,
        default_min_score=default_min_score,
    )
