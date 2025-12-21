"""FastMCP server configuration for CodeRAG."""

from mcp.server.fastmcp import FastMCP

# Create the MCP server instance
mcp = FastMCP(
    name="CodeRAG",
    instructions="""CodeRAG is a RAG-based Q&A system for code repositories.

Available capabilities:
- Index GitHub repositories for code analysis
- Ask questions about indexed code with verifiable citations
- Search code semantically
- Manage indexed repositories

Use the tools to:
1. index_repository: Index a new GitHub repository
2. query_code: Ask questions about indexed code
3. search_code: Search code without LLM generation
4. list_repositories: See all indexed repositories
5. get_repository_info: Get details about a specific repository
6. update_repository: Incrementally update a repository
7. delete_repository: Remove an indexed repository

Use the prompts for guided workflows:
- analyze_repository: Comprehensive repository analysis
- find_implementation: Find feature implementations
- code_review: Perform code reviews
""",
)


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server with all tools, resources, and prompts."""
    # Import tools, resources, and prompts to register them
    from coderag.mcp import tools  # noqa: F401
    from coderag.mcp import resources  # noqa: F401
    from coderag.mcp import prompts  # noqa: F401

    return mcp
