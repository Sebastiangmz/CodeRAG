"""MCP resource definitions for CodeRAG."""

import json

from coderag.mcp.handlers import get_mcp_handlers
from coderag.mcp.server import mcp


@mcp.resource("repository://{repo_id}")
async def get_repository_resource(repo_id: str) -> str:
    """Get repository metadata as JSON.

    Args:
        repo_id: Repository ID (full or first 8 characters)

    Returns:
        Repository metadata as JSON string
    """
    handlers = get_mcp_handlers()
    result = await handlers.get_repository_info(repo_id=repo_id)
    return json.dumps(result, indent=2)


@mcp.resource("repositories://list")
async def get_repositories_list() -> str:
    """Get all repositories as JSON.

    Returns:
        List of all repositories as JSON string
    """
    handlers = get_mcp_handlers()
    result = await handlers.list_repositories()
    return json.dumps(result, indent=2)
