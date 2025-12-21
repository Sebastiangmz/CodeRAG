"""MCP (Model Context Protocol) server for CodeRAG."""

from coderag.mcp.handlers import MCPHandlers, get_mcp_handlers
from coderag.mcp.server import create_mcp_server, mcp

__all__ = [
    "MCPHandlers",
    "get_mcp_handlers",
    "create_mcp_server",
    "mcp",
]
