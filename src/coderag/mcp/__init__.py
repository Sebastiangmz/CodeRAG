"""MCP (Model Context Protocol) server for CodeRAG."""

from typing import Any

from coderag.mcp.handlers import MCPHandlers, get_mcp_handlers

__all__ = [
    "MCPHandlers",
    "get_mcp_handlers",
    "create_mcp_server",
    "mcp",
]


def __getattr__(name: str) -> Any:
    if name in {"create_mcp_server", "mcp"}:
        from coderag.mcp import server

        return getattr(server, name)
    if name in {"tools", "resources", "prompts"}:
        import importlib

        return importlib.import_module(f"coderag.mcp.{name}")
    raise AttributeError(name)
