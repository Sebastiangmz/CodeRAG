"""CLI entry point for running MCP server in stdio mode."""

import sys
import os

# Suppress all stdout output except MCP protocol
os.environ["PYTHONUNBUFFERED"] = "1"

# Redirect any stray prints to stderr
import io
_original_stdout = sys.stdout


def main():
    """Run the MCP server in stdio mode for Claude Desktop."""
    # Suppress logging to stdout - redirect to stderr
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(message)s"
    )

    # Suppress structlog output
    import structlog
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    )

    from coderag.mcp.server import create_mcp_server

    mcp = create_mcp_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
