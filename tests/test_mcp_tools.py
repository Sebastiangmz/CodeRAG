"""Tests for MCP tool registrations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestToolRegistration:
    """Tests for MCP tool registration."""

    def test_tools_are_registered(self):
        """Test that all tools are registered with the MCP server."""
        with patch("coderag.config.get_settings") as mock_get_settings, \
             patch("coderag.mcp.handlers.GitHubURLValidator"), \
             patch("coderag.mcp.handlers.RepositoryLoader"), \
             patch("coderag.mcp.handlers.FileFilter"), \
             patch("coderag.mcp.handlers.CodeChunker"), \
             patch("coderag.mcp.handlers.EmbeddingGenerator"), \
             patch("coderag.mcp.handlers.VectorStore"):

            mock_settings = MagicMock()
            mock_settings.data_dir = MagicMock()
            mock_file = MagicMock()
            mock_file.exists.return_value = False
            mock_settings.data_dir.__truediv__ = MagicMock(return_value=mock_file)
            mock_get_settings.return_value = mock_settings

            from coderag.mcp.server import create_mcp_server

            mcp = create_mcp_server()

            # FastMCP tools are registered and can be verified by checking the mcp object
            # We just verify the server was created with our expected name
            assert mcp is not None
            assert mcp.name == "CodeRAG"

            # Verify the tools module was imported (which registers the tools)
            from coderag.mcp import tools
            assert hasattr(tools, "index_repository")
            assert hasattr(tools, "query_code")
            assert hasattr(tools, "list_repositories")
            assert hasattr(tools, "get_repository_info")
            assert hasattr(tools, "delete_repository")
            assert hasattr(tools, "update_repository")
            assert hasattr(tools, "search_code")


class TestToolDelegation:
    """Tests for tool delegation to handlers."""

    @pytest.mark.asyncio
    async def test_index_repository_delegates_to_handler(self):
        """Test that index_repository tool delegates to handler."""
        with patch("coderag.mcp.tools.get_mcp_handlers") as mock_get_handlers:
            mock_handler = MagicMock()
            mock_handler.index_repository = AsyncMock(
                return_value={"success": True, "repo_id": "test-id"}
            )
            mock_get_handlers.return_value = mock_handler

            from coderag.mcp.tools import index_repository

            result = await index_repository(
                url="https://github.com/owner/repo",
                branch="main",
            )

            mock_handler.index_repository.assert_called_once()
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_query_code_delegates_to_handler(self):
        """Test that query_code tool delegates to handler."""
        with patch("coderag.mcp.tools.get_mcp_handlers") as mock_get_handlers:
            mock_handler = MagicMock()
            mock_handler.query_code = AsyncMock(
                return_value={"answer": "Test answer", "grounded": True}
            )
            mock_get_handlers.return_value = mock_handler

            from coderag.mcp.tools import query_code

            result = await query_code(repo_id="test-id", question="What does this do?")

            mock_handler.query_code.assert_called_once_with(
                repo_id="test-id",
                question="What does this do?",
                top_k=5,
            )
            assert result["answer"] == "Test answer"

    @pytest.mark.asyncio
    async def test_list_repositories_delegates_to_handler(self):
        """Test that list_repositories tool delegates to handler."""
        with patch("coderag.mcp.tools.get_mcp_handlers") as mock_get_handlers:
            mock_handler = MagicMock()
            mock_handler.list_repositories = AsyncMock(
                return_value={"repositories": [], "count": 0}
            )
            mock_get_handlers.return_value = mock_handler

            from coderag.mcp.tools import list_repositories

            result = await list_repositories()

            mock_handler.list_repositories.assert_called_once()
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_get_repository_info_delegates_to_handler(self):
        """Test that get_repository_info tool delegates to handler."""
        with patch("coderag.mcp.tools.get_mcp_handlers") as mock_get_handlers:
            mock_handler = MagicMock()
            mock_handler.get_repository_info = AsyncMock(
                return_value={"id": "test-id", "name": "repo"}
            )
            mock_get_handlers.return_value = mock_handler

            from coderag.mcp.tools import get_repository_info

            result = await get_repository_info(repo_id="test-id")

            mock_handler.get_repository_info.assert_called_once_with(repo_id="test-id")
            assert result["id"] == "test-id"

    @pytest.mark.asyncio
    async def test_delete_repository_delegates_to_handler(self):
        """Test that delete_repository tool delegates to handler."""
        with patch("coderag.mcp.tools.get_mcp_handlers") as mock_get_handlers:
            mock_handler = MagicMock()
            mock_handler.delete_repository = AsyncMock(
                return_value={"success": True, "chunks_deleted": 100}
            )
            mock_get_handlers.return_value = mock_handler

            from coderag.mcp.tools import delete_repository

            result = await delete_repository(repo_id="test-id")

            mock_handler.delete_repository.assert_called_once_with(repo_id="test-id")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_update_repository_delegates_to_handler(self):
        """Test that update_repository tool delegates to handler."""
        with patch("coderag.mcp.tools.get_mcp_handlers") as mock_get_handlers:
            mock_handler = MagicMock()
            mock_handler.update_repository = AsyncMock(
                return_value={"success": True, "files_changed": 5}
            )
            mock_get_handlers.return_value = mock_handler

            from coderag.mcp.tools import update_repository

            result = await update_repository(repo_id="test-id")

            mock_handler.update_repository.assert_called_once_with(repo_id="test-id")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_code_delegates_to_handler(self):
        """Test that search_code tool delegates to handler."""
        with patch("coderag.mcp.tools.get_mcp_handlers") as mock_get_handlers:
            mock_handler = MagicMock()
            mock_handler.search_code = AsyncMock(return_value={"results": [], "count": 0})
            mock_get_handlers.return_value = mock_handler

            from coderag.mcp.tools import search_code

            result = await search_code(
                repo_id="test-id",
                query="function",
                top_k=10,
                file_filter="*.py",
                chunk_type="function",
            )

            mock_handler.search_code.assert_called_once_with(
                repo_id="test-id",
                query="function",
                top_k=10,
                file_filter="*.py",
                chunk_type="function",
            )
            assert result["count"] == 0
