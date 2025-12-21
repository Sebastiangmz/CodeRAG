"""Integration tests for MCP server."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestMCPServerCreation:
    """Tests for MCP server creation."""

    def test_create_mcp_server_returns_fastmcp(self):
        """Test that create_mcp_server returns a FastMCP instance."""
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

            assert mcp is not None
            assert mcp.name == "CodeRAG"

    def test_mcp_server_has_tools(self):
        """Test that MCP server has registered tools."""
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

            # Check tools are registered
            assert hasattr(mcp, "_tool_manager")

    def test_mcp_server_has_resources(self):
        """Test that MCP server has registered resources."""
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

            # Check resources are registered
            assert hasattr(mcp, "_resource_manager")

    def test_mcp_server_has_prompts(self):
        """Test that MCP server has registered prompts."""
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

            # Check prompts are registered
            assert hasattr(mcp, "_prompt_manager")


class TestMCPResources:
    """Tests for MCP resources."""

    @pytest.mark.asyncio
    async def test_repository_resource_returns_json(self):
        """Test that repository resource returns valid JSON."""
        with patch("coderag.mcp.resources.get_mcp_handlers") as mock_get_handlers:
            from coderag.mcp.resources import get_repository_resource

            mock_handler = MagicMock()

            async def async_get_repo_info(*args, **kwargs):
                return {"id": "test-id", "name": "repo"}

            mock_handler.get_repository_info = async_get_repo_info
            mock_get_handlers.return_value = mock_handler

            result = await get_repository_resource("test-id")

            # Verify it's valid JSON
            parsed = json.loads(result)
            assert parsed["id"] == "test-id"

    @pytest.mark.asyncio
    async def test_repositories_list_resource_returns_json(self):
        """Test that repositories list resource returns valid JSON."""
        with patch("coderag.mcp.resources.get_mcp_handlers") as mock_get_handlers:
            from coderag.mcp.resources import get_repositories_list

            mock_handler = MagicMock()

            async def async_list_repos():
                return {"repositories": [], "count": 0}

            mock_handler.list_repositories = async_list_repos
            mock_get_handlers.return_value = mock_handler

            result = await get_repositories_list()

            # Verify it's valid JSON
            parsed = json.loads(result)
            assert "repositories" in parsed
            assert "count" in parsed


class TestMCPPrompts:
    """Tests for MCP prompts."""

    @pytest.mark.asyncio
    async def test_analyze_repository_prompt_returns_messages(self):
        """Test that analyze_repository prompt returns prompt messages."""
        from coderag.mcp.prompts import analyze_repository

        result = await analyze_repository("https://github.com/owner/repo")

        assert len(result) == 1
        assert result[0].role == "user"
        assert "index_repository" in result[0].content.text

    @pytest.mark.asyncio
    async def test_find_implementation_prompt_returns_messages(self):
        """Test that find_implementation prompt returns prompt messages."""
        from coderag.mcp.prompts import find_implementation

        result = await find_implementation("test-id", "authentication")

        assert len(result) == 1
        assert result[0].role == "user"
        assert "authentication" in result[0].content.text

    @pytest.mark.asyncio
    async def test_code_review_prompt_returns_messages(self):
        """Test that code_review prompt returns prompt messages."""
        from coderag.mcp.prompts import code_review

        result = await code_review("test-id", "security")

        assert len(result) == 1
        assert result[0].role == "user"
        assert "security" in result[0].content.text

    @pytest.mark.asyncio
    async def test_code_review_prompt_without_focus(self):
        """Test that code_review prompt works without focus area."""
        from coderag.mcp.prompts import code_review

        result = await code_review("test-id")

        assert len(result) == 1
        assert result[0].role == "user"


class TestMCPServerIntegration:
    """Integration tests for MCP server mounting."""

    def test_mcp_can_create_streamable_http_app(self):
        """Test that MCP server can create a streamable HTTP app."""
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
            http_app = mcp.streamable_http_app()

            # Should return a Starlette/FastAPI compatible app
            assert http_app is not None
