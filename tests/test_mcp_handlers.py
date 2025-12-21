"""Unit tests for MCP handlers."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import sys

import pytest

from coderag.models.repository import Repository, RepositoryStatus


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.data_dir = MagicMock()
    settings.data_dir.__truediv__ = MagicMock(return_value=MagicMock())
    settings.ingestion.batch_size = 10
    settings.ingestion.chunk_size = 1000
    return settings


@pytest.fixture
def sample_repository():
    """Create a sample repository for testing."""
    return Repository(
        id="test-repo-id-12345",
        url="https://github.com/owner/repo",
        branch="main",
        status=RepositoryStatus.READY,
        chunk_count=100,
        indexed_at=datetime.now(),
        last_commit="abc123",
    )


@pytest.fixture
def mock_mcp_handlers():
    """Create a mock MCPHandlers instance with all dependencies mocked."""
    with patch.dict(sys.modules, {}):
        # Mock all the heavy dependencies
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
            mock_file.parent = MagicMock()
            mock_settings.data_dir.__truediv__ = MagicMock(return_value=mock_file)
            mock_settings.ingestion.batch_size = 10
            mock_get_settings.return_value = mock_settings

            from coderag.mcp.handlers import MCPHandlers
            handlers = MCPHandlers()
            yield handlers


class TestMCPHandlersSingleton:
    """Tests for MCPHandlers singleton pattern."""

    def test_get_mcp_handlers_returns_same_instance(self):
        """Test that get_mcp_handlers returns the same instance."""
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

            # Reset singleton
            import coderag.mcp.handlers as handlers_module
            handlers_module._mcp_handlers = None

            from coderag.mcp.handlers import get_mcp_handlers

            handler1 = get_mcp_handlers()
            handler2 = get_mcp_handlers()

            assert handler1 is handler2


class TestListRepositories:
    """Tests for list_repositories handler."""

    @pytest.mark.asyncio
    async def test_list_repositories_empty(self, mock_mcp_handlers):
        """Test listing repositories when none exist."""
        result = await mock_mcp_handlers.list_repositories()

        assert result["count"] == 0
        assert result["repositories"] == []

    @pytest.mark.asyncio
    async def test_list_repositories_with_repos(self, mock_mcp_handlers, sample_repository):
        """Test listing repositories when some exist."""
        mock_mcp_handlers.repositories = {sample_repository.id: sample_repository}
        # Prevent reload from disk
        mock_mcp_handlers._reload_repositories = MagicMock()

        result = await mock_mcp_handlers.list_repositories()

        assert result["count"] == 1
        assert len(result["repositories"]) == 1
        assert result["repositories"][0]["id"] == sample_repository.id
        assert result["repositories"][0]["name"] == sample_repository.full_name


class TestGetRepositoryInfo:
    """Tests for get_repository_info handler."""

    @pytest.mark.asyncio
    async def test_get_repository_info_found(self, mock_mcp_handlers, sample_repository):
        """Test getting info for an existing repository."""
        mock_mcp_handlers.repositories = {sample_repository.id: sample_repository}
        mock_mcp_handlers._reload_repositories = MagicMock()
        mock_mcp_handlers.vectorstore.get_repo_files.return_value = ["file1.py", "file2.py"]

        result = await mock_mcp_handlers.get_repository_info(sample_repository.id)

        assert result["id"] == sample_repository.id
        assert result["name"] == sample_repository.name
        assert result["url"] == sample_repository.url
        assert "indexed_files" in result

    @pytest.mark.asyncio
    async def test_get_repository_info_partial_id(self, mock_mcp_handlers, sample_repository):
        """Test getting info using partial repository ID."""
        mock_mcp_handlers.repositories = {sample_repository.id: sample_repository}
        mock_mcp_handlers._reload_repositories = MagicMock()
        mock_mcp_handlers.vectorstore.get_repo_files.return_value = []

        # Use first 8 characters
        result = await mock_mcp_handlers.get_repository_info("test-rep")

        assert result["id"] == sample_repository.id

    @pytest.mark.asyncio
    async def test_get_repository_info_not_found(self, mock_mcp_handlers):
        """Test getting info for non-existent repository."""
        result = await mock_mcp_handlers.get_repository_info("nonexistent")

        assert "error" in result


class TestDeleteRepository:
    """Tests for delete_repository handler."""

    @pytest.mark.asyncio
    async def test_delete_repository_success(self, mock_mcp_handlers, sample_repository):
        """Test successfully deleting a repository."""
        mock_mcp_handlers.repositories = {sample_repository.id: sample_repository}
        mock_mcp_handlers._reload_repositories = MagicMock()
        mock_mcp_handlers._save_repositories = MagicMock()
        mock_mcp_handlers.vectorstore.get_repo_chunk_count.return_value = 100
        mock_mcp_handlers.vectorstore.delete_repo_chunks = MagicMock()

        result = await mock_mcp_handlers.delete_repository(sample_repository.id)

        assert result["success"] is True
        assert result["chunks_deleted"] == 100
        assert sample_repository.id not in mock_mcp_handlers.repositories

    @pytest.mark.asyncio
    async def test_delete_repository_not_found(self, mock_mcp_handlers):
        """Test deleting non-existent repository."""
        result = await mock_mcp_handlers.delete_repository("nonexistent")

        assert result["success"] is False
        assert "error" in result


class TestQueryCode:
    """Tests for query_code handler."""

    @pytest.mark.asyncio
    async def test_query_code_repo_not_found(self, mock_mcp_handlers):
        """Test querying non-existent repository."""
        result = await mock_mcp_handlers.query_code("nonexistent", "What does this do?")

        assert result["grounded"] is False
        assert result["answer"] == ""
        assert "error" in result

    @pytest.mark.asyncio
    async def test_query_code_repo_not_ready(self, mock_mcp_handlers, sample_repository):
        """Test querying repository that's not ready."""
        sample_repository.status = RepositoryStatus.INDEXING
        mock_mcp_handlers.repositories = {sample_repository.id: sample_repository}
        mock_mcp_handlers._reload_repositories = MagicMock()

        result = await mock_mcp_handlers.query_code(sample_repository.id, "What does this do?")

        assert result["grounded"] is False
        assert "not ready" in result["error"]


class TestSearchCode:
    """Tests for search_code handler."""

    @pytest.mark.asyncio
    async def test_search_code_repo_not_found(self, mock_mcp_handlers):
        """Test searching non-existent repository."""
        result = await mock_mcp_handlers.search_code("nonexistent", "function")

        assert result["results"] == []
        assert "error" in result


class TestUpdateRepository:
    """Tests for update_repository handler."""

    @pytest.mark.asyncio
    async def test_update_repository_not_found(self, mock_mcp_handlers):
        """Test updating non-existent repository."""
        result = await mock_mcp_handlers.update_repository("nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_update_repository_no_previous_commit(self, mock_mcp_handlers, sample_repository):
        """Test updating repository without previous indexing."""
        sample_repository.last_commit = None
        mock_mcp_handlers.repositories = {sample_repository.id: sample_repository}
        mock_mcp_handlers._reload_repositories = MagicMock()

        result = await mock_mcp_handlers.update_repository(sample_repository.id)

        assert result["success"] is False
        assert "No previous indexing" in result["error"]
