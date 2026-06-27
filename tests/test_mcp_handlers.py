"""Unit tests for service-backed MCP handlers."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from coderag.mcp.handlers import MCPHandlers, get_mcp_handlers
from coderag.models.repository import Repository, RepositoryStatus
from coderag.services.registry import RepositoryRegistry


class FakeIndexingService:
    def __init__(self, registry: RepositoryRegistry) -> None:
        self.registry = registry
        self.deleted: list[str] = []

    async def index_repository(self, url, options):
        repo = self.registry.add(Repository(id="indexed-repo", url=url, branch=options.branch or "main", status=RepositoryStatus.READY))
        return SimpleNamespace(repo=repo, files_processed=2, chunks_indexed=3)

    def delete_repository(self, repo_id: str):
        repo = self.registry.remove(repo_id)
        if repo is None:
            return None
        self.deleted.append(repo.id)
        return SimpleNamespace(repo=repo, chunks_deleted=3)

    def update_repository(self, repo_id: str):
        repo = self.registry.get(repo_id)
        if repo is None:
            raise ValueError(f"Repository not found: {repo_id}")
        if not repo.last_commit:
            raise ValueError("No previous indexing found. Please re-index the full repository.")
        return SimpleNamespace(
            already_up_to_date=True,
            files_changed=0,
            files_added=0,
            files_modified=0,
            files_deleted=0,
            chunks_added=0,
            chunks_deleted=0,
            total_chunks=repo.chunk_count,
        )

    def get_indexed_files(self, repo_id: str):
        _ = repo_id
        return {"src/app.py"}


class FakeRetrievalService:
    def __init__(self, registry: RepositoryRegistry) -> None:
        self.registry = registry

    def query_code(self, repo_id: str, question: str, top_k: int = 5):
        _ = (question, top_k)
        repo = self.registry.get(repo_id)
        if repo is None:
            return SimpleNamespace(response=None, repo=None, error=f"Repository not found: {repo_id}")
        if repo.status is not RepositoryStatus.READY:
            return SimpleNamespace(response=None, repo=repo, error=f"Repository not ready: status is {repo.status.value}")
        response = SimpleNamespace(answer="Test answer", citations=[], retrieved_chunks=[], grounded=False)
        return SimpleNamespace(response=response, repo=repo, error=None)

    def search_code(self, repo_id: str, query: str, top_k: int = 10, file_filter=None, chunk_type=None):
        _ = (query, top_k, file_filter, chunk_type)
        repo = self.registry.get(repo_id)
        if repo is None:
            raise ValueError(f"Repository not found: {repo_id}")
        return [
            (
                SimpleNamespace(
                    file_path="src/app.py",
                    start_line=1,
                    end_line=3,
                    chunk_type="function",
                    content="def app(): pass",
                ),
                0.876,
            )
        ]


@pytest.fixture
def registry(tmp_path):
    return RepositoryRegistry(tmp_path / "repositories.json")


@pytest.fixture
def sample_repository():
    return Repository(
        id="test-repo-123456",
        url="https://github.com/owner/repo",
        branch="main",
        status=RepositoryStatus.READY,
        chunk_count=42,
        indexed_at=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def mcp_handlers(registry):
    return MCPHandlers(
        registry=registry,
        indexing_service=FakeIndexingService(registry),
        retrieval_service=FakeRetrievalService(registry),
    )


class TestMCPHandlersSingleton:
    def test_get_mcp_handlers_returns_same_instance(self):
        import coderag.mcp.handlers as handlers_module

        handlers_module._mcp_handlers = None
        handler1 = get_mcp_handlers()
        handler2 = get_mcp_handlers()

        assert handler1 is handler2


class TestListRepositories:
    @pytest.mark.asyncio
    async def test_list_repositories_empty(self, mcp_handlers):
        result = await mcp_handlers.list_repositories()

        assert result["count"] == 0
        assert result["repositories"] == []

    @pytest.mark.asyncio
    async def test_list_repositories_with_repos(self, mcp_handlers, registry, sample_repository):
        registry.add(sample_repository)

        result = await mcp_handlers.list_repositories()

        assert result["count"] == 1
        assert result["repositories"][0]["id"] == sample_repository.id
        assert result["repositories"][0]["name"] == sample_repository.full_name


class TestGetRepositoryInfo:
    @pytest.mark.asyncio
    async def test_get_repository_info_found(self, mcp_handlers, registry, sample_repository):
        registry.add(sample_repository)

        result = await mcp_handlers.get_repository_info(sample_repository.id)

        assert result["id"] == sample_repository.id
        assert result["full_name"] == sample_repository.full_name
        assert result["indexed_files"] == ["src/app.py"]

    @pytest.mark.asyncio
    async def test_get_repository_info_partial_id(self, mcp_handlers, registry, sample_repository):
        registry.add(sample_repository)

        result = await mcp_handlers.get_repository_info("test-repo")

        assert result["id"] == sample_repository.id

    @pytest.mark.asyncio
    async def test_get_repository_info_not_found(self, mcp_handlers):
        result = await mcp_handlers.get_repository_info("missing")

        assert "error" in result


class TestDeleteRepository:
    @pytest.mark.asyncio
    async def test_delete_repository_success(self, mcp_handlers, registry, sample_repository):
        registry.add(sample_repository)

        result = await mcp_handlers.delete_repository(sample_repository.id)

        assert result["success"] is True
        assert result["chunks_deleted"] == 3
        assert registry.get(sample_repository.id) is None

    @pytest.mark.asyncio
    async def test_delete_repository_not_found(self, mcp_handlers):
        result = await mcp_handlers.delete_repository("missing")

        assert result["success"] is False
        assert "not found" in result["error"]


class TestIndexRepository:
    @pytest.mark.asyncio
    async def test_index_repository_success_shape(self, mcp_handlers):
        result = await mcp_handlers.index_repository("https://github.com/owner/repo", branch="main")

        assert result == {
            "success": True,
            "repo_id": "indexed-repo",
            "name": "owner/repo",
            "branch": "main",
            "files_processed": 2,
            "chunks_indexed": 3,
        }


class TestQueryCode:
    @pytest.mark.asyncio
    async def test_query_code_repo_not_found(self, mcp_handlers):
        result = await mcp_handlers.query_code("missing", "What does this do?")

        assert result["grounded"] is False
        assert result["answer"] == ""
        assert "error" in result

    @pytest.mark.asyncio
    async def test_query_code_repo_not_ready(self, mcp_handlers, registry, sample_repository):
        sample_repository.status = RepositoryStatus.INDEXING
        registry.add(sample_repository)

        result = await mcp_handlers.query_code(sample_repository.id, "What does this do?")

        assert result["grounded"] is False
        assert "not ready" in result["error"]

    @pytest.mark.asyncio
    async def test_query_code_success_shape(self, mcp_handlers, registry, sample_repository):
        registry.add(sample_repository)

        result = await mcp_handlers.query_code(sample_repository.id, "What does this do?", top_k=3)

        assert result["answer"] == "Test answer"
        assert result["citations"] == []
        assert result["evidence"] == []
        assert result["grounded"] is False



class TestSearchCode:
    @pytest.mark.asyncio
    async def test_search_code_repo_not_found(self, mcp_handlers):
        result = await mcp_handlers.search_code("missing", "function")

        assert result["results"] == []
        assert "error" in result

    @pytest.mark.asyncio
    async def test_search_code_success_shape(self, mcp_handlers, registry, sample_repository):
        registry.add(sample_repository)

        result = await mcp_handlers.search_code(sample_repository.id, "function", top_k=1)

        assert result["count"] == 1
        assert result["results"][0] == {
            "file_path": "src/app.py",
            "start_line": 1,
            "end_line": 3,
            "chunk_type": "function",
            "content": "def app(): pass",
            "relevance_score": 0.876,
        }



class TestUpdateRepository:
    @pytest.mark.asyncio
    async def test_update_repository_not_found(self, mcp_handlers):
        result = await mcp_handlers.update_repository("missing")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_update_repository_no_previous_commit(self, mcp_handlers, registry, sample_repository):
        sample_repository.last_commit = None
        registry.add(sample_repository)

        result = await mcp_handlers.update_repository(sample_repository.id)

        assert result["success"] is False
        assert "No previous indexing" in result["error"]

    @pytest.mark.asyncio
    async def test_update_repository_already_up_to_date_shape(self, mcp_handlers, registry, sample_repository):
        sample_repository.last_commit = "abc123"
        registry.add(sample_repository)

        result = await mcp_handlers.update_repository(sample_repository.id)

        assert result == {
            "success": True,
            "message": "Repository is already up to date",
            "files_changed": 0,
            "chunks_added": 0,
            "chunks_deleted": 0,
        }
