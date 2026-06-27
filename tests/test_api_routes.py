"""REST API route tests for the Milestone 1 service seam."""
# mypy: disable-error-code="no-untyped-def,type-arg,var-annotated,assignment,arg-type,call-arg,attr-defined,union-attr"

from dataclasses import dataclass

import pytest

from coderag.api import routes
from coderag.api.schemas import IndexRepositoryRequest, QueryRequest
from coderag.models.repository import Repository, RepositoryStatus
from coderag.services.indexing import RepositoryDeleteResult
from coderag.services.retrieval import QueryServiceResult


class FakeBackgroundTasks:
    def __init__(self) -> None:
        self.tasks = []

    def add_task(self, func, *args, **kwargs) -> None:
        self.tasks.append((func, args, kwargs))


class FakeIndexingService:
    def __init__(self) -> None:
        self.created_options = None
        self.deleted_repo_id = None
        self.repo = Repository(
            id="repo-123456",
            url="https://github.com/acme/demo",
            branch="main",
            status=RepositoryStatus.PENDING,
        )

    def create_repository_record(self, url, options):
        self.created_options = options
        self.repo.url = url
        self.repo.branch = options.branch or "main"
        return self.repo

    def delete_repository(self, repo_id):
        self.deleted_repo_id = repo_id
        if repo_id == self.repo.id[:8]:
            return RepositoryDeleteResult(repo=self.repo, chunks_deleted=4)
        return None


class FakeRegistry:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def get(self, repo_id: str):
        if self.repo.id.startswith(repo_id):
            return self.repo
        return None

    def list(self):
        return [self.repo]


@dataclass
class FakeGeneratedResponse:
    answer: str = "shared answer"
    citations: list = None
    retrieved_chunks: list = None
    grounded: bool = False
    query_id: str = "query-1"

    def __post_init__(self) -> None:
        self.citations = [] if self.citations is None else self.citations
        self.retrieved_chunks = [] if self.retrieved_chunks is None else self.retrieved_chunks


class FakeRetrievalService:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.calls = []

    def query_code(self, repo_id: str, question: str, top_k: int):
        self.calls.append((repo_id, question, top_k))
        if repo_id == "missing":
            return QueryServiceResult(error="Repository not found: missing")
        if self.repo.status is not RepositoryStatus.READY:
            return QueryServiceResult(repo=self.repo, error=f"Repository not ready: status is {self.repo.status.value}")
        return QueryServiceResult(repo=self.repo, response=FakeGeneratedResponse())


@pytest.mark.asyncio
async def test_index_repository_route_creates_record_through_indexing_service(monkeypatch):
    indexing = FakeIndexingService()
    monkeypatch.setattr(routes, "indexing_service", indexing)
    background = FakeBackgroundTasks()

    response = await routes.index_repository(
        IndexRepositoryRequest(url="https://github.com/acme/demo", branch="main", include_patterns=["*.py"]),
        background,
    )

    assert response.repo_id == "repo-123456"
    assert response.status == "pending"
    assert indexing.created_options.include_patterns == ["*.py"]
    assert background.tasks[0][0] is routes.index_repository_task
    assert background.tasks[0][1][:3] == ("https://github.com/acme/demo", "repo-123456", "main")


@pytest.mark.asyncio
async def test_query_route_uses_retrieval_service_and_preserves_response(monkeypatch):
    repo = Repository(id="repo-ready", url="https://github.com/acme/demo", status=RepositoryStatus.READY)
    retrieval = FakeRetrievalService(repo)
    monkeypatch.setattr(routes, "retrieval_service", retrieval)

    response = await routes.query_repository(QueryRequest(repo_id="repo-ready", question="How?", top_k=3))

    assert response.answer == "shared answer"
    assert response.query_id == "query-1"
    assert retrieval.calls == [("repo-ready", "How?", 3)]


@pytest.mark.asyncio
async def test_query_route_maps_missing_and_not_ready_errors(monkeypatch):
    repo = Repository(id="repo-pending", url="https://github.com/acme/demo", status=RepositoryStatus.INDEXING)
    retrieval = FakeRetrievalService(repo)
    monkeypatch.setattr(routes, "retrieval_service", retrieval)

    with pytest.raises(routes.HTTPException) as missing:
        await routes.query_repository(QueryRequest(repo_id="missing", question="How?"))
    with pytest.raises(routes.HTTPException) as not_ready:
        await routes.query_repository(QueryRequest(repo_id=repo.id, question="How?"))

    assert missing.value.status_code == 404
    assert not_ready.value.status_code == 400
    assert not_ready.value.detail == "Repository not ready (status: indexing)"


@pytest.mark.asyncio
async def test_delete_route_maps_missing_repository(monkeypatch):
    indexing = FakeIndexingService()
    monkeypatch.setattr(routes, "indexing_service", indexing)

    with pytest.raises(routes.HTTPException) as exc_info:
        await routes.delete_repository("missing")

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_route_repository_list_get_and_delete_share_registry_and_indexing_service(monkeypatch):
    indexing = FakeIndexingService()
    registry = FakeRegistry(indexing.repo)
    monkeypatch.setattr(routes, "indexing_service", indexing)
    monkeypatch.setattr(routes, "registry", registry)

    listed = await routes.list_repositories()
    fetched = await routes.get_repository("repo-123")
    deleted = await routes.delete_repository("repo-123")

    assert listed.repositories[0].id == "repo-123456"
    assert fetched.id == "repo-123456"
    assert deleted == {"message": "Repository acme/demo deleted"}
    assert indexing.deleted_repo_id == "repo-123"


@pytest.mark.asyncio
async def test_api_and_mcp_handlers_share_registry_backed_metadata_file(tmp_path, monkeypatch):
    from coderag.mcp.handlers import MCPHandlers
    from coderag.services.registry import RepositoryRegistry

    repos_file = tmp_path / "repositories.json"
    api_registry = RepositoryRegistry(repos_file)
    repo = api_registry.add(
        Repository(
            id="shared-repo",
            url="https://github.com/acme/demo",
            branch="main",
            status=RepositoryStatus.READY,
            chunk_count=7,
        )
    )
    monkeypatch.setattr(routes, "registry", api_registry)

    mcp = MCPHandlers(
        registry=RepositoryRegistry(repos_file),
        indexing_service=FakeIndexingService(),
        retrieval_service=FakeRetrievalService(repo),
    )

    api_list = await routes.list_repositories()
    mcp_list = await mcp.list_repositories()

    assert api_list.repositories[0].id == "shared-repo"
    assert mcp_list["repositories"][0]["id"] == "shared-repo"
    assert mcp_list["repositories"][0]["chunk_count"] == 7
