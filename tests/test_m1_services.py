"""Milestone 1 service seam tests."""
# mypy: disable-error-code="no-untyped-def,type-arg,var-annotated,attr-defined,assignment,union-attr,arg-type"

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from coderag import config as config_module
from coderag.config import ModelSettings
from coderag.models.chunk import Chunk, ChunkMetadata, ChunkType
from coderag.models.repository import Repository, RepositoryStatus
from coderag.services.indexing import IndexingOptions, IndexingService
from coderag.services.providers import ProviderConfigService
from coderag.services.registry import RepositoryRegistry, RepositoryRegistryError
from coderag.services.retrieval import RetrievalService


@dataclass
class FakeRepoInfo:
    url: str = "https://github.com/acme/demo"
    owner: str = "acme"
    name: str = "demo"
    branch: str = "main"

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class FakeValidator:
    def parse_url(self, url: str) -> FakeRepoInfo:
        return FakeRepoInfo(url=url)


class FakeLoader:
    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path
        self.deleted: list[str] = []

    def clone_repository(self, _repo_info: FakeRepoInfo, _branch: str) -> Path:
        return self.repo_path

    def delete_cache(self, repo_info: object) -> None:
        self.deleted.append(f"{repo_info.owner}/{repo_info.name}")


class FakeFilter:
    def __init__(self, include_patterns=None, exclude_patterns=None) -> None:
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns

    def filter_files(self, repo_path: Path):
        yield repo_path / "app.py"

    def should_include(self, _file_path: Path, _repo_path: Path) -> bool:
        return True


class FakeChunker:
    def chunk_document(self, document):
        yield Chunk(
            content=document.content,
            repo_id=document.repo_id,
            metadata=ChunkMetadata(
                file_path=document.file_path,
                start_line=1,
                end_line=1,
                chunk_type=ChunkType.TEXT,
                language="python",
                name=None,
            ),
        )


class FakeEmbedder:
    def __init__(self) -> None:
        self.unloaded = False

    def embed_chunks(self, chunks: list[Chunk], show_progress: bool = False) -> list[Chunk]:
        _ = show_progress
        for chunk in chunks:
            chunk.embedding = [0.1, 0.2]
        return chunks

    def generate_embedding(self, text: str, is_query: bool = False) -> list[float]:
        _ = (text, is_query)
        return [0.1, 0.2]

    def unload(self) -> None:
        self.unloaded = True


class FakeVectorStore:
    def __init__(self) -> None:
        self.added: list[Chunk] = []
        self.deleted_repos: list[str] = []
        self.queries: list[dict] = []

    def add_chunks(self, chunks: list[Chunk]) -> int:
        self.added.extend(chunks)
        return len(chunks)

    def delete_repo_chunks(self, repo_id: str) -> int:
        self.deleted_repos.append(repo_id)
        return 3

    def delete_file_chunks(self, _repo_id: str, _file_path: str) -> int:
        return 1

    def get_repo_chunk_count(self, repo_id: str) -> int:
        return len([chunk for chunk in self.added if chunk.repo_id == repo_id])

    def get_indexed_files(self, repo_id: str) -> set[str]:
        return {chunk.file_path for chunk in self.added if chunk.repo_id == repo_id}

    def query(self, query_embedding, repo_id: str, top_k: int = 5, similarity_threshold: float = 0.0):
        _ = query_embedding
        self.queries.append({"repo_id": repo_id, "top_k": top_k, "threshold": similarity_threshold})
        return [(chunk, 0.87) for chunk in self.added if chunk.repo_id == repo_id][:top_k]


class FakeResponse:
    answer = "shared answer"
    citations = []
    retrieved_chunks = []
    grounded = False
    query_id = "query-1"


class FakeGenerator:
    def __init__(self) -> None:
        self.queries = []

    def generate(self, query):
        self.queries.append(query)
        return FakeResponse()


def test_repository_registry_preserves_partial_ids_and_persists_metadata(tmp_path):
    registry = RepositoryRegistry(tmp_path / "repositories.json")
    repo = Repository(
        id="abcdef123456",
        url="https://github.com/acme/demo",
        branch="main",
        status=RepositoryStatus.READY,
        chunk_count=7,
    )

    registry.add(repo)

    assert registry.resolve_id("abcdef12") == repo.id
    assert registry.get("abcdef12") == repo

    reloaded = RepositoryRegistry(tmp_path / "repositories.json")
    assert reloaded.list()[0].id == repo.id

    removed = reloaded.remove("abcdef12")
    assert removed == repo
    assert reloaded.list() == []


def test_repository_registry_corrupt_json_fails_closed_without_overwriting(tmp_path):
    repos_file = tmp_path / "repositories.json"
    repos_file.write_text("{not-json")
    registry = RepositoryRegistry(repos_file)

    with pytest.raises(RepositoryRegistryError):
        registry.list()
    assert repos_file.read_text() == "{not-json"


def test_indexing_service_full_index_updates_shared_registry_and_vector_store(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "app.py").write_text("print('hello')\n")
    registry = RepositoryRegistry(tmp_path / "repositories.json")
    vectorstore = FakeVectorStore()
    embedder = FakeEmbedder()
    service = IndexingService(
        registry=registry,
        validator=FakeValidator(),
        loader=FakeLoader(repo_path),
        file_filter_factory=FakeFilter,
        chunker=FakeChunker(),
        embedder=embedder,
        vectorstore=vectorstore,
        commit_resolver=lambda _: "abc123",
    )

    result = service.index_repository_sync("https://github.com/acme/demo", IndexingOptions(branch="main"))

    assert result.repo.status is RepositoryStatus.READY
    assert result.files_processed == 1
    assert result.chunks_indexed == 1
    assert result.repo.last_commit == "abc123"
    assert registry.get(result.repo.id).chunk_count == 1
    assert vectorstore.deleted_repos == [result.repo.id]
    assert vectorstore.added[0].repo_id == result.repo.id
    assert embedder.unloaded is True


def test_indexing_service_persists_error_status_on_failure(tmp_path):
    class ExplodingLoader(FakeLoader):
        def clone_repository(self, _repo_info: FakeRepoInfo, _branch: str) -> Path:
            raise RuntimeError("clone failed")

    registry = RepositoryRegistry(tmp_path / "repositories.json")
    service = IndexingService(
        registry=registry,
        validator=FakeValidator(),
        loader=ExplodingLoader(tmp_path / "missing"),
        file_filter_factory=FakeFilter,
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vectorstore=FakeVectorStore(),
    )

    with pytest.raises(RuntimeError, match="clone failed"):
        service.index_repository_sync("https://github.com/acme/demo")

    [repo] = registry.list()
    assert repo.status is RepositoryStatus.ERROR
    assert repo.error_message == "clone failed"


def test_retrieval_service_checks_repository_readiness_before_generation(tmp_path):
    registry = RepositoryRegistry(tmp_path / "repositories.json")
    repo = registry.add(Repository(id="repo-ready", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    generator = FakeGenerator()
    service = RetrievalService(registry=registry, generator_factory=lambda: generator)

    result = service.query_code(repo.id, "How?", top_k=3)

    assert result.error is None
    assert result.response.answer == "shared answer"
    assert generator.queries[0].repo_id == repo.id
    assert generator.queries[0].top_k == 3


def test_retrieval_service_returns_typed_errors_for_missing_and_not_ready_repos(tmp_path):
    registry = RepositoryRegistry(tmp_path / "repositories.json")
    pending = registry.add(Repository(id="pending-repo", url="https://github.com/acme/demo", status=RepositoryStatus.INDEXING))
    service = RetrievalService(registry=registry, generator_factory=lambda: FakeGenerator())

    missing = service.query_code("missing", "How?")
    not_ready = service.query_code(pending.id, "How?")
    assert missing.error == "Repository not found: missing"
    assert not_ready.error == "Repository not ready: status is indexing"


def test_retrieval_service_blocks_generation_for_retrieval_only_profile(tmp_path):
    registry = RepositoryRegistry(tmp_path / "repositories.json")
    repo = registry.add(Repository(id="repo-ready", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    generator = FakeGenerator()
    service = RetrievalService(
        registry=registry,
        generator_factory=lambda: generator,
        provider_config=ProviderConfigService(ModelSettings(profile="retrieval-only", llm_provider="groq")),
    )

    result = service.query_code(repo.id, "How?")

    assert result.error == "LLM generation is disabled by the retrieval-only profile"
    assert generator.queries == []


def test_retrieval_service_blocks_private_profile_with_remote_provider(tmp_path):
    registry = RepositoryRegistry(tmp_path / "repositories.json")
    repo = registry.add(Repository(id="repo-ready", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    service = RetrievalService(
        registry=registry,
        generator_factory=lambda: FakeGenerator(),
        provider_config=ProviderConfigService(ModelSettings(profile="private", llm_provider="groq")),
    )

    result = service.query_code(repo.id, "How?")

    assert result.error == "Private profile requires MODEL_LLM_PROVIDER=local"



def test_retrieval_service_search_code_preserves_vector_results_and_filters(tmp_path):
    registry = RepositoryRegistry(tmp_path / "repositories.json")
    repo = registry.add(Repository(id="repo-ready", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    vectorstore = FakeVectorStore()
    chunk = Chunk(
        content="def hello(): pass",
        repo_id=repo.id,
        metadata=ChunkMetadata(
            file_path="src/app.py",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language="python",
            name="hello",
        ),
    )
    vectorstore.add_chunks([chunk])
    service = RetrievalService(registry=registry, embedder=FakeEmbedder(), vectorstore=vectorstore)

    results = service.search_code(repo.id, "hello", top_k=999, file_filter="src/*.py", chunk_type="function")

    assert results[0][0] == chunk
    assert results[0][1] == 0.87
    assert vectorstore.queries[0] == {"repo_id": repo.id, "top_k": 20, "threshold": 0.3}


def test_provider_config_service_exposes_profiles_without_requiring_local_llm():
    cloud = ProviderConfigService(ModelSettings(profile="cloud", llm_provider="groq"))
    private_remote = ProviderConfigService(ModelSettings(profile="private", llm_provider="groq"))
    retrieval_only = ProviderConfigService(ModelSettings(profile="retrieval-only", llm_provider="groq"))

    assert cloud.profile == "cloud"
    assert cloud.requires_generation_provider is True
    assert cloud.requires_local_llm is False
    assert cloud.uses_cloud_generation is True
    assert private_remote.requires_local_llm is True
    assert private_remote.generation_block_reason() == "Private profile requires MODEL_LLM_PROVIDER=local"
    assert retrieval_only.profile == "retrieval-only"
    assert retrieval_only.requires_generation_provider is False

    with pytest.raises(ValidationError):
        ModelSettings(profile="invalid")


def test_provider_config_service_reads_documented_model_profile_from_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MODEL_PROFILE=retrieval-only\nMODEL_LLM_PROVIDER=groq\n")
    monkeypatch.chdir(tmp_path)
    config_module._settings = None

    try:
        provider_config = ProviderConfigService()

        assert provider_config.profile == "retrieval-only"
        assert provider_config.generation_block_reason() == "LLM generation is disabled by the retrieval-only profile"
    finally:
        config_module._settings = None
