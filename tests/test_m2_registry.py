"""Milestone 2 SQLite registry and durable job-state tests."""
# mypy: disable-error-code="no-untyped-def,type-arg,var-annotated,attr-defined,assignment,union-attr,arg-type"

import json
import sqlite3
from datetime import datetime

import pytest

from coderag.models.chunk import Chunk, ChunkMetadata, ChunkType
from coderag.models.repository import Repository, RepositoryStatus
from coderag.services.indexing import IndexingOptions, IndexingService
from coderag.services.registry import RepositoryRegistry, RepositoryRegistryError
from tests.test_m1_services import (
    FakeChunker,
    FakeEmbedder,
    FakeFilter,
    FakeLoader,
    FakeValidator,
    FakeVectorStore,
)


def test_sqlite_registry_persists_repositories_across_restart(tmp_path):
    db_path = tmp_path / "registry.sqlite3"
    registry = RepositoryRegistry(db_path=db_path, repos_file=tmp_path / "repositories.json")
    repo = registry.add(
        Repository(
            id="abcdef123456",
            url="https://github.com/acme/demo",
            branch="main",
            status=RepositoryStatus.READY,
            chunk_count=7,
            indexed_at=datetime(2026, 6, 27, 12, 0, 0),
            last_commit="abc123",
        )
    )

    reopened = RepositoryRegistry(db_path=db_path, repos_file=tmp_path / "repositories.json")

    assert reopened.get("abcdef12") == repo
    assert reopened.get_unique("abcdef12") == repo
    assert reopened.list()[0].last_commit == "abc123"

    removed = reopened.remove("abcdef12")
    assert removed == repo
    assert RepositoryRegistry(db_path=db_path, repos_file=tmp_path / "repositories.json").list() == []


def test_sqlite_registry_migrates_existing_json_once(tmp_path):
    repos_file = tmp_path / "repositories.json"
    db_path = tmp_path / "registry.sqlite3"
    repos_file.write_text(
        json.dumps(
            [
                Repository(
                    id="json-repo",
                    url="https://github.com/acme/json-demo",
                    branch="main",
                    status=RepositoryStatus.READY,
                    chunk_count=2,
                ).to_dict()
            ]
        )
    )

    registry = RepositoryRegistry(repos_file=repos_file, db_path=db_path)

    assert registry.get("json") is not None
    assert registry.get("json").chunk_count == 2
    repos_file.write_text("[]")
    assert RepositoryRegistry(repos_file=repos_file, db_path=db_path).get("json-repo") is not None


def test_sqlite_registry_does_not_resurrect_json_after_delete(tmp_path):
    repos_file = tmp_path / "repositories.json"
    db_path = tmp_path / "registry.sqlite3"
    repos_file.write_text(
        json.dumps(
            [
                Repository(
                    id="json-repo",
                    url="https://github.com/acme/json-demo",
                    branch="main",
                    status=RepositoryStatus.READY,
                ).to_dict()
            ]
        )
    )

    registry = RepositoryRegistry(repos_file=repos_file, db_path=db_path)
    assert registry.get("json-repo") is not None

    assert registry.remove("json-repo") is not None

    reopened = RepositoryRegistry(repos_file=repos_file, db_path=db_path)
    assert reopened.list() == []


def test_sqlite_registry_save_preserves_file_and_chunk_metadata(tmp_path):
    db_path = tmp_path / "registry.sqlite3"
    registry = RepositoryRegistry(db_path=db_path, repos_file=tmp_path / "repositories.json")
    repo = registry.add(Repository(id="repo-meta", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    registry.record_file_metadata(repo.id, "app.py", content_hash="sha256:abc", size_bytes=12, language="python")
    registry.record_chunk_metadata(
        Chunk(
            id="chunk-1",
            repo_id=repo.id,
            content="def app(): pass",
            metadata=ChunkMetadata(
                file_path="app.py",
                start_line=1,
                end_line=1,
                chunk_type=ChunkType.FUNCTION,
                language="python",
                name="app",
            ),
        )
    )

    repo.status = RepositoryStatus.ERROR
    registry.save({repo.id: repo})

    reopened = RepositoryRegistry(db_path=db_path, repos_file=tmp_path / "repositories.json")
    assert reopened.get(repo.id).status is RepositoryStatus.ERROR
    assert reopened.list_files(repo.id)[0].content_hash == "sha256:abc"
    assert reopened.list_chunks(repo.id)[0].name == "app"

def test_sqlite_registry_removes_single_file_and_chunk_metadata(tmp_path):
    registry = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3", repos_file=tmp_path / "repositories.json")
    repo = registry.add(Repository(id="repo-incremental", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    for path in ("deleted.py", "kept.py"):
        registry.record_file_metadata(repo.id, path, size_bytes=12, language="python")
        registry.record_chunk_metadata(
            Chunk(
                id=f"chunk-{path}",
                repo_id=repo.id,
                content="x = 1",
                metadata=ChunkMetadata(file_path=path, start_line=1, end_line=1, chunk_type=ChunkType.TEXT),
            )
        )

    registry.remove_file_metadata(repo.id, "deleted.py")

    reopened = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3", repos_file=tmp_path / "repositories.json")
    assert [record.path for record in reopened.list_files(repo.id)] == ["kept.py"]
    assert [record.file_path for record in reopened.list_chunks(repo.id)] == ["kept.py"]


def test_sqlite_registry_corrupt_json_migration_fails_closed(tmp_path):
    repos_file = tmp_path / "repositories.json"
    repos_file.write_text("{not-json")
    registry = RepositoryRegistry(repos_file=repos_file, db_path=tmp_path / "registry.sqlite3")

    with pytest.raises(RepositoryRegistryError):
        registry.list()

    assert repos_file.read_text() == "{not-json"
    with sqlite3.connect(tmp_path / "registry.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM repositories").fetchone()[0] == 0


def test_sqlite_registry_tracks_jobs_files_chunks_and_metadata(tmp_path):
    registry = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3", repos_file=tmp_path / "repositories.json")
    repo = registry.add(Repository(id="repo-ready", url="https://github.com/acme/demo", status=RepositoryStatus.READY))

    job = registry.begin_job(repo.id, "index")
    registry.record_file_metadata(repo.id, "app.py", content_hash="sha256:abc", size_bytes=12, language="python")
    registry.finish_job(job.id, "succeeded", files_processed=1, chunks_indexed=1)

    reopened = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3", repos_file=tmp_path / "repositories.json")
    [saved_job] = reopened.list_jobs(repo.id)
    [saved_file] = reopened.list_files(repo.id)

    assert saved_job.operation == "index"
    assert saved_job.status == "succeeded"
    assert saved_job.files_processed == 1
    assert saved_file.path == "app.py"
    assert saved_file.content_hash == "sha256:abc"


def test_indexing_service_persists_success_and_failure_jobs(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "app.py").write_text("print('hello')\n")
    registry = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3", repos_file=tmp_path / "repositories.json")
    service = IndexingService(
        registry=registry,
        validator=FakeValidator(),
        loader=FakeLoader(repo_path),
        file_filter_factory=FakeFilter,
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vectorstore=FakeVectorStore(),
        commit_resolver=lambda _: "abc123",
    )

    result = service.index_repository_sync("https://github.com/acme/demo", IndexingOptions(branch="main"))

    reopened = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3", repos_file=tmp_path / "repositories.json")
    [success_job] = reopened.list_jobs(result.repo.id)
    assert success_job.status == "succeeded"
    assert success_job.files_processed == 1
    assert success_job.chunks_indexed == 1
    assert reopened.list_files(result.repo.id)[0].path == "app.py"
    assert reopened.list_chunks(result.repo.id)[0].file_path == "app.py"

    class ExplodingLoader(FakeLoader):
        def clone_repository(self, _repo_info, _branch):
            raise RuntimeError("clone failed")

    failing_registry = RepositoryRegistry(db_path=tmp_path / "failing.sqlite3", repos_file=tmp_path / "failing.json")
    failing_service = IndexingService(
        registry=failing_registry,
        validator=FakeValidator(),
        loader=ExplodingLoader(tmp_path / "missing"),
        file_filter_factory=FakeFilter,
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vectorstore=FakeVectorStore(),
    )

    with pytest.raises(RuntimeError, match="clone failed"):
        failing_service.index_repository_sync("https://github.com/acme/demo")

    [failed_repo] = failing_registry.list()
    [failed_job] = failing_registry.list_jobs(failed_repo.id)
    assert failed_repo.status is RepositoryStatus.ERROR
    assert failed_job.status == "failed"
    assert failed_job.error == "clone failed"


def test_delete_repository_persists_delete_job_after_repo_removal(tmp_path):
    registry = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3", repos_file=tmp_path / "repositories.json")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo = registry.add(Repository(id="repo-delete", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    vectorstore = FakeVectorStore()
    service = IndexingService(
        registry=registry,
        loader=FakeLoader(repo_path),
        vectorstore=vectorstore,
    )

    result = service.delete_repository(repo.id)

    assert result is not None
    assert registry.get(repo.id) is None
    [delete_job] = registry.list_jobs(repo.id)
    assert delete_job.operation == "delete"
    assert delete_job.status == "succeeded"


def test_update_repository_persists_failed_update_job(tmp_path):
    registry = RepositoryRegistry(db_path=tmp_path / "registry.sqlite3", repos_file=tmp_path / "repositories.json")
    repo = registry.add(Repository(id="repo-update", url="https://github.com/acme/demo", status=RepositoryStatus.READY))
    service = IndexingService(registry=registry)

    with pytest.raises(ValueError, match="No previous indexing"):
        service.update_repository(repo.id)

    [update_job] = registry.list_jobs(repo.id)
    assert update_job.operation == "update"
    assert update_job.status == "failed"
    assert update_job.error == "No previous indexing found. Please re-index the full repository."
