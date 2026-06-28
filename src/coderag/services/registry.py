"""SQLite-backed repository metadata registry service."""

from __future__ import annotations

import json
import sqlite3
from builtins import list as list_type
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from coderag.config import get_settings
from coderag.logging import get_logger
from coderag.models.chunk import Chunk
from coderag.models.repository import Repository

logger = get_logger(__name__)
SCHEMA_VERSION = 1


class RepositoryRegistryError(RuntimeError):
    """Raised when existing repository metadata cannot be safely read."""


@dataclass(frozen=True)
class RegistryJob:
    """Durable indexing/update/delete job state."""

    id: str
    repo_id: str | None
    operation: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    files_processed: int | None = None
    chunks_indexed: int | None = None


@dataclass(frozen=True)
class RepositoryFileRecord:
    """Indexed file metadata stored by repository."""

    repo_id: str
    path: str
    content_hash: str | None = None
    size_bytes: int | None = None
    language: str | None = None
    indexed_at: datetime | None = None


@dataclass(frozen=True)
class RepositoryChunkRecord:
    """Indexed chunk metadata stored by repository."""

    id: str
    repo_id: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str | None = None
    content_hash: str | None = None
    metadata_json: str | None = None


class RepositoryRegistry:
    """SQLite-backed repository metadata registry.

    SQLite is the canonical Milestone 2 store. A pre-1.0 alpha
    ``repositories.json`` file is read once as a migration source when the
    SQLite repository table is empty; corrupt JSON fails closed and is never
    overwritten.
    """

    def __init__(self, repos_file: Path | None = None, db_path: Path | None = None) -> None:
        settings = get_settings()
        if db_path is None and repos_file is not None and repos_file.suffix in {".db", ".sqlite", ".sqlite3"}:
            db_path = repos_file
            repos_file = repos_file.with_name("repositories.json")

        self.repos_file = repos_file or settings.data_dir / "repositories.json"
        self.db_path = db_path or self.repos_file.with_name("registry.sqlite3")
        self._ensure_schema()

    def load(self) -> dict[str, Repository]:
        """Load all repositories keyed by id."""
        self._migrate_json_if_needed()
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM repositories ORDER BY rowid").fetchall()
        return {row["id"]: self._repo_from_row(row) for row in rows}

    def save(self, repositories: dict[str, Repository]) -> None:
        """Persist repositories into SQLite without deleting metadata for retained repos."""
        self._migrate_json_if_needed()
        with self._connect() as conn:
            incoming_ids = set(repositories)
            existing_ids = {
                row["id"]
                for row in conn.execute("SELECT id FROM repositories").fetchall()
            }
            for repo in repositories.values():
                self._upsert_repo(conn, repo)
            for repo_id in existing_ids - incoming_ids:
                conn.execute("DELETE FROM repositories WHERE id = ?", (repo_id,))

    def list(self) -> list[Repository]:
        """Return repositories from the latest registry snapshot."""
        return list(self.load().values())

    def resolve_id(self, partial_id: str) -> str | None:
        """Resolve a full or prefix repository id using first-match compatibility."""
        repositories = self.load()
        if partial_id in repositories:
            return partial_id

        for repo_id in repositories:
            if repo_id.startswith(partial_id):
                return repo_id
        return None

    def resolve_unique_id(self, partial_id: str) -> str | None:
        """Resolve a full or unambiguous prefix repository id."""
        repositories = self.load()
        if partial_id in repositories:
            return partial_id

        matches = [repo_id for repo_id in repositories if repo_id.startswith(partial_id)]
        if len(matches) == 1:
            return matches[0]
        return None

    def get(self, repo_id: str) -> Repository | None:
        """Get a repository by full id or first prefix match."""
        repositories = self.load()
        resolved_id = repo_id if repo_id in repositories else self.resolve_id(repo_id)
        if resolved_id is None:
            return None
        return repositories.get(resolved_id)

    def get_unique(self, repo_id: str) -> Repository | None:
        """Get a repository by full id or unambiguous prefix."""
        repositories = self.load()
        resolved_id = repo_id if repo_id in repositories else self.resolve_unique_id(repo_id)
        if resolved_id is None:
            return None
        return repositories.get(resolved_id)

    def add(self, repo: Repository) -> Repository:
        """Add a repository and persist it."""
        self._migrate_json_if_needed()
        with self._connect() as conn:
            self._upsert_repo(conn, repo)
        return repo

    def update(self, repo: Repository) -> Repository:
        """Update a repository and persist it."""
        self._migrate_json_if_needed()
        with self._connect() as conn:
            self._upsert_repo(conn, repo)
        return repo

    def remove(self, repo_id: str) -> Repository | None:
        """Remove a repository by full id or prefix and persist the registry."""
        repositories = self.load()
        resolved_id = repo_id if repo_id in repositories else self.resolve_id(repo_id)
        if resolved_id is None:
            return None

        repo = repositories[resolved_id]
        with self._connect() as conn:
            conn.execute("DELETE FROM repository_chunks WHERE repo_id = ?", (resolved_id,))
            conn.execute("DELETE FROM repository_files WHERE repo_id = ?", (resolved_id,))
            conn.execute("DELETE FROM repositories WHERE id = ?", (resolved_id,))
        return repo

    def begin_job(self, repo_id: str | None, operation: str) -> RegistryJob:
        """Create a durable running job record."""
        job = RegistryJob(
            id=str(uuid4()),
            repo_id=repo_id,
            operation=operation,
            status="running",
            started_at=datetime.now(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO index_jobs (
                    id, repo_id, operation, status, started_at, finished_at,
                    error, files_processed, chunks_indexed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._job_values(job),
            )
        return job

    def finish_job(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
        files_processed: int | None = None,
        chunks_indexed: int | None = None,
    ) -> RegistryJob:
        """Mark a durable job complete or failed."""
        finished_at = datetime.now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, finished_at = ?, error = ?, files_processed = ?, chunks_indexed = ?
                WHERE id = ?
                """,
                (
                    status,
                    finished_at.isoformat(),
                    error,
                    files_processed,
                    chunks_indexed,
                    job_id,
                ),
            )
        job = self.get_job(job_id)
        if job is None:
            raise RepositoryRegistryError(f"Job not found: {job_id}")
        return job

    def get_job(self, job_id: str) -> RegistryJob | None:
        """Return one durable job record."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM index_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_from_row(row) if row else None

    def list_jobs(self, repo_id: str | None = None) -> list_type[RegistryJob]:
        """Return durable job records, newest first."""
        with self._connect() as conn:
            if repo_id is None:
                rows = conn.execute("SELECT * FROM index_jobs ORDER BY started_at DESC, rowid DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM index_jobs WHERE repo_id = ? ORDER BY started_at DESC, rowid DESC",
                    (repo_id,),
                ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def clear_file_metadata(self, repo_id: str) -> None:
        """Delete indexed file metadata for a repository."""
        with self._connect() as conn:
            conn.execute("DELETE FROM repository_files WHERE repo_id = ?", (repo_id,))

    def remove_file_metadata(self, repo_id: str, path: str) -> None:
        """Delete one file's durable metadata and chunk metadata."""
        with self._connect() as conn:
            conn.execute("DELETE FROM repository_chunks WHERE repo_id = ? AND file_path = ?", (repo_id, path))
            conn.execute("DELETE FROM repository_files WHERE repo_id = ? AND path = ?", (repo_id, path))
            conn.execute(
                "UPDATE repositories SET file_count = (SELECT COUNT(*) FROM repository_files WHERE repo_id = ?) WHERE id = ?",
                (repo_id, repo_id),
            )

    def record_file_metadata(
        self,
        repo_id: str,
        path: str,
        *,
        content_hash: str | None = None,
        size_bytes: int | None = None,
        language: str | None = None,
    ) -> RepositoryFileRecord:
        """Persist indexed file metadata."""
        record = RepositoryFileRecord(
            repo_id=repo_id,
            path=path,
            content_hash=content_hash,
            size_bytes=size_bytes,
            language=language,
            indexed_at=datetime.now(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO repository_files (repo_id, path, content_hash, size_bytes, language, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id, path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    size_bytes = excluded.size_bytes,
                    language = excluded.language,
                    indexed_at = excluded.indexed_at
                """,
                (
                    record.repo_id,
                    record.path,
                    record.content_hash,
                    record.size_bytes,
                    record.language,
                    record.indexed_at.isoformat() if record.indexed_at else None,
                ),
            )
            conn.execute(
                "UPDATE repositories SET file_count = (SELECT COUNT(*) FROM repository_files WHERE repo_id = ?) WHERE id = ?",
                (repo_id, repo_id),
            )
        return record

    def list_files(self, repo_id: str) -> list_type[RepositoryFileRecord]:
        """Return indexed file metadata for a repository."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM repository_files WHERE repo_id = ? ORDER BY path", (repo_id,)).fetchall()
        return [self._file_from_row(row) for row in rows]

    def clear_chunk_metadata(self, repo_id: str) -> None:
        """Delete indexed chunk metadata for a repository."""
        with self._connect() as conn:
            conn.execute("DELETE FROM repository_chunks WHERE repo_id = ?", (repo_id,))

    def record_chunk_metadata(self, chunk: Chunk) -> RepositoryChunkRecord:
        """Persist indexed chunk metadata."""
        record = RepositoryChunkRecord(
            id=chunk.id,
            repo_id=chunk.repo_id,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type.value,
            name=chunk.name,
            metadata_json=json.dumps(chunk.metadata.__dict__, default=str, sort_keys=True),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO repository_chunks (
                    id, repo_id, file_path, start_line, end_line, chunk_type,
                    name, content_hash, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    repo_id = excluded.repo_id,
                    file_path = excluded.file_path,
                    start_line = excluded.start_line,
                    end_line = excluded.end_line,
                    chunk_type = excluded.chunk_type,
                    name = excluded.name,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record.id,
                    record.repo_id,
                    record.file_path,
                    record.start_line,
                    record.end_line,
                    record.chunk_type,
                    record.name,
                    record.content_hash,
                    record.metadata_json,
                ),
            )
        return record

    def list_chunks(self, repo_id: str) -> list_type[RepositoryChunkRecord]:
        """Return indexed chunk metadata for a repository."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM repository_chunks WHERE repo_id = ? ORDER BY file_path, start_line, end_line",
                (repo_id,),
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS registry_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS repositories (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    clone_path TEXT,
                    indexed_at TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    last_commit TEXT,
                    file_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    index_metadata TEXT NOT NULL DEFAULT '{}',
                    provider_metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS repository_files (
                    repo_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    content_hash TEXT,
                    size_bytes INTEGER,
                    language TEXT,
                    indexed_at TEXT,
                    PRIMARY KEY (repo_id, path),
                    FOREIGN KEY (repo_id) REFERENCES repositories(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS repository_chunks (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    chunk_type TEXT NOT NULL,
                    name TEXT,
                    content_hash TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (repo_id) REFERENCES repositories(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS index_jobs (
                    id TEXT PRIMARY KEY,
                    repo_id TEXT,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    error TEXT,
                    files_processed INTEGER,
                    chunks_indexed INTEGER
                );
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now().isoformat()),
            )

    def _migrate_json_if_needed(self) -> None:
        if not self.repos_file.exists():
            return
        with self._connect() as conn:
            migrated = conn.execute(
                "SELECT value FROM registry_metadata WHERE key = 'repositories_json_migrated_at'"
            ).fetchone()
            if migrated:
                return

            existing = conn.execute("SELECT COUNT(*) FROM repositories").fetchone()[0]
            if existing:
                self._mark_json_migrated(conn)
                return

            try:
                data = json.loads(self.repos_file.read_text())
                repos = [Repository.from_dict(item) for item in data]
            except Exception as exc:
                logger.error("Failed to migrate repositories", path=str(self.repos_file), error=str(exc))
                raise RepositoryRegistryError(f"Failed to migrate repositories from {self.repos_file}") from exc
            for repo in repos:
                self._upsert_repo(conn, repo)
            self._mark_json_migrated(conn)

    def _mark_json_migrated(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO registry_metadata (key, value) VALUES ('repositories_json_migrated_at', ?)",
            (datetime.now().isoformat(),),
        )
    def _upsert_repo(self, conn: sqlite3.Connection, repo: Repository) -> None:
        conn.execute(
            """
            INSERT INTO repositories (
                id, url, branch, clone_path, indexed_at, status, error_message,
                last_commit, file_count, chunk_count, index_metadata, provider_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT file_count FROM repositories WHERE id = ?), 0), ?, COALESCE((SELECT index_metadata FROM repositories WHERE id = ?), '{}'), COALESCE((SELECT provider_metadata FROM repositories WHERE id = ?), '{}'))
            ON CONFLICT(id) DO UPDATE SET
                url = excluded.url,
                branch = excluded.branch,
                clone_path = excluded.clone_path,
                indexed_at = excluded.indexed_at,
                status = excluded.status,
                error_message = excluded.error_message,
                last_commit = excluded.last_commit,
                chunk_count = excluded.chunk_count
            """,
            (
                repo.id,
                repo.url,
                repo.branch,
                str(repo.clone_path) if repo.clone_path else None,
                repo.indexed_at.isoformat() if repo.indexed_at else None,
                repo.status.value,
                repo.error_message,
                repo.last_commit,
                repo.id,
                repo.chunk_count,
                repo.id,
                repo.id,
            ),
        )

    def _repo_from_row(self, row: sqlite3.Row) -> Repository:
        return Repository.from_dict(
            {
                "id": row["id"],
                "url": row["url"],
                "branch": row["branch"],
                "clone_path": row["clone_path"],
                "indexed_at": row["indexed_at"],
                "chunk_count": row["chunk_count"],
                "status": row["status"],
                "error_message": row["error_message"],
                "last_commit": row["last_commit"],
            }
        )

    def _job_values(self, job: RegistryJob) -> tuple[object, ...]:
        return (
            job.id,
            job.repo_id,
            job.operation,
            job.status,
            job.started_at.isoformat(),
            job.finished_at.isoformat() if job.finished_at else None,
            job.error,
            job.files_processed,
            job.chunks_indexed,
        )

    def _job_from_row(self, row: sqlite3.Row) -> RegistryJob:
        return RegistryJob(
            id=row["id"],
            repo_id=row["repo_id"],
            operation=row["operation"],
            status=row["status"],
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            error=row["error"],
            files_processed=row["files_processed"],
            chunks_indexed=row["chunks_indexed"],
        )

    def _file_from_row(self, row: sqlite3.Row) -> RepositoryFileRecord:
        return RepositoryFileRecord(
            repo_id=row["repo_id"],
            path=row["path"],
            content_hash=row["content_hash"],
            size_bytes=row["size_bytes"],
            language=row["language"],
            indexed_at=datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else None,
        )

    def _chunk_from_row(self, row: sqlite3.Row) -> RepositoryChunkRecord:
        return RepositoryChunkRecord(
            id=row["id"],
            repo_id=row["repo_id"],
            file_path=row["file_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            chunk_type=row["chunk_type"],
            name=row["name"],
            content_hash=row["content_hash"],
            metadata_json=row["metadata_json"],
        )
