"""Repository metadata registry service."""

import json
from pathlib import Path

from coderag.config import get_settings
from coderag.logging import get_logger
from coderag.models.repository import Repository

logger = get_logger(__name__)


class RepositoryRegistryError(RuntimeError):
    """Raised when existing repository metadata cannot be safely read."""


class RepositoryRegistry:
    """JSON-backed repository metadata registry.

    This preserves the pre-Milestone-1 `repositories.json` file shape while
    centralizing load/save/partial-id semantics for API, MCP, CLI, and UI
    adapters. Milestone 2 can replace this backing store with SQLite behind the
    same interface.
    """

    def __init__(self, repos_file: Path | None = None) -> None:
        settings = get_settings()
        self.repos_file = repos_file or settings.data_dir / "repositories.json"

    def load(self) -> dict[str, Repository]:
        """Load all repositories keyed by id."""
        if not self.repos_file.exists():
            return {}

        try:
            data = json.loads(self.repos_file.read_text())
            return {item["id"]: Repository.from_dict(item) for item in data}
        except Exception as exc:
            logger.error("Failed to load repositories", path=str(self.repos_file), error=str(exc))
            raise RepositoryRegistryError(f"Failed to load repositories from {self.repos_file}") from exc

    def save(self, repositories: dict[str, Repository]) -> None:
        """Persist repositories using the existing JSON list shape."""
        self.repos_file.parent.mkdir(parents=True, exist_ok=True)
        data = [repo.to_dict() for repo in repositories.values()]
        self.repos_file.write_text(json.dumps(data, indent=2))

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
        repositories = self.load()
        repositories[repo.id] = repo
        self.save(repositories)
        return repo

    def update(self, repo: Repository) -> Repository:
        """Update a repository and persist it."""
        repositories = self.load()
        repositories[repo.id] = repo
        self.save(repositories)
        return repo

    def remove(self, repo_id: str) -> Repository | None:
        """Remove a repository by full id or prefix and persist the registry."""
        repositories = self.load()
        resolved_id = repo_id if repo_id in repositories else self.resolve_id(repo_id)
        if resolved_id is None:
            return None

        repo = repositories.pop(resolved_id)
        self.save(repositories)
        return repo
