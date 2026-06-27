"""Repository metadata persistence helpers for the Gradio UI."""

import json
from pathlib import Path

from coderag.models.repository import Repository


def load_repositories(repos_file: Path) -> dict[str, Repository]:
    """Load UI repository metadata, failing closed on corrupt existing data."""
    if not repos_file.exists():
        return {}

    data = json.loads(repos_file.read_text())
    return {repo_data["id"]: Repository.from_dict(repo_data) for repo_data in data}


def save_repositories(repos_file: Path, repositories: dict[str, Repository]) -> None:
    """Persist UI repository metadata using the existing list-of-dicts JSON shape."""
    repos_file.parent.mkdir(parents=True, exist_ok=True)
    data = [repo.to_dict() for repo in repositories.values()]
    repos_file.write_text(json.dumps(data, indent=2))
