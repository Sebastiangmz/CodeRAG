"""Repository loading and cloning."""

from pathlib import Path
from typing import Callable, Optional

from git import Repo, GitCommandError

from coderag.config import get_settings
from coderag.logging import get_logger
from coderag.ingestion.validator import GitHubRepoInfo

logger = get_logger(__name__)

ProgressCallback = Callable[[str, int], None]


class LoaderError(Exception):
    """Repository loading error."""
    pass


class RepositoryLoader:
    """Loads repositories from GitHub."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.cache_dir = cache_dir or settings.ingestion.repos_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_repo_path(self, repo_info: GitHubRepoInfo) -> Path:
        return self.cache_dir / repo_info.owner / repo_info.name

    def clone_repository(
        self,
        repo_info: GitHubRepoInfo,
        branch: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        repo_path = self.get_repo_path(repo_info)

        # Try branches in order: specified, repo default, main, master
        branches_to_try = []
        if branch:
            branches_to_try.append(branch)
        if repo_info.branch and repo_info.branch not in branches_to_try:
            branches_to_try.append(repo_info.branch)
        if "main" not in branches_to_try:
            branches_to_try.append("main")
        if "master" not in branches_to_try:
            branches_to_try.append("master")

        if repo_path.exists():
            logger.info("Repository exists, updating", path=str(repo_path))
            return self._update_repository(repo_path, branches_to_try[0], progress_callback)

        if progress_callback:
            progress_callback("Cloning repository", 0)

        repo_path.parent.mkdir(parents=True, exist_ok=True)

        last_error = None
        for try_branch in branches_to_try:
            try:
                logger.info("Trying to clone", url=repo_info.clone_url, branch=try_branch)
                Repo.clone_from(
                    repo_info.clone_url,
                    repo_path,
                    branch=try_branch,
                    depth=1,
                    single_branch=True,
                )
                if progress_callback:
                    progress_callback("Clone complete", 100)
                logger.info("Repository cloned", path=str(repo_path), branch=try_branch)
                return repo_path
            except GitCommandError as e:
                last_error = e
                logger.debug("Branch not found, trying next", branch=try_branch)
                # Clean up partial clone if any
                import shutil
                shutil.rmtree(repo_path, ignore_errors=True)
                continue

        raise LoaderError(f"Failed to clone repository (tried branches: {branches_to_try}): {last_error}")

    def _update_repository(
        self,
        repo_path: Path,
        branch: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        try:
            repo = Repo(repo_path)
            if progress_callback:
                progress_callback("Fetching updates", 30)
            repo.remotes.origin.fetch()
            repo.git.checkout(branch)
            repo.remotes.origin.pull()
            if progress_callback:
                progress_callback("Update complete", 100)
            logger.info("Repository updated", path=str(repo_path))
            return repo_path
        except GitCommandError as e:
            logger.warning("Update failed, re-cloning", error=str(e))
            import shutil
            shutil.rmtree(repo_path, ignore_errors=True)
            raise LoaderError(f"Failed to update, please re-clone: {e}")

    def is_cached(self, repo_info: GitHubRepoInfo) -> bool:
        return self.get_repo_path(repo_info).exists()

    def delete_cache(self, repo_info: GitHubRepoInfo) -> None:
        repo_path = self.get_repo_path(repo_info)
        if repo_path.exists():
            import shutil
            shutil.rmtree(repo_path)
            logger.info("Cache deleted", path=str(repo_path))
