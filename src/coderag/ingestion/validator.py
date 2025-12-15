"""GitHub URL validation and parsing."""

import re
from dataclasses import dataclass
from typing import Optional

import httpx

from coderag.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GitHubRepoInfo:
    """Parsed GitHub repository information."""

    owner: str
    name: str
    url: str
    branch: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"

    @property
    def api_url(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.name}"


class ValidationError(Exception):
    """URL validation error."""
    pass


class GitHubURLValidator:
    """Validates and parses GitHub repository URLs."""

    GITHUB_PATTERNS = [
        r"^https?://github\.com/(?P<owner>[^/]+)/(?P<name>[^/]+?)(?:\.git)?/?$",
        r"^git@github\.com:(?P<owner>[^/]+)/(?P<name>[^/]+?)(?:\.git)?$",
        r"^(?P<owner>[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38})/(?P<name>[a-zA-Z0-9._-]+)$",
    ]

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._patterns = [re.compile(p) for p in self.GITHUB_PATTERNS]

    def parse_url(self, url: str) -> GitHubRepoInfo:
        url = url.strip()
        for pattern in self._patterns:
            match = pattern.match(url)
            if match:
                owner = match.group("owner")
                name = match.group("name").rstrip(".git")
                if not self._is_valid_name(owner) or not self._is_valid_name(name):
                    raise ValidationError(f"Invalid owner or repository name: {url}")
                return GitHubRepoInfo(owner=owner, name=name, url=f"https://github.com/{owner}/{name}")
        raise ValidationError(f"Invalid GitHub URL: {url}. Expected: https://github.com/owner/repo")

    def _is_valid_name(self, name: str) -> bool:
        if not name or len(name) > 100:
            return False
        return bool(re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$", name))

    async def validate_repository(self, url: str, check_accessibility: bool = True) -> GitHubRepoInfo:
        repo_info = self.parse_url(url)
        if check_accessibility:
            await self._check_repo_accessible(repo_info)
        logger.info("Repository validated", owner=repo_info.owner, name=repo_info.name)
        return repo_info

    async def _check_repo_accessible(self, repo_info: GitHubRepoInfo) -> None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(repo_info.api_url)
                if response.status_code == 404:
                    raise ValidationError(f"Repository not found: {repo_info.full_name}")
                elif response.status_code == 403:
                    raise ValidationError(f"Access denied: {repo_info.full_name}")
                elif response.status_code != 200:
                    raise ValidationError(f"HTTP error {response.status_code}: {repo_info.full_name}")
                data = response.json()
                if data.get("private", False):
                    raise ValidationError(f"Private repository not supported: {repo_info.full_name}")
                repo_info.branch = data.get("default_branch", "main")
            except httpx.TimeoutException:
                raise ValidationError(f"Timeout checking repository: {repo_info.full_name}")
            except httpx.RequestError as e:
                raise ValidationError(f"Network error: {str(e)}")

    def validate_url_sync(self, url: str) -> GitHubRepoInfo:
        return self.parse_url(url)
