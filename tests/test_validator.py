"""Tests for GitHub URL validator."""

import pytest

from coderag.ingestion.validator import GitHubURLValidator, ValidationError


def test_parse_https_url():
    """Test parsing HTTPS GitHub URL."""
    validator = GitHubURLValidator()
    repo_info = validator.parse_url("https://github.com/owner/repo")

    assert repo_info.owner == "owner"
    assert repo_info.name == "repo"
    assert repo_info.full_name == "owner/repo"
    assert repo_info.clone_url == "https://github.com/owner/repo.git"


def test_parse_git_url():
    """Test parsing git@ GitHub URL."""
    validator = GitHubURLValidator()
    repo_info = validator.parse_url("git@github.com:owner/repo.git")

    assert repo_info.owner == "owner"
    assert repo_info.name == "repo"


def test_parse_short_format():
    """Test parsing owner/repo format."""
    validator = GitHubURLValidator()
    repo_info = validator.parse_url("owner/repo")

    assert repo_info.owner == "owner"
    assert repo_info.name == "repo"


def test_invalid_url():
    """Test invalid URL raises ValidationError."""
    validator = GitHubURLValidator()

    with pytest.raises(ValidationError):
        validator.parse_url("not-a-github-url")

    with pytest.raises(ValidationError):
        validator.parse_url("https://gitlab.com/owner/repo")
