"""File filtering for repository indexing."""

import fnmatch
from pathlib import Path
from typing import Iterator, Optional

from coderag.config import get_settings
from coderag.logging import get_logger

logger = get_logger(__name__)


class FileFilter:
    """Filters files for indexing based on patterns."""

    def __init__(
        self,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        max_file_size_kb: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.include_patterns = include_patterns or settings.ingestion.include_patterns
        self.exclude_patterns = exclude_patterns or settings.ingestion.exclude_patterns
        self.max_file_size = (max_file_size_kb or settings.ingestion.max_file_size_kb) * 1024

    def should_include(self, file_path: Path, repo_root: Path) -> bool:
        relative_path = str(file_path.relative_to(repo_root))

        # Check exclusions first
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                return False
            if fnmatch.fnmatch(file_path.name, pattern):
                return False

        # Check inclusions
        for pattern in self.include_patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
            if fnmatch.fnmatch(relative_path, pattern):
                return True

        return False

    def check_file_size(self, file_path: Path) -> bool:
        try:
            return file_path.stat().st_size <= self.max_file_size
        except OSError:
            return False

    def is_binary(self, file_path: Path) -> bool:
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(8192)
                return b"\x00" in chunk
        except (OSError, IOError):
            return True

    def filter_files(self, repo_root: Path) -> Iterator[Path]:
        skipped_count = 0
        included_count = 0

        for file_path in repo_root.rglob("*"):
            if not file_path.is_file():
                continue

            if not self.should_include(file_path, repo_root):
                skipped_count += 1
                continue

            if not self.check_file_size(file_path):
                logger.debug("Skipping large file", path=str(file_path))
                skipped_count += 1
                continue

            if self.is_binary(file_path):
                logger.debug("Skipping binary file", path=str(file_path))
                skipped_count += 1
                continue

            included_count += 1
            yield file_path

        logger.info("File filtering complete", included=included_count, skipped=skipped_count)
