"""Document entity model for representing source files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DocumentMetadata:
    """Metadata for a source document."""

    file_path: str
    language: Optional[str] = None
    size_bytes: int = 0
    line_count: int = 0
    encoding: str = "utf-8"

    @property
    def extension(self) -> str:
        """Get file extension."""
        return Path(self.file_path).suffix.lstrip(".")


@dataclass
class Document:
    """Represents a source code file loaded for processing."""

    content: str
    metadata: DocumentMetadata
    repo_id: str = ""

    @property
    def file_path(self) -> str:
        """Convenience accessor for file path."""
        return self.metadata.file_path

    @property
    def language(self) -> Optional[str]:
        """Convenience accessor for language."""
        return self.metadata.language

    @classmethod
    def from_file(cls, file_path: Path, repo_root: Path, repo_id: str = "") -> "Document":
        """Create Document from a file path."""
        content = file_path.read_text(encoding="utf-8")
        relative_path = str(file_path.relative_to(repo_root))
        line_count = content.count("\n") + 1 if content else 0

        language = _detect_language(file_path.suffix)

        metadata = DocumentMetadata(
            file_path=relative_path,
            language=language,
            size_bytes=file_path.stat().st_size,
            line_count=line_count,
        )

        return cls(content=content, metadata=metadata, repo_id=repo_id)


def _detect_language(extension: str) -> Optional[str]:
    """Detect programming language from file extension."""
    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".md": "markdown",
        ".rst": "restructuredtext",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
    }
    return extension_map.get(extension.lower())
