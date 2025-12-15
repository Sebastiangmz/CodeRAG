"""Citation parsing and formatting."""

import re
from typing import Optional

from coderag.models.response import Citation


class CitationParser:
    """Parses and validates citations from LLM responses."""

    # Pattern to match citations like [file.py:10-20] or [path/to/file.py:10-20]
    CITATION_PATTERN = re.compile(r"\[([^\]]+):(\d+)-(\d+)\]")

    def parse_citations(self, text: str) -> list[Citation]:
        """Extract all citations from text.

        Args:
            text: Text containing citations

        Returns:
            List of parsed Citation objects
        """
        citations = []
        for match in self.CITATION_PATTERN.finditer(text):
            file_path = match.group(1)
            start_line = int(match.group(2))
            end_line = int(match.group(3))

            citations.append(Citation(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
            ))

        return citations

    def validate_citation(self, citation: Citation, available_files: set[str]) -> bool:
        """Check if a citation references an existing file."""
        return citation.file_path in available_files

    def validate_citations(
        self,
        citations: list[Citation],
        available_files: set[str],
    ) -> tuple[list[Citation], list[Citation]]:
        """Validate multiple citations.

        Returns:
            Tuple of (valid_citations, invalid_citations)
        """
        valid = []
        invalid = []

        for citation in citations:
            if self.validate_citation(citation, available_files):
                valid.append(citation)
            else:
                invalid.append(citation)

        return valid, invalid

    def format_citation(self, file_path: str, start_line: int, end_line: int) -> str:
        """Format a citation string."""
        return f"[{file_path}:{start_line}-{end_line}]"

    def has_citations(self, text: str) -> bool:
        """Check if text contains any citations."""
        return bool(self.CITATION_PATTERN.search(text))

    def count_citations(self, text: str) -> int:
        """Count citations in text."""
        return len(self.CITATION_PATTERN.findall(text))

    def extract_unique_files(self, citations: list[Citation]) -> set[str]:
        """Get unique file paths from citations."""
        return {c.file_path for c in citations}
