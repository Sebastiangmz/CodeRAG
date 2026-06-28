"""Citation parsing and formatting."""

import re
from dataclasses import dataclass

from coderag.models.response import Citation, CitationVerification, RetrievedChunk


@dataclass(frozen=True)
class CitationVerificationResult:
    """Verification summary for a response's citations."""

    verifications: list[CitationVerification]
    grounded: bool


class CitationVerifier:
    """Verifies citations against retrieved evidence chunks."""

    def verify(
        self,
        citations: list[Citation],
        retrieved_chunks: list[RetrievedChunk],
    ) -> CitationVerificationResult:
        verifications = [self.verify_one(citation, retrieved_chunks) for citation in citations]
        grounded = bool(verifications) and all(verification.verified for verification in verifications)
        return CitationVerificationResult(verifications=verifications, grounded=grounded)

    def verify_one(self, citation: Citation, retrieved_chunks: list[RetrievedChunk]) -> CitationVerification:
        if citation.start_line < 1 or citation.end_line < citation.start_line:
            return CitationVerification.from_citation(citation, verified=False, reason="invalid_line_range")

        chunks_for_path = [chunk for chunk in retrieved_chunks if chunk.file_path == citation.file_path]
        if not chunks_for_path:
            return CitationVerification.from_citation(citation, verified=False, reason="path_not_retrieved")

        for chunk in chunks_for_path:
            if self._ranges_overlap(citation.start_line, citation.end_line, chunk.start_line, chunk.end_line):
                return CitationVerification.from_citation(
                    citation,
                    verified=True,
                    reason="verified",
                    chunk_id=chunk.chunk_id,
                )

        return CitationVerification.from_citation(citation, verified=False, reason="line_range_not_retrieved")

    @staticmethod
    def _ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
        return start_a <= end_b and start_b <= end_a


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
