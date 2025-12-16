"""Tests for citation parsing."""

import pytest

from coderag.generation.citations import CitationParser
from coderag.models.response import Citation


def test_parse_single_citation():
    """Test parsing a single citation."""
    parser = CitationParser()
    text = "The function is defined in [src/auth.py:45-78]."

    citations = parser.parse_citations(text)

    assert len(citations) == 1
    assert citations[0].file_path == "src/auth.py"
    assert citations[0].start_line == 45
    assert citations[0].end_line == 78


def test_parse_multiple_citations():
    """Test parsing multiple citations."""
    parser = CitationParser()
    text = "See [file1.py:10-20] and [file2.py:30-40] for details."

    citations = parser.parse_citations(text)

    assert len(citations) == 2
    assert citations[0].file_path == "file1.py"
    assert citations[1].file_path == "file2.py"


def test_format_citation():
    """Test citation formatting."""
    parser = CitationParser()

    formatted = parser.format_citation("src/test.py", 10, 20)

    assert formatted == "[src/test.py:10-20]"


def test_has_citations():
    """Test checking if text has citations."""
    parser = CitationParser()

    assert parser.has_citations("Text with [file.py:1-10] citation")
    assert not parser.has_citations("Text without citations")


def test_count_citations():
    """Test counting citations."""
    parser = CitationParser()

    text = "Multiple [a.py:1-2] citations [b.py:3-4] here [c.py:5-6]"
    assert parser.count_citations(text) == 3

    text = "No citations here"
    assert parser.count_citations(text) == 0
