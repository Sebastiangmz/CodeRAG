"""Code chunking with Tree-sitter and text fallback."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from coderag.config import get_settings
from coderag.logging import get_logger
from coderag.models.chunk import Chunk, ChunkMetadata, ChunkType
from coderag.models.document import Document

logger = get_logger(__name__)


@dataclass
class ChunkerConfig:
    """Chunker configuration."""
    chunk_size: int = 1500
    chunk_overlap: int = 200
    min_chunk_size: int = 50


class CodeChunker:
    """Chunks code files into semantic units."""

    def __init__(self, config: Optional[ChunkerConfig] = None) -> None:
        settings = get_settings()
        self.config = config or ChunkerConfig(
            chunk_size=settings.ingestion.chunk_size,
            chunk_overlap=settings.ingestion.chunk_overlap,
        )
        self._tree_sitter_available = self._check_tree_sitter()

    def _check_tree_sitter(self) -> bool:
        try:
            import tree_sitter_python
            return True
        except ImportError:
            logger.warning("Tree-sitter not available, using text chunking")
            return False

    def chunk_document(self, document: Document) -> Iterator[Chunk]:
        if document.language == "python" and self._tree_sitter_available:
            yield from self._chunk_python(document)
        else:
            yield from self._chunk_text(document)

    def _chunk_python(self, document: Document) -> Iterator[Chunk]:
        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser

            PY_LANGUAGE = Language(tspython.language())
            parser = Parser(PY_LANGUAGE)
            tree = parser.parse(bytes(document.content, "utf-8"))

            yield from self._extract_python_chunks(tree.root_node, document)

        except Exception as e:
            logger.warning("Tree-sitter parsing failed, falling back to text", error=str(e))
            yield from self._chunk_text(document)

    def _extract_python_chunks(self, node, document: Document) -> Iterator[Chunk]:
        lines = document.content.split("\n")

        for child in node.children:
            if child.type in ("function_definition", "async_function_definition"):
                yield self._create_chunk_from_node(child, document, lines, ChunkType.FUNCTION)
            elif child.type == "class_definition":
                yield self._create_chunk_from_node(child, document, lines, ChunkType.CLASS)
                # Also extract methods
                for class_child in child.children:
                    if class_child.type == "block":
                        for block_child in class_child.children:
                            if block_child.type in ("function_definition", "async_function_definition"):
                                yield self._create_chunk_from_node(
                                    block_child, document, lines, ChunkType.METHOD,
                                    parent_name=self._get_node_name(child)
                                )

        # If no semantic chunks found, fall back to text chunking
        if not any(child.type in ("function_definition", "class_definition", "async_function_definition")
                   for child in node.children):
            yield from self._chunk_text(document)

    def _create_chunk_from_node(
        self,
        node,
        document: Document,
        lines: list[str],
        chunk_type: ChunkType,
        parent_name: Optional[str] = None,
    ) -> Chunk:
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        content = "\n".join(lines[start_line - 1:end_line])
        name = self._get_node_name(node)
        signature = self._get_signature(node, lines)
        docstring = self._get_docstring(node, lines)

        metadata = ChunkMetadata(
            file_path=document.file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            language=document.language,
            name=name,
            signature=signature,
            docstring=docstring,
            parent_name=parent_name,
        )

        return Chunk(content=content, metadata=metadata, repo_id=document.repo_id)

    def _get_node_name(self, node) -> Optional[str]:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return None

    def _get_signature(self, node, lines: list[str]) -> Optional[str]:
        if node.type in ("function_definition", "async_function_definition"):
            start_line = node.start_point[0]
            return lines[start_line].strip()
        return None

    def _get_docstring(self, node, lines: list[str]) -> Optional[str]:
        for child in node.children:
            if child.type == "block":
                for block_child in child.children:
                    if block_child.type == "expression_statement":
                        for expr_child in block_child.children:
                            if expr_child.type == "string":
                                return expr_child.text.decode("utf-8").strip('"""\'\'\'')
        return None

    def _chunk_text(self, document: Document) -> Iterator[Chunk]:
        lines = document.content.split("\n")
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        current_start = 0
        while current_start < len(lines):
            # Calculate chunk boundaries
            char_count = 0
            end_line = current_start

            while end_line < len(lines) and char_count < chunk_size:
                char_count += len(lines[end_line]) + 1
                end_line += 1

            content = "\n".join(lines[current_start:end_line])

            if len(content.strip()) >= self.config.min_chunk_size:
                metadata = ChunkMetadata(
                    file_path=document.file_path,
                    start_line=current_start + 1,
                    end_line=end_line,
                    chunk_type=ChunkType.TEXT,
                    language=document.language,
                )
                yield Chunk(content=content, metadata=metadata, repo_id=document.repo_id)

            # Move start with overlap
            overlap_lines = 0
            overlap_chars = 0
            while overlap_lines < end_line - current_start and overlap_chars < overlap:
                overlap_chars += len(lines[end_line - 1 - overlap_lines]) + 1
                overlap_lines += 1

            current_start = end_line - overlap_lines
            if current_start <= 0 or end_line >= len(lines):
                break

    def chunk_files(self, documents: Iterator[Document]) -> Iterator[Chunk]:
        total_chunks = 0
        for doc in documents:
            doc_chunks = 0
            for chunk in self.chunk_document(doc):
                doc_chunks += 1
                total_chunks += 1
                yield chunk
            logger.debug("Document chunked", file=doc.file_path, chunks=doc_chunks)
        logger.info("Chunking complete", total_chunks=total_chunks)
