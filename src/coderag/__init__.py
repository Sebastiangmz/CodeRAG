"""CodeRAG: RAG-based Q&A system for code repositories with verifiable citations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("code-rag-me")
except PackageNotFoundError:
    __version__ = "0.1.2"
