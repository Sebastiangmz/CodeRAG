"""Retrieval module: Query processing and similarity search."""

from typing import Any

__all__ = ["Retriever"]


def __getattr__(name: str) -> Any:
    if name == "Retriever":
        from coderag.retrieval.retriever import Retriever

        return Retriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
