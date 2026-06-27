"""Shared CodeRAG application services."""

from coderag.services.indexing import (
    IncrementalIndexingResult,
    IndexingOptions,
    IndexingResult,
    IndexingService,
)
from coderag.services.providers import ProviderConfigService
from coderag.services.registry import RepositoryRegistry, RepositoryRegistryError
from coderag.services.retrieval import QueryServiceResult, RetrievalService

__all__ = [
    "IndexingOptions",
    "IncrementalIndexingResult",
    "IndexingResult",
    "IndexingService",
    "ProviderConfigService",
    "QueryServiceResult",
    "RepositoryRegistry",
    "RepositoryRegistryError",
    "RetrievalService",
]
