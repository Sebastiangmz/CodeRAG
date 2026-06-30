"""Hybrid retrieval over vector and lexical indexed chunks."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from coderag.models.chunk import Chunk
from coderag.models.response import RetrievedChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class RetrievalBudget:
    """Deterministic caps for context-producing retrieval."""

    max_chunks: int
    max_tokens: int
    max_chunks_per_file: int = 3


@dataclass
class _Candidate:
    chunk: Chunk
    vector_score: float = 0.0
    lexical_score: float = 0.0
    path_boost: float = 0.0
    name_boost: float = 0.0
    sources: set[str] | None = None

    def __post_init__(self) -> None:
        if self.sources is None:
            self.sources = set()

    @property
    def combined_score(self) -> float:
        return (0.35 * self.vector_score) + (0.55 * self.lexical_score) + self.path_boost + self.name_boost


class HybridRetriever:
    """Merge semantic vector hits with deterministic lexical matches."""

    def __init__(self, vectorstore: Any, embedder: Any | None = None, similarity_threshold: float = 0.0) -> None:
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        repo_id: str,
        top_k: int = 5,
        budget: RetrievalBudget | None = None,
    ) -> list[RetrievedChunk]:
        """Return ranked, deduplicated chunks using vector and lexical evidence."""
        top_k = max(int(top_k), 1)
        budget = budget or RetrievalBudget(max_chunks=top_k, max_tokens=4000, max_chunks_per_file=3)
        query_tokens = _tokenize(query)
        candidates: list[_Candidate] = []

        for chunk, score in self._semantic_candidates(query, repo_id, top_k):
            candidate = _Candidate(chunk=chunk, vector_score=float(score), sources={"vector"})
            candidates.append(candidate)

        candidates.extend(self._lexical_candidates(query_tokens, repo_id))
        merged = self._merge_candidates(candidates)
        ranked = sorted(
            merged,
            key=lambda candidate: (
                -candidate.combined_score,
                candidate.chunk.file_path,
                candidate.chunk.start_line,
                candidate.chunk.end_line,
                candidate.chunk.id,
            ),
        )
        return self._apply_budget(ranked, budget)

    def _semantic_candidates(self, query: str, repo_id: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self.embedder is None or not hasattr(self.vectorstore, "query"):
            return []
        query_embedding = self.embedder.generate_embedding(query, is_query=True)
        return list(
            self.vectorstore.query(
                query_embedding=query_embedding,
                repo_id=repo_id,
                top_k=top_k,
                similarity_threshold=self.similarity_threshold,
            )
        )

    def _lexical_candidates(self, query_tokens: list[str], repo_id: str) -> list[_Candidate]:
        if not query_tokens or not hasattr(self.vectorstore, "get_repo_chunks"):
            return []
        chunks = list(self.vectorstore.get_repo_chunks(repo_id))
        if not chunks:
            return []

        query_counts = Counter(query_tokens)
        chunk_tokens = [_tokenize(_lexical_text(chunk)) for chunk in chunks]
        document_frequency: Counter[str] = Counter()
        for tokens in chunk_tokens:
            document_frequency.update(set(tokens))

        raw_scores: list[float] = []
        for tokens in chunk_tokens:
            counts = Counter(tokens)
            length_norm = max(len(tokens), 1)
            score = 0.0
            for token, query_weight in query_counts.items():
                term_frequency = counts[token]
                if term_frequency == 0:
                    continue
                idf = math.log((1 + len(chunks)) / (1 + document_frequency[token])) + 1.0
                score += query_weight * idf * (term_frequency / length_norm)
            raw_scores.append(score)

        max_score = max(raw_scores, default=0.0)
        candidates: list[_Candidate] = []
        for chunk, raw_score in zip(chunks, raw_scores, strict=True):
            if raw_score <= 0.0:
                continue
            lexical_score = raw_score / max_score if max_score > 0.0 else 0.0
            path_boost = 0.08 if _contains_any(chunk.file_path, query_tokens) else 0.0
            name_boost = 0.12 if chunk.name and _contains_any(chunk.name, query_tokens) else 0.0
            candidates.append(
                _Candidate(
                    chunk=chunk,
                    lexical_score=lexical_score,
                    path_boost=path_boost,
                    name_boost=name_boost,
                    sources={"lexical"},
                )
            )
        return candidates

    def _merge_candidates(self, candidates: list[_Candidate]) -> list[_Candidate]:
        merged: list[_Candidate] = []
        for candidate in candidates:
            existing = next((item for item in merged if _overlaps(item.chunk, candidate.chunk)), None)
            if existing is None:
                merged.append(candidate)
                continue
            if candidate.vector_score > existing.vector_score:
                existing.vector_score = candidate.vector_score
            if candidate.lexical_score > existing.lexical_score:
                existing.lexical_score = candidate.lexical_score
                existing.path_boost = max(existing.path_boost, candidate.path_boost)
                existing.name_boost = max(existing.name_boost, candidate.name_boost)
            if candidate.sources:
                existing.sources = (existing.sources or set()) | candidate.sources
        return merged

    def _apply_budget(self, candidates: list[_Candidate], budget: RetrievalBudget) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        tokens_used = 0
        per_file: defaultdict[str, int] = defaultdict(int)
        max_chunks = max(int(budget.max_chunks), 1)
        max_tokens = max(int(budget.max_tokens), 1)
        max_chunks_per_file = max(int(budget.max_chunks_per_file), 1)

        for candidate in candidates:
            if len(selected) >= max_chunks:
                break
            chunk = candidate.chunk
            token_estimate = _estimate_tokens(chunk.content)
            if per_file[chunk.file_path] >= max_chunks_per_file:
                continue
            if selected and tokens_used + token_estimate > max_tokens:
                continue
            if not selected and token_estimate > max_tokens:
                token_estimate = max_tokens

            selected.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    relevance_score=candidate.combined_score,
                    chunk_type=chunk.chunk_type.value,
                    name=chunk.name,
                    score_breakdown={
                        "vector": candidate.vector_score,
                        "lexical": candidate.lexical_score,
                        "path": candidate.path_boost,
                        "name": candidate.name_boost,
                        "combined": candidate.combined_score,
                    },
                    retrieval_sources=_ordered_sources(candidate.sources or set()),
                    token_estimate=token_estimate,
                    ranking_reason=_ranking_reason(candidate),
                )
            )
            tokens_used += token_estimate
            per_file[chunk.file_path] += 1
        return selected


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _lexical_text(chunk: Chunk) -> str:
    return " ".join(part for part in [chunk.file_path, chunk.name or "", chunk.content] if part)


def _contains_any(text: str, tokens: list[str]) -> bool:
    haystack = text.lower()
    return any(token in haystack for token in tokens)


def _overlaps(left: Chunk, right: Chunk) -> bool:
    return left.file_path == right.file_path and left.start_line <= right.end_line and right.start_line <= left.end_line


def _estimate_tokens(content: str) -> int:
    words = content.split()
    if words:
        return len(words)
    return max(1, math.ceil(len(content) / 4))


def _ordered_sources(sources: set[str]) -> list[str]:
    return [source for source in ["vector", "lexical"] if source in sources]


def _ranking_reason(candidate: _Candidate) -> str:
    parts: list[str] = []
    if candidate.vector_score > 0:
        parts.append(f"vector={candidate.vector_score:.3f}")
    if candidate.lexical_score > 0:
        parts.append(f"lexical={candidate.lexical_score:.3f}")
    if candidate.path_boost > 0:
        parts.append(f"path_boost={candidate.path_boost:.3f}")
    if candidate.name_boost > 0:
        parts.append(f"name_boost={candidate.name_boost:.3f}")
    return ", ".join(parts) or "ranked by stable fallback order"
