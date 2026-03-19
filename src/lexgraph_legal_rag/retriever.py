"""LegalRetriever and MultiAgentRetriever — query the FAISS index with citations."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Tuple

import numpy as np

from .indexer import DocumentChunk
from .indexer import LegalDocumentIndexer


logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with relevance score and citation."""

    chunk: DocumentChunk
    score: float

    @property
    def citation(self) -> str:
        return self.chunk.citation

    @property
    def text(self) -> str:
        return self.chunk.text

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(score={self.score:.4f}, "
            f"doc={self.chunk.doc_name!r}, "
            f"section={self.chunk.section!r})"
        )


@dataclass
class LegalRetriever:
    """
    Retrieves top-k relevant chunks from a LegalDocumentIndexer.

    Returns chunks with source citations: doc name, section, char offset.
    No external API calls — pure FAISS + TF-IDF.
    """

    indexer: LegalDocumentIndexer = field(default_factory=LegalDocumentIndexer)
    top_k: int = 5
    min_score: float = 0.0

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for the given query.

        Args:
            query: Natural language query or legal question.

        Returns:
            List of RetrievedChunk sorted by relevance (highest first).
        """
        self.indexer.ensure_index()

        if self.indexer._index is None:
            logger.warning("Index is empty — no documents indexed yet")
            return []

        # Embed query with the same vectorizer
        query_vec = (
            self.indexer.vectorizer.transform([query]).astype(np.float32).toarray()
        )
        # L2-normalize
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        k = min(self.top_k, len(self.indexer._chunks))
        scores, indices = self.indexer._index.search(query_vec, k)

        results = []
        for idx, score in zip(indices.ravel(), scores.ravel()):
            if idx < 0:
                continue
            if float(score) < self.min_score:
                continue
            chunk = self.indexer._chunks[int(idx)]
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))

        results.sort(key=lambda r: r.score, reverse=True)
        logger.debug("Query '%s' → %d results (top score: %.4f)",
                     query[:50], len(results),
                     results[0].score if results else 0.0)
        return results

    def retrieve_with_context(self, query: str) -> str:
        """Return formatted string with retrieved chunks and citations."""
        results = self.retrieve(query)
        if not results:
            return f"No relevant documents found for: {query}"

        lines = [f"Query: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] Score: {r.score:.4f} | {r.citation}")
            lines.append(f"    {r.text[:300]}{'...' if len(r.text) > 300 else ''}")
            lines.append("")
        return "\n".join(lines)


def _expand_query(query: str) -> List[str]:
    """
    Generate alternative phrasings of a legal query.

    Used by MultiAgentRetriever to improve recall.
    """
    variants = [query]

    # Strip question words and rephrase as keyword query
    keyword_form = re.sub(
        r"^(what|where|when|how|who|is|are|does|do|can|may|shall|will)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip()
    if keyword_form and keyword_form != query:
        variants.append(keyword_form)

    # Add "law" / "legal" qualifier if not present
    if "legal" not in query.lower() and "law" not in query.lower():
        variants.append(f"legal {query}")

    # Extract key noun phrases (simple heuristic: capitalized words)
    caps_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", query)
    if caps_words:
        variants.append(" ".join(caps_words))

    return list(dict.fromkeys(variants))  # deduplicate preserving order


@dataclass
class MultiAgentRetriever:
    """
    Runs multiple query variants in parallel and merges/deduplicates results.

    Each "agent" is a different query phrasing — simulates multi-perspective
    retrieval without needing multiple models or external APIs.

    Feeds into DocGraph's contract extractor and law prover.
    """

    indexer: LegalDocumentIndexer = field(default_factory=LegalDocumentIndexer)
    top_k: int = 5
    min_score: float = 0.0
    max_workers: int = 4

    def _retriever(self) -> LegalRetriever:
        return LegalRetriever(
            indexer=self.indexer,
            top_k=self.top_k,
            min_score=self.min_score,
        )

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Run expanded query variants in parallel, merge and rank results.

        Args:
            query: The original natural language query.

        Returns:
            Deduplicated list of RetrievedChunk sorted by score.
        """
        variants = _expand_query(query)
        logger.info("MultiAgentRetriever: %d query variants for '%s'",
                    len(variants), query[:50])

        retriever = self._retriever()
        all_results: List[RetrievedChunk] = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(variants))) as exe:
            futures = {exe.submit(retriever.retrieve, v): v for v in variants}
            for fut in as_completed(futures):
                try:
                    all_results.extend(fut.result())
                except Exception as e:
                    logger.warning("Variant '%s' failed: %s", futures[fut], e)

        # Deduplicate by (doc_name, section) — keep highest score
        seen: dict[Tuple[str, str], RetrievedChunk] = {}
        for r in all_results:
            key = (r.chunk.doc_name, r.chunk.section)
            if key not in seen or r.score > seen[key].score:
                seen[key] = r

        merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        logger.info("MultiAgentRetriever: %d unique results after dedup", len(merged))
        return merged[: self.top_k * 2]  # return up to 2x top_k for diversity

    def retrieve_with_context(self, query: str) -> str:
        """Return formatted string with multi-agent retrieved chunks and citations."""
        results = self.retrieve(query)
        if not results:
            return f"No relevant documents found for: {query}"

        lines = [f"Multi-Agent Query: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] Score: {r.score:.4f} | {r.citation}")
            lines.append(f"    {r.text[:300]}{'...' if len(r.text) > 300 else ''}")
            lines.append("")
        return "\n".join(lines)
