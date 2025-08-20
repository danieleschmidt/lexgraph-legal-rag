"""Semantic search utilities using simple embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .metrics import SEARCH_LATENCY
from .metrics import SEARCH_REQUESTS
from .models import LegalDocument


@dataclass
class EmbeddingModel:
    """Lightweight embedding model using TF-IDF vectors."""

    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(stop_words="english")
    )

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)


@dataclass
class EmbeddingIndex:
    """Store document embeddings and perform similarity search."""

    model: EmbeddingModel = field(default_factory=EmbeddingModel)
    _matrix: Any | None = None
    _docs: list[LegalDocument] = field(default_factory=list)

    @property
    def documents(self) -> list[LegalDocument]:
        return self._docs

    def add(self, docs: Iterable[LegalDocument]) -> None:
        docs = list(docs)
        if not docs:
            return
        self._docs.extend(docs)
        texts = [d.text for d in self._docs]
        self._matrix = self.model.fit_transform(texts)

    def save(self, path: str | Path) -> None:
        """Persist the embedding index to ``path`` using JSON."""
        docs = [
            {"id": d.id, "text": d.text, "metadata": d.metadata} for d in self._docs
        ]
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(docs, fh)

    def load(self, path: str | Path) -> None:
        """Load a previously saved embedding index from ``path``."""
        with Path(path).open("r", encoding="utf-8") as fh:
            docs_data = json.load(fh)
        docs = [LegalDocument(**d) for d in docs_data]
        self.model.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self._docs = []
        if docs:
            self.add(docs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[LegalDocument, float]]:
        if self._matrix is None:
            return []
        with SEARCH_LATENCY.labels(search_type="semantic").time():
            query_vec = self.model.transform([query])
            scores = cosine_similarity(query_vec, self._matrix).ravel()
            if not len(scores):
                return []
            indices = np.argsort(scores)[::-1][:top_k]
            results = [(self._docs[i], float(scores[i])) for i in indices]
        SEARCH_REQUESTS.labels(search_type="semantic").inc()
        return results

    def batch_search(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[tuple[LegalDocument, float]]]:
        """Perform batch search for multiple queries efficiently."""
        if self._matrix is None:
            return [[] for _ in queries]

        results = []
        with SEARCH_LATENCY.labels(search_type="semantic_batch").time():
            # Transform all queries at once for better efficiency
            query_vecs = self.model.transform(queries)
            # Compute all similarity scores in one batch operation
            all_scores = cosine_similarity(query_vecs, self._matrix)

            for i, _query in enumerate(queries):
                scores = all_scores[i]
                if not len(scores):
                    results.append([])
                    continue

                # Get top_k results for this query
                indices = np.argsort(scores)[::-1][:top_k]
                query_results = [
                    (self._docs[idx], float(scores[idx])) for idx in indices
                ]
                results.append(query_results)

        SEARCH_REQUESTS.labels(search_type="semantic_batch").inc(len(queries))
        return results


@dataclass
class SemanticSearchPipeline:
    """Pipeline for indexing and searching documents using embeddings."""

    index: EmbeddingIndex = field(default_factory=EmbeddingIndex)

    def ingest(self, docs: Iterable[LegalDocument]) -> None:
        """Add a collection of documents to the index."""
        self.index.add(docs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[LegalDocument, float]]:
        """Return documents most relevant to ``query`` according to embedding similarity."""
        return self.index.search(query, top_k=top_k)

    def batch_search(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[tuple[LegalDocument, float]]]:
        """Return documents most relevant to multiple queries using batch processing."""
        return self.index.batch_search(queries, top_k=top_k)

    def save(self, path: str | Path) -> None:
        """Persist the embedding index to ``path``."""
        self.index.save(path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved embedding index from ``path``."""
        self.index.load(path)
