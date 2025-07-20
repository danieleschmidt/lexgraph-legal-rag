"""Semantic search utilities using simple embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Any
from pathlib import Path

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .metrics import SEARCH_LATENCY, SEARCH_REQUESTS

from .models import LegalDocument


@dataclass
class EmbeddingModel:
    """Lightweight embedding model using TF-IDF vectors."""

    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(stop_words="english")
    )

    def fit_transform(self, texts: List[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]):
        return self.vectorizer.transform(texts)


@dataclass
class EmbeddingIndex:
    """Store document embeddings and perform similarity search."""

    model: EmbeddingModel = field(default_factory=EmbeddingModel)
    _matrix: Any | None = None
    _docs: List[LegalDocument] = field(default_factory=list)

    @property
    def documents(self) -> List[LegalDocument]:
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

    def search(self, query: str, top_k: int = 5) -> List[Tuple[LegalDocument, float]]:
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


@dataclass
class SemanticSearchPipeline:
    """Pipeline for indexing and searching documents using embeddings."""

    index: EmbeddingIndex = field(default_factory=EmbeddingIndex)

    def ingest(self, docs: Iterable[LegalDocument]) -> None:
        """Add a collection of documents to the index."""
        self.index.add(docs)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[LegalDocument, float]]:
        """Return documents most relevant to ``query`` according to embedding similarity."""
        return self.index.search(query, top_k=top_k)

    def save(self, path: str | Path) -> None:
        """Persist the embedding index to ``path``."""
        self.index.save(path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved embedding index from ``path``."""
        self.index.load(path)
