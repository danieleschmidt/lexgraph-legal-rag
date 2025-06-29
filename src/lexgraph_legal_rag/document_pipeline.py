"""Simple legal document indexing and search pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Tuple

import logging

import json

from sklearn.feature_extraction.text import TfidfVectorizer

from .metrics import SEARCH_LATENCY, SEARCH_REQUESTS

from .models import LegalDocument
from .semantic_search import SemanticSearchPipeline


logger = logging.getLogger(__name__)


class VectorIndex:
    """In-memory vector index using TF-IDF vectors."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer()
        self._matrix: Any | None = None
        self._docs: List[LegalDocument] = []

    @property
    def documents(self) -> List[LegalDocument]:
        return self._docs

    def add(self, docs: Iterable[LegalDocument]) -> None:
        docs = list(docs)
        self._docs.extend(docs)
        texts = [d.text for d in self._docs]
        self._matrix = self._vectorizer.fit_transform(texts)
        logger.debug("Indexed %d documents", len(docs))

    def save(self, path: str | Path) -> None:
        """Persist the vector index to ``path`` using JSON serialization."""
        docs = [
            {"id": d.id, "text": d.text, "metadata": d.metadata} for d in self._docs
        ]
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(docs, fh)
        logger.info("Saved vector index to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved vector index from ``path``."""
        with Path(path).open("r", encoding="utf-8") as fh:
            docs_data = json.load(fh)
        docs = [LegalDocument(**d) for d in docs_data]
        self._vectorizer = TfidfVectorizer()
        self._matrix = None
        self._docs = []
        if docs:
            self.add(docs)
        logger.info("Loaded vector index from %s", path)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[LegalDocument, float]]:
        if self._matrix is None:
            return []
        with SEARCH_LATENCY.time():
            query_vec = self._vectorizer.transform([query])
            scores = (self._matrix @ query_vec.T).toarray().ravel()
            if not len(scores):
                return []
            indices = scores.argsort()[::-1][:top_k]
            results = [(self._docs[i], float(scores[i])) for i in indices]
        SEARCH_REQUESTS.inc()
        logger.debug("Search for '%s' returned %d results", query, len(results))
        return results


class LegalDocumentPipeline:
    """Pipeline for indexing and searching legal documents."""

    def __init__(self, use_semantic: bool = False) -> None:
        self.index = VectorIndex()
        self.semantic = SemanticSearchPipeline() if use_semantic else None

    def save_index(self, path: str | Path) -> None:
        """Persist the current vector index to ``path``."""
        self.index.save(path)
        if self.semantic is not None:
            base = Path(path)
            semantic_path = base.with_suffix(base.suffix + ".sem")
            self.semantic.save(semantic_path)
        logger.info("Pipeline index saved to %s", path)

    def load_index(self, path: str | Path) -> None:
        """Load a previously saved vector index from ``path``."""
        self.index.load(path)
        base = Path(path)
        semantic_path = base.with_suffix(base.suffix + ".sem")
        if semantic_path.exists():
            if self.semantic is None:
                self.semantic = SemanticSearchPipeline()
            self.semantic.load(semantic_path)
        logger.info("Pipeline index loaded from %s", path)

    def ingest_folder(self, folder: str | Path, pattern: str = "*.txt") -> None:
        folder_path = Path(folder)
        docs = []
        for path in folder_path.glob(pattern):
            text = path.read_text(encoding="utf-8")
            docs.append(
                LegalDocument(id=path.stem, text=text, metadata={"path": str(path)})
            )
        if docs:
            self.index.add(docs)
            if self.semantic is not None:
                self.semantic.ingest(docs)
            logger.info("Ingested %d documents from %s", len(docs), folder)

    def search(
        self, query: str, top_k: int = 5, semantic: bool | None = None
    ) -> List[Tuple[LegalDocument, float]]:
        if (semantic or semantic is None and self.semantic) and self.semantic:
            return self.semantic.search(query, top_k=top_k)
        return self.index.search(query, top_k=top_k)
