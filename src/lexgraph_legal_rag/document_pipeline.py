"""Simple legal document indexing and search pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from .models import LegalDocument
from .semantic_search import SemanticSearchPipeline


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
        self._docs.extend(list(docs))
        texts = [d.text for d in self._docs]
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[LegalDocument, float]]:
        if self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = (self._matrix @ query_vec.T).toarray().ravel()
        if not len(scores):
            return []
        indices = scores.argsort()[::-1][:top_k]
        return [(self._docs[i], float(scores[i])) for i in indices]


class LegalDocumentPipeline:
    """Pipeline for indexing and searching legal documents."""

    def __init__(self, use_semantic: bool = False) -> None:
        self.index = VectorIndex()
        self.semantic = SemanticSearchPipeline() if use_semantic else None

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

    def search(
        self, query: str, top_k: int = 5, semantic: bool | None = None
    ) -> List[Tuple[LegalDocument, float]]:
        if (semantic or semantic is None and self.semantic) and self.semantic:
            return self.semantic.search(query, top_k=top_k)
        return self.index.search(query, top_k=top_k)
