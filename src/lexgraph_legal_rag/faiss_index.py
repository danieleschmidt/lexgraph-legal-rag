"""FAISS-based vector index for scalable similarity search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple
from pathlib import Path

import json

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .metrics import SEARCH_LATENCY, SEARCH_REQUESTS

from .models import LegalDocument


@dataclass
class FaissVectorIndex:
    """Vector index using FAISS for nearest-neighbor search."""

    vectorizer: TfidfVectorizer = field(default_factory=TfidfVectorizer)
    index: faiss.IndexFlatIP | None = None
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
        matrix = self.vectorizer.fit_transform(texts).astype(np.float32)
        vectors = matrix.toarray()
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.reset()
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[LegalDocument, float]]:
        if self.index is None:
            return []
        with SEARCH_LATENCY.time():
            query_vec = self.vectorizer.transform([query]).astype(np.float32).toarray()
            scores, indices = self.index.search(query_vec, top_k)
            results: List[Tuple[LegalDocument, float]] = []
            for idx, score in zip(indices.ravel(), scores.ravel()):
                if idx < 0:
                    continue
                results.append((self._docs[int(idx)], float(score)))
        SEARCH_REQUESTS.inc()
        return results

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise ValueError("Index is empty")
        faiss.write_index(self.index, str(path))
        meta_path = Path(path).with_suffix(".meta.json")
        docs = [
            {"id": d.id, "text": d.text, "metadata": d.metadata} for d in self._docs
        ]
        meta_path.write_text(json.dumps(docs), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        self.index = faiss.read_index(str(path))
        meta_path = Path(path).with_suffix(".meta.json")
        docs_data = json.loads(meta_path.read_text(encoding="utf-8"))
        self._docs = [LegalDocument(**d) for d in docs_data]
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit([d.text for d in self._docs])
