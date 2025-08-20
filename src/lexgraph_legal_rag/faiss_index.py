"""FAISS-based vector index for scalable similarity search."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .metrics import SEARCH_LATENCY
from .metrics import SEARCH_REQUESTS
from .models import LegalDocument


class FaissIndexPool:
    """Thread-safe pool of FAISS indices for concurrent access."""

    def __init__(self, max_pool_size: int = 3) -> None:
        self.max_pool_size = max_pool_size
        self._available_indices: list[faiss.IndexFlatIP] = []
        self._in_use_indices: set = set()
        self._lock = threading.Lock()
        self._master_index: faiss.IndexFlatIP | None = None
        self._last_sync_time = 0.0

    def set_master_index(self, index: faiss.IndexFlatIP) -> None:
        """Set the master index that will be cloned for the pool."""
        with self._lock:
            self._master_index = index
            # Clear existing pool
            self._available_indices.clear()
            self._in_use_indices.clear()
            self._last_sync_time = time.time()

    def get_index(self) -> faiss.IndexFlatIP:
        """Get an index from the pool, creating one if necessary."""
        with self._lock:
            if self._available_indices:
                index = self._available_indices.pop()
                self._in_use_indices.add(id(index))
                return index
            elif self._master_index is not None:
                # Clone the master index for concurrent use
                cloned = faiss.clone_index(self._master_index)
                self._in_use_indices.add(id(cloned))
                return cloned
            else:
                raise RuntimeError("No master index available for cloning")

    def return_index(self, index: faiss.IndexFlatIP) -> None:
        """Return an index to the pool."""
        with self._lock:
            index_id = id(index)
            if index_id in self._in_use_indices:
                self._in_use_indices.remove(index_id)
                if len(self._available_indices) < self.max_pool_size:
                    self._available_indices.append(index)

    def get_pool_stats(self) -> dict:
        """Get statistics about the index pool."""
        with self._lock:
            return {
                "available": len(self._available_indices),
                "in_use": len(self._in_use_indices),
                "max_size": self.max_pool_size,
                "has_master": self._master_index is not None,
            }


@dataclass
class FaissVectorIndex:
    """Vector index using FAISS for nearest-neighbor search with connection pooling."""

    vectorizer: TfidfVectorizer = field(default_factory=TfidfVectorizer)
    index: faiss.IndexFlatIP | None = None
    _docs: list[LegalDocument] = field(default_factory=list)
    _index_pool: FaissIndexPool = field(
        default_factory=lambda: FaissIndexPool(max_pool_size=3)
    )
    _use_pool: bool = True

    @property
    def documents(self) -> list[LegalDocument]:
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

        # Update the connection pool with the new index
        if self._use_pool:
            self._index_pool.set_master_index(self.index)

    def search(self, query: str, top_k: int = 5) -> list[tuple[LegalDocument, float]]:
        if self.index is None:
            return []

        # Use connection pool for concurrent access if enabled
        if self._use_pool:
            return self._search_with_pool(query, top_k)
        else:
            return self._search_direct(query, top_k)

    def _search_direct(
        self, query: str, top_k: int
    ) -> list[tuple[LegalDocument, float]]:
        """Direct search without connection pooling."""
        with SEARCH_LATENCY.labels(search_type="faiss").time():
            query_vec = self.vectorizer.transform([query]).astype(np.float32).toarray()
            scores, indices = self.index.search(query_vec, top_k)
            results: list[tuple[LegalDocument, float]] = []
            for idx, score in zip(indices.ravel(), scores.ravel()):
                if idx < 0:
                    continue
                results.append((self._docs[int(idx)], float(score)))
        SEARCH_REQUESTS.labels(search_type="faiss").inc()
        return results

    def _search_with_pool(
        self, query: str, top_k: int
    ) -> list[tuple[LegalDocument, float]]:
        """Search using connection pool for thread safety."""
        pooled_index = None
        try:
            pooled_index = self._index_pool.get_index()
            with SEARCH_LATENCY.labels(search_type="faiss_pooled").time():
                query_vec = (
                    self.vectorizer.transform([query]).astype(np.float32).toarray()
                )
                scores, indices = pooled_index.search(query_vec, top_k)
                results: list[tuple[LegalDocument, float]] = []
                for idx, score in zip(indices.ravel(), scores.ravel()):
                    if idx < 0:
                        continue
                    results.append((self._docs[int(idx)], float(score)))
            SEARCH_REQUESTS.labels(search_type="faiss_pooled").inc()
            return results
        finally:
            if pooled_index is not None:
                self._index_pool.return_index(pooled_index)

    def batch_search(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[tuple[LegalDocument, float]]]:
        """Perform batch search for multiple queries efficiently."""
        if self.index is None:
            return [[] for _ in queries]

        pooled_index = None
        try:
            pooled_index = (
                self._index_pool.get_index() if self._use_pool else self.index
            )
            results = []

            with SEARCH_LATENCY.labels(search_type="faiss_batch").time():
                # Transform all queries at once
                query_vecs = (
                    self.vectorizer.transform(queries).astype(np.float32).toarray()
                )

                # Perform batch search
                all_scores, all_indices = pooled_index.search(query_vecs, top_k)

                for i in range(len(queries)):
                    query_results: list[tuple[LegalDocument, float]] = []
                    for idx, score in zip(all_indices[i], all_scores[i]):
                        if idx < 0:
                            continue
                        query_results.append((self._docs[int(idx)], float(score)))
                    results.append(query_results)

            SEARCH_REQUESTS.labels(search_type="faiss_batch").inc(len(queries))
            return results
        finally:
            if pooled_index is not None and self._use_pool:
                self._index_pool.return_index(pooled_index)

    def get_pool_stats(self) -> dict:
        """Get connection pool statistics."""
        if self._use_pool:
            return self._index_pool.get_pool_stats()
        return {"pool_enabled": False}

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

        # Initialize the connection pool with the loaded index
        if self._use_pool:
            self._index_pool.set_master_index(self.index)
