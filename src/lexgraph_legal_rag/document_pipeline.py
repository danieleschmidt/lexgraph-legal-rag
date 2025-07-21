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
from .cache import get_query_cache


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
        
        # Invalidate cache when index is updated
        from .cache import get_query_cache
        cache = get_query_cache()
        cache.invalidate_pattern("*")  # Invalidate all cached results
        
        logger.debug("Indexed %d documents and invalidated cache", len(docs))

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
        with SEARCH_LATENCY.labels(search_type="vector").time():
            query_vec = self._vectorizer.transform([query])
            # Use more efficient dot product calculation
            scores = (self._matrix @ query_vec.T).toarray().ravel()
            if not len(scores):
                return []
            
            # Use partial sort for better performance when top_k << total_docs
            if top_k < len(scores) // 10:  # Use partial sort when top_k is much smaller
                import numpy as np
                # Get indices of top_k largest elements without full sort
                indices = np.argpartition(scores, -top_k)[-top_k:]
                # Sort only the top_k elements
                indices = indices[np.argsort(scores[indices])[::-1]]
            else:
                indices = scores.argsort()[::-1][:top_k]
            
            results = [(self._docs[i], float(scores[i])) for i in indices]
        SEARCH_REQUESTS.labels(search_type="vector").inc()
        logger.debug("Search for '%s' returned %d results", query, len(results))
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[LegalDocument, float]]]:
        """Perform batch search for multiple queries efficiently."""
        if self._matrix is None:
            return [[] for _ in queries]
        
        results = []
        with SEARCH_LATENCY.labels(search_type="vector_batch").time():
            # Transform all queries at once for better efficiency
            query_vecs = self._vectorizer.transform(queries)
            # Compute all similarity scores in one batch operation
            all_scores = (self._matrix @ query_vecs.T).toarray()
            
            for i, query in enumerate(queries):
                scores = all_scores[:, i]
                if not len(scores):
                    results.append([])
                    continue
                
                # Use partial sort optimization
                if top_k < len(scores) // 10:
                    import numpy as np
                    indices = np.argpartition(scores, -top_k)[-top_k:]
                    indices = indices[np.argsort(scores[indices])[::-1]]
                else:
                    indices = scores.argsort()[::-1][:top_k]
                
                query_results = [(self._docs[idx], float(scores[idx])) for idx in indices]
                results.append(query_results)
        
        SEARCH_REQUESTS.labels(search_type="vector_batch").inc(len(queries))
        logger.debug("Batch search for %d queries completed", len(queries))
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
        self, query: str, top_k: int = 5, semantic: bool | None = None, use_cache: bool = True
    ) -> List[Tuple[LegalDocument, float]]:
        # Determine which search method to use
        use_semantic = (semantic or semantic is None and self.semantic) and self.semantic
        
        # Try cache first if enabled
        if use_cache:
            cache = get_query_cache()
            cached_results = cache.get(
                query=query,
                top_k=top_k,
                semantic=use_semantic,
            )
            if cached_results is not None:
                logger.debug("Retrieved search results from cache")
                SEARCH_REQUESTS.labels(search_type="cached").inc()
                return cached_results
        
        # Perform actual search
        search_type = "semantic" if use_semantic else "vector"
        if use_semantic:
            results = self.semantic.search(query, top_k=top_k)
        else:
            results = self.index.search(query, top_k=top_k)
        
        # Cache results if caching is enabled
        if use_cache and results:
            cache = get_query_cache()
            cache.put(
                query=query,
                top_k=top_k,
                results=results,
                semantic=use_semantic,
            )
        
        return results
    
    def batch_search(
        self, queries: List[str], top_k: int = 5, semantic: bool | None = None, use_cache: bool = True
    ) -> List[List[Tuple[LegalDocument, float]]]:
        """Perform batch search for multiple queries efficiently.
        
        This method is optimized to reduce the N+1 query pattern by batching
        multiple search requests into a single operation.
        """
        if not queries:
            return []
        
        # Determine which search method to use
        use_semantic = (semantic or semantic is None and self.semantic) and self.semantic
        
        # Check cache for all queries if enabled
        cached_results = []
        uncached_queries = []
        uncached_indices = []
        
        if use_cache:
            cache = get_query_cache()
            for i, query in enumerate(queries):
                cached_result = cache.get(
                    query=query,
                    top_k=top_k,
                    semantic=use_semantic,
                )
                if cached_result is not None:
                    cached_results.append((i, cached_result))
                else:
                    uncached_queries.append(query)
                    uncached_indices.append(i)
        else:
            uncached_queries = queries
            uncached_indices = list(range(len(queries)))
        
        # Perform batch search for uncached queries
        batch_results = []
        if uncached_queries:
            if use_semantic:
                # Semantic search batch processing (if implemented)
                if hasattr(self.semantic, 'batch_search'):
                    batch_results = self.semantic.batch_search(uncached_queries, top_k=top_k)
                else:
                    # Fallback to individual searches
                    batch_results = [self.semantic.search(query, top_k=top_k) for query in uncached_queries]
            else:
                # Vector search batch processing
                batch_results = self.index.batch_search(uncached_queries, top_k=top_k)
            
            # Cache the new results
            if use_cache:
                cache = get_query_cache()
                for query, results in zip(uncached_queries, batch_results):
                    if results:
                        cache.put(
                            query=query,
                            top_k=top_k,
                            semantic=use_semantic,
                            results=results
                        )
        
        # Combine cached and new results in original order
        final_results = [[] for _ in queries]
        
        # Fill in cached results
        for orig_idx, cached_result in cached_results:
            final_results[orig_idx] = cached_result
        
        # Fill in new batch results
        for batch_idx, orig_idx in enumerate(uncached_indices):
            if batch_idx < len(batch_results):
                final_results[orig_idx] = batch_results[batch_idx]
        
        logger.debug("Batch search completed for %d queries (%d cached, %d new)", 
                    len(queries), len(cached_results), len(uncached_queries))
        
        return final_results

    @property
    def documents(self) -> List[LegalDocument]:
        return self.index.documents
