"""LegalDocumentIndexer — chunks legal documents and indexes them into FAISS."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import LegalDocument


logger = logging.getLogger(__name__)

# Regex patterns for common legal section headers
_SECTION_PATTERNS = [
    re.compile(r"^\s*(section|article|clause|part|exhibit|schedule)\s+[\d.]+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*[\d]+\.\s+[A-Z]", re.MULTILINE),
    re.compile(r"^\s*[A-Z][A-Z\s]{3,}$", re.MULTILINE),  # ALL CAPS headings
    re.compile(r"^\s*WHEREAS[,\s]", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*NOW,?\s+THEREFORE", re.MULTILINE | re.IGNORECASE),
]


@dataclass
class DocumentChunk:
    """A chunk of a legal document with citation metadata."""

    doc_name: str
    section: str
    text: str
    char_offset: int
    chunk_index: int

    @property
    def citation(self) -> str:
        return f"{self.doc_name} § {self.section} (offset {self.char_offset})"


@dataclass
class LegalDocumentIndexer:
    """
    Chunks legal documents by section/clause and indexes them in FAISS.

    Uses TF-IDF embeddings (char-trigram weighted) — no external API calls.
    Serves as the indexing layer for DocGraph's retrieval pipeline.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=8192,
            sublinear_tf=True,
        )
    )
    _chunks: List[DocumentChunk] = field(default_factory=list)
    _index: Optional[faiss.IndexFlatIP] = field(default=None, init=False)
    _vectors: Optional[np.ndarray] = field(default=None, init=False)

    def _split_by_section(self, text: str, doc_name: str) -> List[DocumentChunk]:
        """Split document text at section boundaries, fall back to fixed-size chunks."""
        # Find all section boundaries
        boundaries = [0]
        section_names: dict[int, str] = {0: "preamble"}

        for pattern in _SECTION_PATTERNS:
            for m in pattern.finditer(text):
                pos = m.start()
                if pos not in boundaries:
                    boundaries.append(pos)
                    section_names[pos] = m.group(0).strip()[:60]

        boundaries = sorted(set(boundaries))
        boundaries.append(len(text))

        chunks = []
        chunk_idx = 0
        for i, start in enumerate(boundaries[:-1]):
            end = boundaries[i + 1]
            section_text = text[start:end].strip()
            if not section_text:
                continue

            section_label = section_names.get(start, f"section_{i}")

            # If the section is small enough, use it as one chunk
            if len(section_text) <= self.chunk_size:
                chunks.append(DocumentChunk(
                    doc_name=doc_name,
                    section=section_label,
                    text=section_text,
                    char_offset=start,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1
            else:
                # Sub-chunk large sections with overlap
                pos = 0
                sub = 0
                while pos < len(section_text):
                    end_pos = min(pos + self.chunk_size, len(section_text))
                    chunk_text = section_text[pos:end_pos].strip()
                    if chunk_text:
                        chunks.append(DocumentChunk(
                            doc_name=doc_name,
                            section=f"{section_label}[{sub}]",
                            text=chunk_text,
                            char_offset=start + pos,
                            chunk_index=chunk_idx,
                        ))
                        chunk_idx += 1
                        sub += 1
                    pos += self.chunk_size - self.chunk_overlap
                    if pos >= len(section_text):
                        break

        return chunks

    def index_document(self, doc_name: str, text: str) -> int:
        """Index a single document. Returns the number of chunks created."""
        chunks = self._split_by_section(text, doc_name)
        self._chunks.extend(chunks)
        logger.info("Indexed %d chunks from '%s'", len(chunks), doc_name)
        # Invalidate the current index — will rebuild on next search
        self._index = None
        self._vectors = None
        return len(chunks)

    def index_documents(self, documents: dict[str, str]) -> int:
        """Index multiple documents. Returns total chunk count."""
        total = 0
        for name, text in documents.items():
            total += self.index_document(name, text)
        self._build_index()
        return total

    def _build_index(self) -> None:
        """Build FAISS index from current chunks."""
        if not self._chunks:
            logger.warning("No chunks to index")
            return

        texts = [c.text for c in self._chunks]
        matrix = self.vectorizer.fit_transform(texts).astype(np.float32).toarray()

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        self._vectors = matrix
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)
        logger.info("Built FAISS index: %d vectors, dim=%d", len(self._chunks), matrix.shape[1])

    def ensure_index(self) -> None:
        """Build index if not already built."""
        if self._index is None and self._chunks:
            self._build_index()

    def save(self, path: str | Path) -> None:
        """Save index and chunk metadata to disk."""
        import json
        path = Path(path)
        if self._index is None:
            self._build_index()
        faiss.write_index(self._index, str(path))
        meta = [
            {
                "doc_name": c.doc_name,
                "section": c.section,
                "text": c.text,
                "char_offset": c.char_offset,
                "chunk_index": c.chunk_index,
            }
            for c in self._chunks
        ]
        path.with_suffix(".chunks.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load index and chunk metadata from disk."""
        import json
        path = Path(path)
        self._index = faiss.read_index(str(path))
        meta = json.loads(path.with_suffix(".chunks.json").read_text(encoding="utf-8"))
        self._chunks = [DocumentChunk(**m) for m in meta]
        # Rebuild vectorizer from loaded texts
        self.vectorizer.fit([c.text for c in self._chunks])

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def to_legal_documents(self) -> List[LegalDocument]:
        """Convert chunks to LegalDocument objects for pipeline compatibility."""
        return [
            LegalDocument(
                id=f"{c.doc_name}::{c.section}",
                text=c.text,
                metadata={
                    "doc_name": c.doc_name,
                    "section": c.section,
                    "char_offset": c.char_offset,
                    "chunk_index": c.chunk_index,
                },
            )
            for c in self._chunks
        ]
