"""Tests for LegalDocumentIndexer."""

import json
import sys
import pathlib

import pytest
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lexgraph_legal_rag.indexer import (
    DocumentChunk,
    LegalDocumentIndexer,
)


SAMPLE_TEXT = """
Section 1. Definitions.
"Confidential Information" means any non-public technical or business information.

Section 2. Obligations.
The Receiving Party shall hold Confidential Information in strict confidence
and shall not disclose it to third parties without prior written consent.

Section 3. Term.
This Agreement shall remain in effect for three years from the effective date.
"""

SAMPLE_TEXT_2 = """
Article 1. Grant of License.
Licensor grants Licensee a non-exclusive license to use the Software.

Article 2. Restrictions.
Licensee shall not reverse engineer or decompile the Software.

Article 3. Warranty Disclaimer.
THE SOFTWARE IS PROVIDED AS IS WITHOUT WARRANTY OF ANY KIND.
"""


class TestDocumentChunk:
    def test_citation_format(self):
        chunk = DocumentChunk(
            doc_name="NDA", section="Section 1", text="foo", char_offset=42, chunk_index=0
        )
        assert "NDA" in chunk.citation
        assert "Section 1" in chunk.citation
        assert "42" in chunk.citation


class TestLegalDocumentIndexer:
    def test_index_single_document(self):
        indexer = LegalDocumentIndexer()
        n = indexer.index_document("test_doc", SAMPLE_TEXT)
        assert n > 0
        assert indexer.chunk_count == n

    def test_section_chunking(self):
        indexer = LegalDocumentIndexer()
        indexer.index_document("nda", SAMPLE_TEXT)
        # Should find section headings
        sections = [c.section for c in indexer._chunks]
        assert any("Section" in s or "preamble" in s.lower() for s in sections)

    def test_index_multiple_documents(self):
        indexer = LegalDocumentIndexer()
        docs = {"doc1": SAMPLE_TEXT, "doc2": SAMPLE_TEXT_2}
        total = indexer.index_documents(docs)
        assert total > 0
        assert indexer.chunk_count == total
        # Both docs should be represented
        doc_names = {c.doc_name for c in indexer._chunks}
        assert "doc1" in doc_names
        assert "doc2" in doc_names

    def test_faiss_index_built(self):
        indexer = LegalDocumentIndexer()
        indexer.index_documents({"doc": SAMPLE_TEXT})
        assert indexer._index is not None
        assert indexer._vectors is not None

    def test_vectors_are_normalized(self):
        indexer = LegalDocumentIndexer()
        indexer.index_documents({"doc": SAMPLE_TEXT})
        norms = np.linalg.norm(indexer._vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_save_and_load(self, tmp_path):
        indexer = LegalDocumentIndexer()
        indexer.index_documents({"nda": SAMPLE_TEXT, "license": SAMPLE_TEXT_2})
        index_path = tmp_path / "legal.faiss"
        indexer.save(index_path)

        # Check files written
        assert index_path.exists()
        assert (tmp_path / "legal.chunks.json").exists()

        # Load into fresh indexer
        indexer2 = LegalDocumentIndexer()
        indexer2.load(index_path)
        assert indexer2.chunk_count == indexer.chunk_count

        meta = json.loads((tmp_path / "legal.chunks.json").read_text())
        assert len(meta) == indexer.chunk_count

    def test_to_legal_documents(self):
        indexer = LegalDocumentIndexer()
        indexer.index_documents({"doc": SAMPLE_TEXT})
        docs = indexer.to_legal_documents()
        assert len(docs) == indexer.chunk_count
        for doc in docs:
            assert doc.id
            assert doc.text
            assert "doc_name" in doc.metadata
            assert "section" in doc.metadata
            assert "char_offset" in doc.metadata

    def test_large_section_sub_chunked(self):
        """Sections larger than chunk_size should be split into sub-chunks."""
        long_section = "Section 1. Very Long Clause.\n" + ("word " * 300)
        indexer = LegalDocumentIndexer(chunk_size=200, chunk_overlap=20)
        indexer.index_documents({"long_doc": long_section})
        # Should have multiple sub-chunks for the long section
        assert indexer.chunk_count > 1

    def test_empty_document(self):
        indexer = LegalDocumentIndexer()
        n = indexer.index_document("empty", "   ")
        assert n == 0

    def test_ensure_index_idempotent(self):
        indexer = LegalDocumentIndexer()
        indexer.index_documents({"doc": SAMPLE_TEXT})
        idx1 = indexer._index
        indexer.ensure_index()  # Should not rebuild
        assert indexer._index is idx1
