"""Tests for LegalRetriever and MultiAgentRetriever."""

import sys
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lexgraph_legal_rag.indexer import LegalDocumentIndexer
from lexgraph_legal_rag.retriever import (
    LegalRetriever,
    MultiAgentRetriever,
    RetrievedChunk,
    _expand_query,
)


NDA_TEXT = """
Section 1. Confidential Information.
Confidential Information means any non-public technical or business data.

Section 2. Obligations.
The Receiving Party shall maintain strict confidentiality and not disclose
any Confidential Information without prior written consent of the Disclosing Party.

Section 3. Term.
This Agreement remains in effect for three years from the execution date.
Upon termination, all Confidential Information must be returned or destroyed.
"""

LICENSE_TEXT = """
Article 1. License Grant.
Licensor grants a non-exclusive, non-transferable license to use the Software.

Article 2. Restrictions.
Licensee shall not reverse engineer, decompile, or sublicense the Software.

Article 3. Warranty Disclaimer.
THE SOFTWARE IS PROVIDED AS IS WITHOUT ANY WARRANTY. LICENSOR DISCLAIMS ALL
WARRANTIES INCLUDING MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

Article 4. Limitation of Liability.
IN NO EVENT SHALL LICENSOR BE LIABLE FOR CONSEQUENTIAL OR INDIRECT DAMAGES.
MAXIMUM LIABILITY SHALL NOT EXCEED FEES PAID IN THE PRECEDING TWELVE MONTHS.
"""

EMPLOYMENT_TEXT = """
Clause 1. Position.
Employee is hired as Senior Software Engineer.

Clause 2. Compensation.
Employee shall receive an annual salary of $120,000 payable bi-weekly.

Clause 3. Intellectual Property.
All inventions created during employment are assigned to Employer.
Employee waives all moral rights in work product.

Clause 4. Non-Compete.
Employee shall not compete with Employer for one year following termination.
"""


def make_indexer() -> LegalDocumentIndexer:
    indexer = LegalDocumentIndexer(chunk_size=300, chunk_overlap=30)
    indexer.index_documents({
        "NDA": NDA_TEXT,
        "SLA": LICENSE_TEXT,
        "Employment": EMPLOYMENT_TEXT,
    })
    return indexer


class TestRetrievedChunk:
    def test_citation_property(self):
        from lexgraph_legal_rag.indexer import DocumentChunk
        chunk = DocumentChunk(doc_name="Test", section="Sec 1", text="foo", char_offset=10, chunk_index=0)
        rc = RetrievedChunk(chunk=chunk, score=0.5)
        assert "Test" in rc.citation
        assert "Sec 1" in rc.citation
        assert rc.text == "foo"

    def test_repr(self):
        from lexgraph_legal_rag.indexer import DocumentChunk
        chunk = DocumentChunk(doc_name="NDA", section="§1", text="bar", char_offset=0, chunk_index=0)
        rc = RetrievedChunk(chunk=chunk, score=0.75)
        assert "0.75" in repr(rc)
        assert "NDA" in repr(rc)


class TestExpandQuery:
    def test_returns_at_least_original(self):
        variants = _expand_query("what is indemnification?")
        assert "what is indemnification?" in variants
        assert len(variants) >= 1

    def test_strips_question_words(self):
        variants = _expand_query("what is indemnification?")
        # Should include a keyword form
        assert len(variants) > 1

    def test_adds_legal_qualifier(self):
        variants = _expand_query("liability for damages")
        # Should add "legal" variant since query doesn't contain "legal"
        assert any("legal" in v.lower() for v in variants)

    def test_no_duplicates(self):
        variants = _expand_query("indemnification clause")
        assert len(variants) == len(set(variants))


class TestLegalRetriever:
    def test_retrieves_relevant_results(self):
        indexer = make_indexer()
        retriever = LegalRetriever(indexer=indexer, top_k=3)
        results = retriever.retrieve("confidentiality obligations")
        assert len(results) > 0
        # Top result should be from NDA
        assert results[0].chunk.doc_name == "NDA"

    def test_returns_results_sorted_by_score(self):
        indexer = make_indexer()
        retriever = LegalRetriever(indexer=indexer, top_k=5)
        results = retriever.retrieve("warranty disclaimer software")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        indexer = make_indexer()
        retriever = LegalRetriever(indexer=indexer, top_k=2)
        results = retriever.retrieve("indemnification")
        assert len(results) <= 2

    def test_empty_index_returns_empty(self):
        indexer = LegalDocumentIndexer()
        retriever = LegalRetriever(indexer=indexer, top_k=3)
        results = retriever.retrieve("some query")
        assert results == []

    def test_results_have_citations(self):
        indexer = make_indexer()
        retriever = LegalRetriever(indexer=indexer, top_k=3)
        results = retriever.retrieve("intellectual property")
        for r in results:
            assert r.citation
            assert r.chunk.doc_name
            assert r.chunk.section

    def test_retrieve_with_context_format(self):
        indexer = make_indexer()
        retriever = LegalRetriever(indexer=indexer, top_k=3)
        context = retriever.retrieve_with_context("license restrictions")
        assert "Query:" in context
        assert "Score:" in context

    def test_retrieve_with_context_no_results(self):
        indexer = LegalDocumentIndexer()
        retriever = LegalRetriever(indexer=indexer)
        result = retriever.retrieve_with_context("anything")
        assert "No relevant documents" in result

    def test_correct_document_for_specific_query(self):
        """Each query should return the most relevant document."""
        indexer = make_indexer()
        retriever = LegalRetriever(indexer=indexer, top_k=3)

        # Warranty question → SLA
        results = retriever.retrieve("warranty disclaimer no warranty")
        assert results[0].chunk.doc_name == "SLA"

        # IP assignment → Employment
        results = retriever.retrieve("inventions created during employment assigned")
        assert results[0].chunk.doc_name == "Employment"

    def test_min_score_filter(self):
        indexer = make_indexer()
        retriever = LegalRetriever(indexer=indexer, top_k=10, min_score=0.5)
        results = retriever.retrieve("confidential information obligations")
        # All results should be above min_score
        for r in results:
            assert r.score >= 0.5


class TestMultiAgentRetriever:
    def test_returns_results(self):
        indexer = make_indexer()
        mar = MultiAgentRetriever(indexer=indexer, top_k=3)
        results = mar.retrieve("software liability limitations")
        assert len(results) > 0

    def test_results_deduplicated(self):
        """Each (doc_name, section) pair should appear at most once."""
        indexer = make_indexer()
        mar = MultiAgentRetriever(indexer=indexer, top_k=5)
        results = mar.retrieve("confidential information obligations")
        keys = [(r.chunk.doc_name, r.chunk.section) for r in results]
        assert len(keys) == len(set(keys))

    def test_scores_sorted_descending(self):
        indexer = make_indexer()
        mar = MultiAgentRetriever(indexer=indexer, top_k=5)
        results = mar.retrieve("intellectual property rights")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_multi_agent_vs_single_agent(self):
        """MultiAgentRetriever should return at least as many results as single agent."""
        indexer = make_indexer()
        retriever = LegalRetriever(indexer=indexer, top_k=3)
        mar = MultiAgentRetriever(indexer=indexer, top_k=3)

        single_results = retriever.retrieve("salary compensation payment")
        multi_results = mar.retrieve("salary compensation payment")
        # Multi-agent should return >= single agent results (dedup across variants)
        assert len(multi_results) >= len(single_results)

    def test_empty_index_returns_empty(self):
        indexer = LegalDocumentIndexer()
        mar = MultiAgentRetriever(indexer=indexer, top_k=3)
        results = mar.retrieve("anything")
        assert results == []

    def test_retrieve_with_context_format(self):
        indexer = make_indexer()
        mar = MultiAgentRetriever(indexer=indexer, top_k=3)
        context = mar.retrieve_with_context("non-compete clause")
        assert "Multi-Agent Query:" in context
        assert "Score:" in context

    def test_parallel_execution(self):
        """Should complete without errors even with multiple workers."""
        indexer = make_indexer()
        mar = MultiAgentRetriever(indexer=indexer, top_k=3, max_workers=4)
        results = mar.retrieve("termination and breach of contract")
        assert isinstance(results, list)
