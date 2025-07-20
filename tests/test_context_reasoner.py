import asyncio

from lexgraph_legal_rag.context_reasoning import ContextAwareReasoner
from lexgraph_legal_rag.document_pipeline import LegalDocument


def test_reason_without_documents():
    reasoner = ContextAwareReasoner()
    result = asyncio.run(reasoner.reason("please explain arbitration clause"))
    # The multi-agent system now provides intelligent legal explanations
    assert "arbitration" in result.lower()
    assert "dispute resolution" in result.lower() or "legal" in result.lower()
    assert len(result) > 10  # Should provide meaningful explanation


def test_reason_with_citations_sync():
    reasoner = ContextAwareReasoner()
    docs = [
        LegalDocument(
            id="1", text="arbitration clause text", metadata={"path": "doc1.txt"}
        )
    ]
    reasoner.pipeline.index.add(docs)
    chunks = list(reasoner.reason_with_citations_sync("arbitration clause"))
    assert chunks[0]
    assert chunks[-1].startswith("Citations:\n")
    assert "1:" in chunks[-1]
