import asyncio

from lexgraph_legal_rag.context_reasoning import ContextAwareReasoner
from lexgraph_legal_rag.document_pipeline import LegalDocument


def test_reason_without_documents():
    reasoner = ContextAwareReasoner()
    result = asyncio.run(reasoner.reason("please explain arbitration clause"))
    assert (
        result
        == "explanation of summary of retrieved: please explain arbitration clause"
    )


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
