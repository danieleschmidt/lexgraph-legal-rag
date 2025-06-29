from lexgraph_legal_rag.document_pipeline import VectorIndex, LegalDocument
from lexgraph_legal_rag.metrics import SEARCH_REQUESTS


def test_search_metrics_increment():
    idx = VectorIndex()
    idx.add([LegalDocument(id="1", text="foo bar")])
    before = SEARCH_REQUESTS._value.get()
    idx.search("foo")
    after = SEARCH_REQUESTS._value.get()
    assert after == before + 1
