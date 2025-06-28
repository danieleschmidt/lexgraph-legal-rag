from lexgraph_legal_rag.document_pipeline import VectorIndex
from lexgraph_legal_rag.semantic_search import SemanticSearchPipeline


def test_vector_index_search_empty():
    index = VectorIndex()
    assert index.search("anything") == []


def test_semantic_pipeline_search_empty():
    pipeline = SemanticSearchPipeline()
    assert pipeline.search("anything") == []
