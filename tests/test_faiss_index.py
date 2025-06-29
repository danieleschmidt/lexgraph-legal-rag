from lexgraph_legal_rag.faiss_index import FaissVectorIndex
from lexgraph_legal_rag.models import LegalDocument


def test_faiss_index_add_and_search(tmp_path):
    docs = [
        LegalDocument(id="1", text="alpha beta"),
        LegalDocument(id="2", text="beta gamma"),
        LegalDocument(id="3", text="gamma delta"),
    ]
    idx = FaissVectorIndex()
    idx.add(docs)
    results = idx.search("beta", top_k=2)
    assert results
    ids = [d.id for d, _ in results]
    assert "1" in ids and "2" in ids

    file = tmp_path / "index.faiss"
    idx.save(file)

    new_idx = FaissVectorIndex()
    new_idx.load(file)
    loaded_results = new_idx.search("delta", top_k=1)
    assert loaded_results[0][0].id == "3"
