from lexgraph_legal_rag.document_pipeline import VectorIndex, LegalDocument


def test_vector_index_save_and_load(tmp_path):
    docs = [
        LegalDocument(id="1", text="arbitration clause"),
        LegalDocument(id="2", text="indemnification clause"),
    ]
    index = VectorIndex()
    index.add(docs)
    save_path = tmp_path / "index.bin"
    index.save(save_path)

    new_index = VectorIndex()
    new_index.load(save_path)
    results = new_index.search("arbitration")
    assert results
    assert results[0][0].id == "1"
