from lexgraph_legal_rag.document_pipeline import LegalDocumentPipeline, LegalDocument


def test_semantic_index_save_and_load(tmp_path):
    docs = [
        LegalDocument(id="1", text="arbitration clause"),
        LegalDocument(id="2", text="indemnification clause"),
    ]
    pipeline = LegalDocumentPipeline(use_semantic=True)
    pipeline.index.add(docs)
    pipeline.semantic.ingest(docs)
    save_path = tmp_path / "index.bin"
    pipeline.save_index(save_path)

    new_pipeline = LegalDocumentPipeline(use_semantic=False)
    new_pipeline.load_index(save_path)
    results = new_pipeline.search("arbitration", semantic=True)
    assert results
    assert results[0][0].id == "1"
