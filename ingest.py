import argparse
from pathlib import Path

from lexgraph_legal_rag.document_pipeline import LegalDocumentPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest legal documents")
    parser.add_argument(
        "--docs", required=True, help="Folder containing text documents"
    )
    parser.add_argument("--index", default="index.bin", help="Path to save the index")
    parser.add_argument(
        "--semantic", action="store_true", help="Enable semantic search"
    )
    args = parser.parse_args()

    pipeline = LegalDocumentPipeline(use_semantic=args.semantic)
    docs_path = Path(args.docs)
    pipeline.ingest_folder(docs_path)
    index_path = Path(args.index)
    pipeline.save_index(index_path)
    print(f"Indexed documents from {docs_path} to {index_path}")


if __name__ == "__main__":
    main()
