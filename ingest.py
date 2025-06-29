import argparse
from pathlib import Path

from lexgraph_legal_rag.document_pipeline import LegalDocumentPipeline
from lexgraph_legal_rag.logging_config import configure_logging
from lexgraph_legal_rag.metrics import start_metrics_server


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest legal documents")
    parser.add_argument(
        "--docs", required=True, help="Folder containing text documents"
    )
    parser.add_argument("--index", default="index.bin", help="Path to save the index")
    parser.add_argument(
        "--semantic", action="store_true", help="Enable semantic search"
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Expose Prometheus metrics on the given port",
    )
    args = parser.parse_args()
    start_metrics_server(args.metrics_port)

    pipeline = LegalDocumentPipeline(use_semantic=args.semantic)
    docs_path = Path(args.docs)
    pipeline.ingest_folder(docs_path)
    index_path = Path(args.index)
    pipeline.save_index(index_path)
    print(f"Indexed documents from {docs_path} to {index_path}")


if __name__ == "__main__":
    main()
