import argparse
from pathlib import Path

from lexgraph_legal_rag.context_reasoning import ContextAwareReasoner
from lexgraph_legal_rag.logging_config import configure_logging
from lexgraph_legal_rag.metrics import start_metrics_server
from lexgraph_legal_rag.config import validate_environment


def main() -> None:
    configure_logging()
    
    # Validate configuration at startup
    config = validate_environment()
    parser = argparse.ArgumentParser(description="Query the agent")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--index", default="index.bin", help="Path to load the index")
    parser.add_argument(
        "--hops", type=int, default=3, help="Number of context documents"
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Expose Prometheus metrics on the given port",
    )
    args = parser.parse_args()
    start_metrics_server(args.metrics_port)

    reasoner = ContextAwareReasoner()
    if Path(args.index).exists():
        reasoner.pipeline.load_index(args.index)
    else:
        print(f"Index {args.index} not found; proceeding without documents")

    for chunk in reasoner.reason_with_citations_sync(args.query, hops=args.hops):
        print(chunk)


if __name__ == "__main__":
    main()
