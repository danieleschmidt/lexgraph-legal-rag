#!/usr/bin/env python3
"""Document ingestion module for LexGraph Legal RAG system."""

import argparse
from pathlib import Path
import sys
from typing import Optional

from .document_pipeline import LegalDocumentPipeline
from .logging_config import configure_logging
from .metrics import start_metrics_server
from .config import validate_environment


def ingest_documents(
    docs_path: Path,
    index_path: Path,
    enable_semantic: bool = False,
    chunk_size: int = 512,
    metrics_port: Optional[int] = None
) -> int:
    """Ingest documents and create vector index.
    
    Args:
        docs_path: Path to directory containing legal documents
        index_path: Path where the index will be saved
        enable_semantic: Whether to enable semantic search
        chunk_size: Size of text chunks for indexing
        metrics_port: Port for metrics server
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Start metrics server if requested
        if metrics_port:
            start_metrics_server(metrics_port)
        
        # Validate inputs
        if not docs_path.exists():
            print(f"Error: Documents directory {docs_path} does not exist", file=sys.stderr)
            return 1
        
        if not docs_path.is_dir():
            print(f"Error: {docs_path} is not a directory", file=sys.stderr)
            return 1
        
        # Initialize pipeline
        pipeline = LegalDocumentPipeline(use_semantic=enable_semantic)
        
        print(f"Starting document ingestion...")
        print(f"  Source: {docs_path}")
        print(f"  Index: {index_path}")
        print(f"  Semantic search: {'enabled' if enable_semantic else 'disabled'}")
        print(f"  Chunk size: {chunk_size}")
        
        # Ingest documents
        num_documents = pipeline.ingest_folder(docs_path, chunk_size=chunk_size)
        
        # Save index
        pipeline.save_index(index_path)
        
        # Get statistics
        stats = pipeline.get_index_stats()
        
        print(f"\nIngestion completed successfully:")
        print(f"  Documents processed: {num_documents}")
        print(f"  Text chunks: {stats.get('chunk_count', 0)}")
        print(f"  Vector dimensions: {stats.get('vector_dim', 0)}")
        print(f"  Index size: {stats.get('index_size_mb', 0):.1f} MB")
        print(f"  Index saved to: {index_path}")
        
        if enable_semantic:
            semantic_path = index_path.with_suffix('.sem')
            if semantic_path.exists():
                print(f"  Semantic index: {semantic_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error during document ingestion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for ingest command."""
    parser = argparse.ArgumentParser(
        description="Ingest legal documents and create searchable index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ingestion
  lexgraph-ingest --docs ./legal_docs --index legal.bin

  # With semantic search enabled
  lexgraph-ingest --docs ./contracts --index contracts.bin --semantic

  # Custom chunk size with metrics
  lexgraph-ingest --docs ./docs --index docs.bin --chunk-size 256 --metrics-port 9090
        """
    )
    
    parser.add_argument(
        "--docs", 
        required=True,
        help="Directory containing legal documents to index"
    )
    parser.add_argument(
        "--index", 
        default="index.bin",
        help="Output path for the vector index file"
    )
    parser.add_argument(
        "--semantic", 
        action="store_true",
        help="Enable semantic search indexing (requires additional processing)"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=512,
        help="Text chunk size for indexing (default: 512)"
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        help="Port to expose Prometheus metrics server"
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level=args.log_level)
    
    # Validate environment (allow test mode for ingestion)
    try:
        validate_environment(allow_test_mode=True)
    except SystemExit:
        print("Warning: Configuration validation failed, proceeding anyway", file=sys.stderr)
    
    # Convert paths
    docs_path = Path(args.docs)
    index_path = Path(args.index)
    
    # Run ingestion
    return ingest_documents(
        docs_path=docs_path,
        index_path=index_path,
        enable_semantic=args.semantic,
        chunk_size=args.chunk_size,
        metrics_port=args.metrics_port
    )


if __name__ == "__main__":
    sys.exit(main())