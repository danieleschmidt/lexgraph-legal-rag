#!/usr/bin/env python3
"""Command-line interface for LexGraph Legal RAG system."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from .config import validate_environment
from .context_reasoning import ContextAwareReasoner
from .document_pipeline import LegalDocumentPipeline
from .logging_config import configure_logging
from .metrics import start_metrics_server
from .multi_agent import MultiAgentGraph


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LexGraph Legal RAG - Multi-agent legal document analysis system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index legal documents
  lexgraph ingest --docs ./legal_corpus --index legal.bin

  # Query with citations
  lexgraph query "What constitutes breach of contract?" --index legal.bin

  # Start API server
  lexgraph serve --port 8000

  # Run with metrics monitoring
  lexgraph query "liability clauses" --metrics-port 9090
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Query command
    query_parser = subparsers.add_parser(
        'query', help='Query legal documents with multi-agent analysis'
    )
    query_parser.add_argument(
        'query', 
        help='Legal query to analyze'
    )
    query_parser.add_argument(
        '--index', 
        default='index.bin',
        help='Path to the vector index file'
    )
    query_parser.add_argument(
        '--hops', 
        type=int, 
        default=3,
        help='Number of context reasoning hops'
    )
    query_parser.add_argument(
        '--top-k', 
        type=int, 
        default=5,
        help='Number of documents to retrieve'
    )
    query_parser.add_argument(
        '--citations', 
        action='store_true',
        help='Include detailed citations in output'
    )
    query_parser.add_argument(
        '--format', 
        choices=['text', 'json'], 
        default='text',
        help='Output format'
    )

    # Ingest command
    ingest_parser = subparsers.add_parser(
        'ingest', help='Index legal documents for search'
    )
    ingest_parser.add_argument(
        '--docs', 
        required=True,
        help='Path to directory containing legal documents'
    )
    ingest_parser.add_argument(
        '--index', 
        default='index.bin',
        help='Output path for the vector index'
    )
    ingest_parser.add_argument(
        '--semantic', 
        action='store_true',
        help='Enable semantic search indexing'
    )
    ingest_parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=512,
        help='Text chunk size for indexing'
    )

    # Serve command
    serve_parser = subparsers.add_parser(
        'serve', help='Start the FastAPI server'
    )
    serve_parser.add_argument(
        '--host', 
        default='0.0.0.0',
        help='Host to bind the server'
    )
    serve_parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help='Port to bind the server'
    )
    serve_parser.add_argument(
        '--workers', 
        type=int, 
        default=1,
        help='Number of worker processes'
    )
    serve_parser.add_argument(
        '--reload', 
        action='store_true',
        help='Enable auto-reload in development'
    )

    # Global options
    parser.add_argument(
        '--metrics-port',
        type=int,
        help='Port for Prometheus metrics server'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--config-file',
        help='Path to configuration file'
    )

    return parser


async def handle_query_command(args) -> int:
    """Handle the query command."""
    import json
    
    try:
        # Initialize the reasoning system
        reasoner = ContextAwareReasoner()
        
        # Load index if it exists
        if Path(args.index).exists():
            reasoner.pipeline.load_index(args.index)
            print(f"Loaded index from {args.index}")
        else:
            print(f"Warning: Index {args.index} not found, proceeding without documents")
        
        # Execute query
        print(f"\nQuery: {args.query}")
        print("-" * 50)
        
        if args.citations:
            # Stream response with citations
            if args.format == 'json':
                results = []
                async for chunk in reasoner.reason_with_citations(args.query, hops=args.hops):
                    results.append(chunk)
                    print(json.dumps({"chunk": chunk}, indent=2))
                
                # Final JSON output
                print(json.dumps({
                    "query": args.query,
                    "response": "".join(results),
                    "hops": args.hops,
                    "index": args.index
                }, indent=2))
            else:
                async for chunk in reasoner.reason_with_citations(args.query, hops=args.hops):
                    print(chunk, end='', flush=True)
                print()  # Final newline
        else:
            # Simple response without citations
            response = await reasoner.reason(args.query, hops=args.hops)
            
            if args.format == 'json':
                print(json.dumps({
                    "query": args.query,
                    "response": response,
                    "hops": args.hops,
                    "index": args.index
                }, indent=2))
            else:
                print(response)
        
        return 0
        
    except Exception as e:
        print(f"Error executing query: {e}", file=sys.stderr)
        return 1


def handle_ingest_command(args) -> int:
    """Handle the ingest command."""
    try:
        from .document_pipeline import LegalDocumentPipeline
        
        docs_path = Path(args.docs)
        if not docs_path.exists():
            print(f"Error: Documents directory {args.docs} does not exist", file=sys.stderr)
            return 1
        
        # Initialize pipeline
        pipeline = LegalDocumentPipeline()
        
        print(f"Indexing documents from {args.docs}...")
        print(f"Chunk size: {args.chunk_size}")
        print(f"Semantic search: {'enabled' if args.semantic else 'disabled'}")
        
        # Index documents
        pipeline.ingest_directory(
            docs_path, 
            chunk_size=args.chunk_size,
            enable_semantic=args.semantic
        )
        
        # Save index
        pipeline.save_index(args.index)
        print(f"Index saved to {args.index}")
        
        # Print statistics
        stats = pipeline.get_index_stats()
        print(f"\nIndexing complete:")
        print(f"  Documents processed: {stats.get('document_count', 0)}")
        print(f"  Text chunks: {stats.get('chunk_count', 0)}")
        print(f"  Index size: {stats.get('index_size_mb', 0):.1f} MB")
        
        return 0
        
    except Exception as e:
        print(f"Error during indexing: {e}", file=sys.stderr)
        return 1


def handle_serve_command(args) -> int:
    """Handle the serve command."""
    try:
        import uvicorn
        from .api import create_api
        
        # Validate environment
        config = validate_environment()
        
        # Create FastAPI app
        app = create_api()
        
        print(f"Starting LexGraph Legal RAG API server...")
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"Workers: {args.workers}")
        print(f"Docs: http://{args.host}:{args.port}/docs")
        
        # Start server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=args.log_level.lower()
        )
        
        return 0
        
    except ImportError as e:
        print(f"Error: uvicorn not installed. Install with: pip install uvicorn", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level=getattr(args, 'log_level', 'INFO'))
    
    # Start metrics server if requested
    if getattr(args, 'metrics_port', None):
        start_metrics_server(args.metrics_port)
    
    # Validate environment (allow test mode for CLI usage)
    try:
        validate_environment(allow_test_mode=True)
    except SystemExit:
        # Handle config validation failures gracefully in CLI
        print("Warning: Configuration validation failed, some features may not work properly", file=sys.stderr)
    
    # Route to command handlers
    if args.command == 'query':
        return asyncio.run(handle_query_command(args))
    elif args.command == 'ingest':
        return handle_ingest_command(args)
    elif args.command == 'serve':
        return handle_serve_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())