#!/usr/bin/env python3
"""Simple demo of the LexGraph Legal RAG system."""

import asyncio
from pathlib import Path
from src.lexgraph_legal_rag.document_pipeline import LegalDocumentPipeline
from src.lexgraph_legal_rag.multi_agent import MultiAgentGraph


async def main():
    """Demo the complete legal RAG system."""
    # Initialize pipeline with semantic search
    print("🔧 Initializing Legal RAG Pipeline...")
    pipeline = LegalDocumentPipeline(use_semantic=True)
    
    # Check if demo documents exist
    docs_path = Path('demo_documents')
    if not docs_path.exists():
        print("❌ No demo documents found. Please create demo_documents/ directory.")
        return
    
    # Ingest documents
    print("📄 Ingesting legal documents...")
    num_docs = pipeline.ingest_directory(docs_path, enable_semantic=True)
    print(f"✅ Successfully ingested {num_docs} documents")
    
    # Save index for future use
    print("💾 Saving index...")
    pipeline.save_index('demo_index.bin')
    
    # Get pipeline stats
    stats = pipeline.get_index_stats()
    print(f"📊 Index Stats: {stats['document_count']} docs, {stats['vector_dim']} dimensions, {stats['index_size_mb']:.1f} MB")
    
    # Initialize multi-agent system
    print("🤖 Initializing Multi-Agent System...")
    agent_graph = MultiAgentGraph(pipeline=pipeline)
    
    # Test queries
    test_queries = [
        "What are liability limits in commercial leases?",
        "Explain software warranty disclaimers",
        "How does California Civil Code 1542 work with releases?",
        "Find termination clauses in contracts",
        "What intellectual property rights are mentioned?"
    ]
    
    print("\n🔍 Testing Legal Query Processing...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        try:
            # Run the agent graph
            result = await agent_graph.run(query)
            print(f"📋 Answer: {result}")
            
            # Also show citations
            print("\n📚 With Citations:")
            citation_chunks = []
            async for chunk in agent_graph.run_with_citations(query, pipeline, top_k=2):
                citation_chunks.append(chunk)
            
            print(''.join(citation_chunks))
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())