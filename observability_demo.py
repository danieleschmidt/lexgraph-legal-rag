#!/usr/bin/env python3
"""
Observability demonstration script
Shows how to use enhanced monitoring and tracing in the RAG system
"""
import time
import asyncio
from lexgraph_legal_rag.observability import (
    initialize_observability,
    trace_operation,
    track_agent_operation,
    track_rag_query,
    ObservabilityMixin
)

class RAGAgent(ObservabilityMixin):
    """Example RAG agent with observability."""
    
    def __init__(self, agent_type: str):
        super().__init__(agent_type)
        self.agent_type = agent_type
    
    async def process_query(self, query: str):
        """Process a query with full observability."""
        with self.trace("process_query", query=query) as span:
            start_time = time.time()
            
            try:
                # Simulate processing
                await asyncio.sleep(0.1)
                
                # Track success
                duration = time.time() - start_time
                self.track_operation("process_query", duration, True)
                
                track_rag_query(
                    query_type="demo_query",
                    agent=self.agent_type,
                    latency_seconds=duration,
                    success=True,
                    result_count=3
                )
                
                return f"Processed by {self.agent_type}: {query}"
                
            except Exception as e:
                duration = time.time() - start_time
                self.track_operation("process_query", duration, False)
                self.track_error(e)
                raise

async def demo_observability():
    """Demonstrate enhanced observability features."""
    print("🔍 Initializing enhanced observability...")
    
    # Initialize with tracing and metrics
    initialize_observability(
        enable_tracing=True,
        enable_metrics=True,
        prometheus_port=8003
    )
    
    print("✅ Observability initialized")
    print("📊 Prometheus metrics available at http://localhost:8003")
    
    # Create demo agents
    retriever = RAGAgent("retriever")
    summarizer = RAGAgent("summarizer")
    
    # Simulate RAG pipeline with full tracing
    with trace_operation("rag_pipeline", "demo") as pipeline_span:
        print("\n🚀 Running RAG pipeline with observability...")
        
        # Step 1: Retrieve documents
        result1 = await retriever.process_query("legal precedent for contract disputes")
        print(f"Step 1: {result1}")
        
        # Step 2: Summarize results
        result2 = await summarizer.process_query(result1)
        print(f"Step 2: {result2}")
        
        print("\n📈 Pipeline completed with full observability!")
        print("   - Distributed traces captured")  
        print("   - Metrics recorded for each operation")
        print("   - Performance data available in Prometheus")

if __name__ == "__main__":
    asyncio.run(demo_observability())