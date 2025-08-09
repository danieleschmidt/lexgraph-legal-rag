#!/usr/bin/env python3
"""Enhanced pipeline with comprehensive error handling and monitoring."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional
import structlog

from src.lexgraph_legal_rag.document_pipeline import LegalDocumentPipeline
from src.lexgraph_legal_rag.multi_agent import MultiAgentGraph
from src.lexgraph_legal_rag.logging_config import configure_logging
from src.lexgraph_legal_rag.metrics import start_metrics_server
from src.lexgraph_legal_rag.validation_fixed import validate_query_input, validate_document_content
from src.lexgraph_legal_rag.cache import get_query_cache


class EnhancedLegalRAGPipeline:
    """Production-ready legal RAG pipeline with comprehensive error handling."""
    
    def __init__(
        self, 
        use_semantic: bool = True, 
        enable_caching: bool = True,
        enable_monitoring: bool = True,
        metrics_port: Optional[int] = None
    ):
        # Configure structured logging
        configure_logging(level="INFO")
        self.logger = structlog.get_logger(__name__)
        
        # Initialize metrics server if requested
        if enable_monitoring and metrics_port:
            start_metrics_server(metrics_port)
            self.logger.info("Started metrics server", port=metrics_port)
        
        # Initialize core components with error handling
        try:
            self.pipeline = LegalDocumentPipeline(use_semantic=use_semantic)
            self.agent_graph = MultiAgentGraph(pipeline=self.pipeline)
            self.enable_caching = enable_caching
            
            # Initialize cache if enabled
            if enable_caching:
                self.cache = get_query_cache()
                self.logger.info("Query caching enabled")
            
            self.logger.info(
                "Enhanced Legal RAG Pipeline initialized",
                semantic_search=use_semantic,
                caching_enabled=enable_caching,
                monitoring_enabled=enable_monitoring
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize pipeline", error=str(e))
            raise
    
    def ingest_documents(
        self, 
        docs_path: Path, 
        chunk_size: int = 512,
        validate_content: bool = True
    ) -> dict:
        """Ingest documents with comprehensive validation and error handling."""
        self.logger.info("Starting document ingestion", path=str(docs_path), chunk_size=chunk_size)
        
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_path}")
        
        if not docs_path.is_dir():
            raise ValueError(f"Path is not a directory: {docs_path}")
        
        stats = {
            "files_processed": 0,
            "files_failed": 0,
            "documents_created": 0,
            "validation_errors": 0,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Pre-process validation if enabled
            if validate_content:
                self.logger.info("Validating document content")
                validation_results = self._validate_documents(docs_path)
                stats["validation_errors"] = len(validation_results.get("errors", []))
                
                if validation_results.get("errors"):
                    self.logger.warning("Document validation issues found", 
                                      errors=validation_results["errors"][:5])  # Log first 5 errors
            
            # Ingest documents
            num_files = self.pipeline.ingest_directory(docs_path, chunk_size=chunk_size, enable_semantic=True)
            stats["files_processed"] = num_files
            stats["documents_created"] = len(self.pipeline.documents)
            
            # Get index statistics
            index_stats = self.pipeline.get_index_stats()
            stats.update(index_stats)
            
            stats["processing_time"] = time.time() - start_time
            
            self.logger.info("Document ingestion completed", **stats)
            return stats
            
        except Exception as e:
            stats["processing_time"] = time.time() - start_time
            self.logger.error("Document ingestion failed", error=str(e), **stats)
            raise
    
    def _validate_documents(self, docs_path: Path) -> dict:
        """Validate document content and structure."""
        validation_results = {"errors": [], "warnings": [], "files_checked": 0}
        
        for file_path in docs_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {'.txt', '.md'}:
                validation_results["files_checked"] += 1
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    validation_result = validate_document_content(content, str(file_path))
                    
                    if not validation_result.is_valid:
                        validation_results["errors"].extend(validation_result.errors)
                        validation_results["warnings"].extend(validation_result.warnings)
                        
                except Exception as e:
                    validation_results["errors"].append(f"Failed to read {file_path}: {e}")
        
        return validation_results
    
    async def query(
        self, 
        query_text: str, 
        include_citations: bool = True,
        use_cache: bool = None,
        timeout: float = 30.0
    ) -> dict:
        """Process query with comprehensive error handling and monitoring."""
        start_time = time.time()
        query_id = f"query_{int(start_time * 1000)}"
        
        self.logger.info("Processing query", query_id=query_id, query=query_text[:100])
        
        # Use instance setting if not overridden
        if use_cache is None:
            use_cache = self.enable_caching
        
        result = {
            "query_id": query_id,
            "query": query_text,
            "answer": None,
            "citations": [],
            "processing_time": 0,
            "cached": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Input validation
            validation_result = validate_query_input(query_text)
            if not validation_result.is_valid:
                result["errors"] = validation_result.errors
                result["warnings"] = validation_result.warnings
                self.logger.warning("Query validation failed", errors=validation_result.errors)
                return result
            
            # Check cache if enabled
            cache_key = f"query:{hash(query_text)}"
            if use_cache:
                try:
                    cached_result = self.cache.get_raw(cache_key)
                    if cached_result:
                        result.update(cached_result)
                        result["cached"] = True
                        result["processing_time"] = time.time() - start_time
                        self.logger.info("Query served from cache", query_id=query_id)
                        return result
                except Exception as e:
                    self.logger.warning("Cache lookup failed", error=str(e))
            
            # Process query with timeout
            try:
                if include_citations:
                    # Collect streaming citations
                    answer_parts = []
                    citations = []
                    
                    async def collect_response():
                        async for chunk in self.agent_graph.run_with_citations(query_text, self.pipeline, top_k=3):
                            if chunk.startswith("Citations:"):
                                citations.append(chunk)
                            else:
                                answer_parts.append(chunk)
                    
                    # Run with timeout
                    await asyncio.wait_for(collect_response(), timeout=timeout)
                    
                    result["answer"] = ''.join(answer_parts)
                    result["citations"] = citations
                else:
                    # Simple query without citations
                    result["answer"] = await asyncio.wait_for(
                        self.agent_graph.run(query_text), timeout=timeout
                    )
                
                # Cache successful results
                if use_cache and result["answer"]:
                    try:
                        cache_data = {
                            "answer": result["answer"],
                            "citations": result["citations"],
                            "timestamp": time.time()
                        }
                        self.cache.put_raw(cache_key, cache_data, ttl=3600)  # 1 hour TTL
                    except Exception as e:
                        self.logger.warning("Failed to cache result", error=str(e))
                
                result["processing_time"] = time.time() - start_time
                self.logger.info("Query processed successfully", 
                               query_id=query_id, 
                               processing_time=result["processing_time"],
                               answer_length=len(result["answer"]) if result["answer"] else 0)
                
            except asyncio.TimeoutError:
                error_msg = f"Query processing timed out after {timeout}s"
                result["errors"].append(error_msg)
                self.logger.error("Query timeout", query_id=query_id, timeout=timeout)
                
        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            result["errors"].append(error_msg)
            self.logger.error("Query processing error", query_id=query_id, error=str(e))
        
        result["processing_time"] = time.time() - start_time
        return result
    
    def get_pipeline_status(self) -> dict:
        """Get comprehensive pipeline status and health metrics."""
        try:
            stats = self.pipeline.get_index_stats()
            
            # Add cache statistics
            if self.enable_caching:
                try:
                    cache_stats = self.cache.get_stats()
                    stats["cache"] = cache_stats
                except Exception as e:
                    stats["cache"] = {"error": str(e)}
            
            # Add system metrics
            import psutil
            stats["system"] = {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "disk_usage": psutil.disk_usage('.').percent
            }
            
            stats["status"] = "healthy"
            stats["timestamp"] = time.time()
            
        except Exception as e:
            stats = {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
        
        return stats
    
    def save_index(self, path: str) -> None:
        """Save pipeline index with error handling."""
        try:
            self.pipeline.save_index(path)
            self.logger.info("Index saved successfully", path=path)
        except Exception as e:
            self.logger.error("Failed to save index", path=path, error=str(e))
            raise
    
    def load_index(self, path: str) -> None:
        """Load pipeline index with error handling."""
        try:
            self.pipeline.load_index(path)
            self.logger.info("Index loaded successfully", path=path)
        except Exception as e:
            self.logger.error("Failed to load index", path=path, error=str(e))
            raise


async def main():
    """Demo the enhanced pipeline."""
    print("🔧 Initializing Enhanced Legal RAG Pipeline...")
    
    # Initialize with monitoring
    pipeline = EnhancedLegalRAGPipeline(
        use_semantic=True, 
        enable_caching=True,
        enable_monitoring=True,
        metrics_port=9090
    )
    
    # Ingest documents with validation
    docs_path = Path('demo_documents')
    if docs_path.exists():
        print("📄 Ingesting documents with validation...")
        stats = pipeline.ingest_documents(docs_path, validate_content=True)
        print(f"✅ Ingestion completed: {stats}")
        
        # Save index
        pipeline.save_index('enhanced_index.bin')
    
    # Test queries with comprehensive error handling
    test_queries = [
        "What are liability limits in commercial leases?",
        "Explain California Civil Code 1542",
        "Find termination clauses",
        "",  # Empty query test
        "What is indemnification in contracts?"
    ]
    
    print("\n🔍 Testing Enhanced Query Processing...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        result = await pipeline.query(query, include_citations=True)
        
        if result["errors"]:
            print(f"❌ Errors: {result['errors']}")
        elif result["answer"]:
            print(f"✅ Answer ({result['processing_time']:.2f}s, cached={result['cached']}): {result['answer'][:100]}...")
        else:
            print("⚠️ No answer generated")
    
    # Show pipeline status
    print("\n📊 Pipeline Status:")
    status = pipeline.get_pipeline_status()
    print(f"Status: {status['status']}")
    print(f"Documents: {status.get('document_count', 'unknown')}")
    print(f"Memory: {status.get('system', {}).get('memory_percent', 'unknown')}%")


if __name__ == "__main__":
    asyncio.run(main())