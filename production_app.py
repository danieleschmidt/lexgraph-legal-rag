#!/usr/bin/env python3
"""Production-ready LexGraph Legal RAG application with full SDLC implementation."""

import asyncio
import os
import sys
import signal
from pathlib import Path
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

# Import our enhanced systems
from src.lexgraph_legal_rag.api import create_api
from src.lexgraph_legal_rag.logging_config import configure_logging
from scalable_pipeline import ScalableLegalRAGPipeline
import structlog


class ProductionLexGraphApp:
    """Production-ready LexGraph Legal RAG application."""
    
    def __init__(self):
        self.pipeline: Optional[ScalableLegalRAGPipeline] = None
        self.app = None
        self.logger = None
        self._shutdown_event = asyncio.Event()
        
    async def initialize(
        self,
        docs_paths: list[str] = None,
        api_key: str = None,
        enable_monitoring: bool = True,
        metrics_port: int = 9090,
        max_workers: int = 4
    ):
        """Initialize the production application."""
        # Configure production logging
        configure_logging(level="INFO")
        self.logger = structlog.get_logger(__name__)
        
        self.logger.info("Initializing LexGraph Legal RAG Production Application")
        
        # Initialize scalable pipeline
        self.pipeline = ScalableLegalRAGPipeline(
            use_semantic=True,
            enable_caching=True,
            enable_monitoring=enable_monitoring,
            metrics_port=metrics_port,
            max_workers=max_workers,
            batch_size=10,
            enable_prefetching=True,
            cache_ttl=3600,
            auto_scale=True
        )
        
        # Ingest documents if provided
        if docs_paths:
            doc_paths = [Path(p) for p in docs_paths if Path(p).exists()]
            if doc_paths:
                self.logger.info("Ingesting legal documents", paths=doc_paths)
                stats = await self.pipeline.ingest_documents_parallel(doc_paths)
                self.logger.info("Document ingestion completed", **stats)
            else:
                self.logger.warning("No valid document paths provided", paths=docs_paths)
        
        # Get API key from environment or parameter
        if api_key is None:
            api_key = os.environ.get("API_KEY")
        
        if not api_key:
            self.logger.error("API_KEY not provided - application cannot start")
            raise ValueError("API_KEY is required for production deployment")
        
        # Create FastAPI application
        @asynccontextmanager
        async def lifespan(app):
            """Application lifespan manager."""
            self.logger.info("Application starting up")
            yield
            self.logger.info("Application shutting down")
            await self.shutdown()
        
        self.app = create_api(
            api_key=api_key,
            enable_docs=True,  # Enable in production for API documentation
            test_mode=False
        )
        
        # Add health check endpoints that use our pipeline
        self._add_production_endpoints()
        
        self.logger.info("Production application initialized successfully")
    
    def _add_production_endpoints(self):
        """Add production-specific endpoints."""
        
        @self.app.get("/production/health")
        async def production_health():
            """Production health check with pipeline status."""
            if not self.pipeline:
                return {"status": "unhealthy", "error": "Pipeline not initialized"}
            
            try:
                metrics = self.pipeline.get_performance_metrics()
                return {
                    "status": "healthy",
                    "pipeline_status": "operational",
                    "metrics": {
                        "query_count": metrics["query_count"],
                        "avg_response_time": metrics["avg_response_time"],
                        "cache_hit_rate": metrics["cache_hit_rate"],
                        "workers": metrics["pipeline_workers"]
                    },
                    "system_load": metrics["system_load"]
                }
            except Exception as e:
                return {
                    "status": "degraded", 
                    "error": str(e),
                    "pipeline_status": "error"
                }
        
        @self.app.post("/production/query")
        async def production_query(request: dict):
            """Production query endpoint with enhanced processing."""
            if not self.pipeline:
                return {"error": "Pipeline not available", "status": "error"}
            
            query = request.get("query", "")
            include_citations = request.get("include_citations", True)
            timeout = min(request.get("timeout", 30.0), 60.0)  # Max 60s timeout
            
            if not query:
                return {"error": "Query is required", "status": "error"}
            
            try:
                results = await self.pipeline.query_batch(
                    [query], 
                    include_citations=include_citations,
                    timeout=timeout
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    return {
                        "status": "success",
                        "query": query,
                        "answer": result.get("answer"),
                        "citations": result.get("citations", []),
                        "processing_time": result.get("processing_time", 0),
                        "cached": result.get("cached", False),
                        "errors": result.get("errors", [])
                    }
                else:
                    return {"error": "No result generated", "status": "error"}
                    
            except Exception as e:
                self.logger.error("Production query error", error=str(e))
                return {"error": str(e), "status": "error"}
        
        @self.app.post("/production/query-batch")
        async def production_query_batch(request: dict):
            """Production batch query endpoint."""
            if not self.pipeline:
                return {"error": "Pipeline not available", "status": "error"}
            
            queries = request.get("queries", [])
            include_citations = request.get("include_citations", False)  # Default false for batch
            timeout = min(request.get("timeout", 30.0), 120.0)  # Max 2min for batch
            
            if not queries or len(queries) == 0:
                return {"error": "Queries list is required", "status": "error"}
            
            if len(queries) > 50:  # Limit batch size
                return {"error": "Maximum 50 queries per batch", "status": "error"}
            
            try:
                results = await self.pipeline.query_batch(
                    queries,
                    include_citations=include_citations,
                    timeout=timeout
                )
                
                return {
                    "status": "success",
                    "query_count": len(queries),
                    "results": results,
                    "performance": self.pipeline.get_performance_metrics()
                }
                
            except Exception as e:
                self.logger.error("Production batch query error", error=str(e))
                return {"error": str(e), "status": "error"}
        
        @self.app.get("/production/metrics")
        async def production_metrics():
            """Get detailed production metrics."""
            if not self.pipeline:
                return {"error": "Pipeline not available"}
            
            return self.pipeline.get_performance_metrics()
    
    async def shutdown(self):
        """Graceful shutdown of the application."""
        self.logger.info("Initiating graceful shutdown")
        
        if self.pipeline:
            # Save pipeline state before shutdown
            try:
                self.pipeline.save_indices("production_shutdown_backup")
                self.logger.info("Pipeline state saved for backup")
            except Exception as e:
                self.logger.error("Failed to save pipeline state", error=str(e))
        
        self._shutdown_event.set()
        self.logger.info("Shutdown completed")
    
    def handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("Received shutdown signal", signal=signum)
        asyncio.create_task(self.shutdown())
    
    async def run_production_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        log_level: str = "info"
    ):
        """Run the production server with proper configuration."""
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self.handle_shutdown_signal)
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            access_log=True,
            server_header=False,  # Security  
            reload=False,         # Production
        )
        
        server = uvicorn.Server(config)
        
        self.logger.info(
            "Starting LexGraph Legal RAG Production Server",
            host=host,
            port=port,
            workers=workers
        )
        
        try:
            await server.serve()
        except Exception as e:
            self.logger.error("Server error", error=str(e))
            raise
        finally:
            await self.shutdown()


async def main():
    """Main entry point for production application."""
    
    # Get configuration from environment
    docs_paths = os.environ.get("DOCS_PATHS", "").split(",") if os.environ.get("DOCS_PATHS") else []
    docs_paths = [p.strip() for p in docs_paths if p.strip()]
    
    api_key = os.environ.get("API_KEY")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))
    max_pipeline_workers = int(os.environ.get("MAX_PIPELINE_WORKERS", "4"))
    metrics_port = int(os.environ.get("METRICS_PORT", "9090"))
    
    # Initialize and run production application
    app = ProductionLexGraphApp()
    
    try:
        await app.initialize(
            docs_paths=docs_paths,
            api_key=api_key,
            enable_monitoring=True,
            metrics_port=metrics_port,
            max_workers=max_pipeline_workers
        )
        
        await app.run_production_server(
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
        await app.shutdown()
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Quick demo mode if run directly
    print("🚀 LexGraph Legal RAG - Production Demo Mode")
    print("Setting up demo environment...")
    
    # Set demo environment variables
    os.environ["API_KEY"] = "production-demo-key-12345"
    os.environ["DOCS_PATHS"] = "demo_documents"
    os.environ["HOST"] = "127.0.0.1"
    os.environ["PORT"] = "8000"
    
    print("Starting production server on http://127.0.0.1:8000")
    print("📊 Metrics available on port 9090")
    print("📖 API docs available at http://127.0.0.1:8000/docs")
    print("\nProduction endpoints:")
    print("  GET  /production/health")
    print("  POST /production/query")
    print("  POST /production/query-batch")
    print("  GET  /production/metrics")
    print("\nPress Ctrl+C to shutdown gracefully\n")
    
    asyncio.run(main())