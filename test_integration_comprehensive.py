#!/usr/bin/env python3
"""Comprehensive integration tests for the LexGraph Legal RAG system."""

import asyncio
import pytest
import tempfile
from pathlib import Path
import os
import shutil

from src.lexgraph_legal_rag.document_pipeline import LegalDocumentPipeline
from src.lexgraph_legal_rag.multi_agent import MultiAgentGraph
from src.lexgraph_legal_rag.validation_fixed import validate_query_input, validate_document_content
from src.lexgraph_legal_rag.cache import get_query_cache

# Import our enhanced pipelines
import sys
sys.path.append(str(Path(__file__).parent))
from enhanced_pipeline import EnhancedLegalRAGPipeline
from scalable_pipeline import ScalableLegalRAGPipeline


@pytest.fixture
def temp_docs_dir():
    """Create temporary directory with test documents."""
    temp_dir = tempfile.mkdtemp()
    docs_path = Path(temp_dir) / "test_docs"
    docs_path.mkdir()
    
    # Create test legal documents
    (docs_path / "contract1.txt").write_text(
        "SOFTWARE LICENSE AGREEMENT\n\n"
        "This agreement grants a license to use the software. "
        "The license is subject to the following terms:\n\n"
        "LIABILITY LIMITATION: Licensor's liability is limited to $1000. "
        "WARRANTY DISCLAIMER: Software is provided AS IS without warranties. "
        "TERMINATION: License terminates upon breach."
    )
    
    (docs_path / "statute.txt").write_text(
        "CALIFORNIA CIVIL CODE SECTION 1542\n\n"
        "A general release does not extend to unknown claims. "
        "This provision protects releasing parties from waiving unknown claims. "
        "To waive Section 1542 protections, explicit language is required."
    )
    
    (docs_path / "policy.txt").write_text(
        "PRIVACY POLICY\n\n"
        "We collect and process personal data in accordance with applicable laws. "
        "Data is stored securely and shared only with authorized parties. "
        "Users have rights to access, modify, and delete their data."
    )
    
    yield docs_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestDocumentPipeline:
    """Test core document pipeline functionality."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        pipeline = LegalDocumentPipeline(use_semantic=True)
        assert pipeline.index is not None
        assert pipeline.semantic is not None
        assert len(pipeline.documents) == 0
    
    def test_document_ingestion(self, temp_docs_dir):
        """Test document ingestion and indexing."""
        pipeline = LegalDocumentPipeline(use_semantic=True)
        
        # Ingest documents
        num_files = pipeline.ingest_directory(temp_docs_dir, enable_semantic=True)
        assert num_files == 3
        assert len(pipeline.documents) > 0
        
        # Check index statistics
        stats = pipeline.get_index_stats()
        assert stats["document_count"] > 0
        assert stats["semantic_enabled"] is True
        assert stats["vector_dim"] > 0
    
    def test_search_functionality(self, temp_docs_dir):
        """Test document search capabilities."""
        pipeline = LegalDocumentPipeline(use_semantic=True)
        pipeline.ingest_directory(temp_docs_dir, enable_semantic=True)
        
        # Test basic search
        results = pipeline.search("liability limitation", top_k=3)
        assert len(results) > 0
        
        doc, score = results[0]
        assert score > 0
        assert "liability" in doc.text.lower() or "license" in doc.text.lower()
        
        # Test semantic search
        semantic_results = pipeline.search("software terms", top_k=3, semantic=True)
        assert len(semantic_results) > 0
    
    def test_batch_search(self, temp_docs_dir):
        """Test batch search functionality."""
        pipeline = LegalDocumentPipeline(use_semantic=True)
        pipeline.ingest_directory(temp_docs_dir, enable_semantic=True)
        
        queries = [
            "liability limitation",
            "privacy data collection",
            "california civil code"
        ]
        
        results = pipeline.batch_search(queries, top_k=2)
        assert len(results) == len(queries)
        
        for query_results in results:
            assert len(query_results) <= 2
            for doc, score in query_results:
                assert score > 0
    
    def test_index_persistence(self, temp_docs_dir):
        """Test saving and loading indices."""
        pipeline1 = LegalDocumentPipeline(use_semantic=True)
        pipeline1.ingest_directory(temp_docs_dir, enable_semantic=True)
        
        original_count = len(pipeline1.documents)
        
        # Save index
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            index_path = tmp.name
        
        try:
            pipeline1.save_index(index_path)
            
            # Load into new pipeline
            pipeline2 = LegalDocumentPipeline(use_semantic=True)
            pipeline2.load_index(index_path)
            
            assert len(pipeline2.documents) == original_count
            
            # Test search works after loading
            results = pipeline2.search("liability", top_k=1)
            assert len(results) > 0
            
        finally:
            if os.path.exists(index_path):
                os.unlink(index_path)


class TestMultiAgentSystem:
    """Test multi-agent RAG system."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, temp_docs_dir):
        """Test multi-agent system initialization."""
        pipeline = LegalDocumentPipeline(use_semantic=True)
        pipeline.ingest_directory(temp_docs_dir, enable_semantic=True)
        
        agent_graph = MultiAgentGraph(pipeline=pipeline)
        assert agent_graph.retriever is not None
        assert agent_graph.summarizer is not None
        assert agent_graph.clause_explainer is not None
        assert agent_graph.router is not None
    
    @pytest.mark.asyncio
    async def test_query_processing(self, temp_docs_dir):
        """Test end-to-end query processing."""
        pipeline = LegalDocumentPipeline(use_semantic=True)
        pipeline.ingest_directory(temp_docs_dir, enable_semantic=True)
        
        agent_graph = MultiAgentGraph(pipeline=pipeline)
        
        # Test different types of queries
        queries = [
            "What are the liability limits?",
            "Explain California Civil Code 1542",
            "Find privacy policy terms",
            "Summarize software license terms"
        ]
        
        for query in queries:
            result = await agent_graph.run(query)
            assert isinstance(result, str)
            assert len(result) > 0
            assert not result.startswith("Error")
    
    @pytest.mark.asyncio
    async def test_citations_generation(self, temp_docs_dir):
        """Test citation generation."""
        pipeline = LegalDocumentPipeline(use_semantic=True)
        pipeline.ingest_directory(temp_docs_dir, enable_semantic=True)
        
        agent_graph = MultiAgentGraph(pipeline=pipeline)
        
        citations = []
        async for chunk in agent_graph.run_with_citations("liability limitation", pipeline, top_k=2):
            citations.append(chunk)
        
        citation_text = ''.join(citations)
        assert "Citations:" in citation_text
        assert len(citations) > 1  # Should have answer + citations


class TestValidation:
    """Test input validation and security."""
    
    def test_query_validation(self):
        """Test query input validation."""
        # Valid queries
        valid_result = validate_query_input("What are liability terms?")
        assert valid_result.is_valid
        assert valid_result.sanitized_input == "What are liability terms?"
        
        # Empty query
        empty_result = validate_query_input("")
        assert not empty_result.is_valid
        assert "too short" in str(empty_result.errors)
        
        # Very long query
        long_query = "x" * 10000
        long_result = validate_query_input(long_query)
        assert not long_result.is_valid
        assert "too long" in str(long_result.errors)
        
        # Potentially malicious query
        malicious_query = "<script>alert('xss')</script>What is liability?"
        malicious_result = validate_query_input(malicious_query)
        assert len(malicious_result.warnings) > 0
    
    def test_document_validation(self):
        """Test document content validation."""
        # Valid legal document
        legal_content = "This agreement establishes liability limits and warranty terms."
        legal_result = validate_document_content(legal_content, "test.txt")
        assert legal_result.is_valid
        
        # Too short content
        short_result = validate_document_content("short", "test.txt")
        assert not short_result.is_valid
        
        # Non-legal content
        non_legal = "This is just a random text without legal terms."
        non_legal_result = validate_document_content(non_legal, "test.txt")
        assert non_legal_result.is_valid  # Valid but may have warnings
        
        # Content with suspicious patterns
        suspicious = "This contract contains <script>malicious code</script>"
        suspicious_result = validate_document_content(suspicious, "test.txt")
        assert len(suspicious_result.warnings) > 0


class TestEnhancedPipeline:
    """Test enhanced pipeline with error handling and monitoring."""
    
    @pytest.mark.asyncio
    async def test_enhanced_initialization(self):
        """Test enhanced pipeline initialization."""
        pipeline = EnhancedLegalRAGPipeline(
            use_semantic=True,
            enable_caching=True,
            enable_monitoring=False,  # Disable for testing
            metrics_port=None
        )
        
        assert pipeline.pipeline is not None
        assert pipeline.agent_graph is not None
        assert pipeline.enable_caching is True
    
    @pytest.mark.asyncio
    async def test_enhanced_query_processing(self, temp_docs_dir):
        """Test enhanced query processing with error handling."""
        pipeline = EnhancedLegalRAGPipeline(
            enable_monitoring=False,
            metrics_port=None
        )
        
        # Ingest documents
        stats = pipeline.ingest_documents(temp_docs_dir, validate_content=True)
        assert stats["files_processed"] > 0
        assert stats["validation_errors"] == 0
        
        # Test successful query
        result = await pipeline.query("What are liability terms?", include_citations=False)
        assert result["answer"] is not None
        assert len(result["errors"]) == 0
        
        # Test invalid query
        invalid_result = await pipeline.query("", include_citations=False)
        assert len(invalid_result["errors"]) > 0
        assert invalid_result["answer"] is None
    
    @pytest.mark.asyncio
    async def test_pipeline_status(self, temp_docs_dir):
        """Test pipeline status monitoring."""
        pipeline = EnhancedLegalRAGPipeline(
            enable_monitoring=False,
            metrics_port=None
        )
        
        pipeline.ingest_documents(temp_docs_dir)
        status = pipeline.get_pipeline_status()
        
        assert status["status"] == "healthy"
        assert "document_count" in status
        assert "system" in status
        assert status["document_count"] > 0


class TestScalablePipeline:
    """Test scalable pipeline with performance optimizations."""
    
    @pytest.mark.asyncio
    async def test_scalable_initialization(self):
        """Test scalable pipeline initialization."""
        pipeline = ScalableLegalRAGPipeline(
            enable_monitoring=False,
            metrics_port=None,
            max_workers=2,
            auto_scale=False  # Disable for testing
        )
        
        assert len(pipeline.pipeline_pool) == 2
        assert len(pipeline.agent_pool) == 2
        assert pipeline.max_workers == 2
    
    @pytest.mark.asyncio
    async def test_parallel_ingestion(self, temp_docs_dir):
        """Test parallel document ingestion."""
        pipeline = ScalableLegalRAGPipeline(
            enable_monitoring=False,
            metrics_port=None,
            max_workers=2,
            auto_scale=False
        )
        
        stats = await pipeline.ingest_documents_parallel([temp_docs_dir])
        assert stats["total_files_processed"] > 0
        assert stats["parallel_jobs"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_query_processing(self, temp_docs_dir):
        """Test batch query processing."""
        pipeline = ScalableLegalRAGPipeline(
            enable_monitoring=False,
            metrics_port=None,
            max_workers=2,
            batch_size=4,
            auto_scale=False
        )
        
        await pipeline.ingest_documents_parallel([temp_docs_dir])
        
        queries = [
            "What are liability terms?",
            "Explain privacy policy",
            "Find software license terms",
            "What is Civil Code 1542?"
        ]
        
        results = await pipeline.query_batch(queries, include_citations=False, timeout=10.0)
        
        assert len(results) == len(queries)
        successful = [r for r in results if not r.get("errors")]
        assert len(successful) > 0  # At least some should succeed
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, temp_docs_dir):
        """Test performance metrics collection."""
        pipeline = ScalableLegalRAGPipeline(
            enable_monitoring=False,
            metrics_port=None,
            max_workers=2,
            auto_scale=False
        )
        
        await pipeline.ingest_documents_parallel([temp_docs_dir])
        
        # Process some queries to generate metrics
        await pipeline.query_batch(["liability", "privacy"], include_citations=False)
        
        metrics = pipeline.get_performance_metrics()
        assert "query_count" in metrics
        assert "avg_response_time" in metrics
        assert "cache_hit_rate" in metrics
        assert metrics["query_count"] > 0


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    @pytest.mark.asyncio
    async def test_missing_documents_error(self):
        """Test handling of missing document directories."""
        pipeline = EnhancedLegalRAGPipeline(
            enable_monitoring=False,
            metrics_port=None
        )
        
        nonexistent_path = Path("/nonexistent/path")
        
        with pytest.raises(FileNotFoundError):
            pipeline.ingest_documents(nonexistent_path)
    
    @pytest.mark.asyncio
    async def test_query_timeout_handling(self, temp_docs_dir):
        """Test query timeout handling."""
        pipeline = EnhancedLegalRAGPipeline(
            enable_monitoring=False,
            metrics_port=None
        )
        
        pipeline.ingest_documents(temp_docs_dir)
        
        # Test with very short timeout to trigger timeout
        result = await pipeline.query("liability terms", timeout=0.001)
        
        # Should either succeed quickly or timeout gracefully
        assert "answer" in result
        assert "errors" in result
        assert "processing_time" in result


if __name__ == "__main__":
    # Run tests programmatically
    pytest.main([__file__, "-v", "--tb=short"])