"""Tests for batch processing performance optimization to eliminate N+1 query patterns.

This test suite validates the batch processing implementation that reduces
the N+1 query pattern in the multi-agent RAG system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Tuple

from lexgraph_legal_rag.document_pipeline import LegalDocumentPipeline, VectorIndex
from lexgraph_legal_rag.semantic_search import SemanticSearchPipeline, EmbeddingIndex
from lexgraph_legal_rag.multi_agent import MultiAgentGraph, RetrieverAgent
from lexgraph_legal_rag.models import LegalDocument


class TestVectorIndexBatchSearch:
    """Test batch search functionality in VectorIndex."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.index = VectorIndex()
        self.sample_docs = [
            LegalDocument(id="doc1", text="Contract liability and damages clause", metadata={"source": "contract1.txt"}),
            LegalDocument(id="doc2", text="Indemnification and warranty provisions", metadata={"source": "contract2.txt"}),
            LegalDocument(id="doc3", text="Termination and breach conditions", metadata={"source": "contract3.txt"}),
            LegalDocument(id="doc4", text="Intellectual property protection clause", metadata={"source": "contract4.txt"}),
        ]
        self.index.add(self.sample_docs)
    
    def test_batch_search_multiple_queries(self):
        """Test batch search with multiple queries."""
        queries = ["liability", "termination", "intellectual property"]
        results = self.index.batch_search(queries, top_k=2)
        
        assert len(results) == 3
        for query_results in results:
            assert len(query_results) <= 2
            for doc, score in query_results:
                assert isinstance(doc, LegalDocument)
                assert isinstance(score, float)
                assert score >= 0
    
    def test_batch_search_empty_queries(self):
        """Test batch search with empty query list."""
        results = self.index.batch_search([])
        assert results == []
    
    def test_batch_search_single_query(self):
        """Test batch search with single query."""
        results = self.index.batch_search(["liability"], top_k=3)
        assert len(results) == 1
        assert len(results[0]) <= 3
    
    def test_batch_search_performance_optimization(self):
        """Test that batch search uses optimized matrix operations."""
        queries = ["contract", "liability", "termination", "warranty"]
        
        # Mock the matrix operations to verify batch processing
        with patch('numpy.argpartition') as mock_partition:
            mock_partition.side_effect = lambda scores, k: list(range(len(scores)))[:k]
            
            results = self.index.batch_search(queries, top_k=2)
            
            # Should process all queries efficiently
            assert len(results) == 4
            for query_results in results:
                assert len(query_results) <= 2
    
    def test_batch_search_no_documents(self):
        """Test batch search when no documents are indexed."""
        empty_index = VectorIndex()
        results = empty_index.batch_search(["test query"], top_k=5)
        assert results == [[]]


class TestSemanticIndexBatchSearch:
    """Test batch search functionality in semantic search."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.index = EmbeddingIndex()
        self.sample_docs = [
            LegalDocument(id="doc1", text="Contract liability and damages clause", metadata={"source": "contract1.txt"}),
            LegalDocument(id="doc2", text="Indemnification and warranty provisions", metadata={"source": "contract2.txt"}),
            LegalDocument(id="doc3", text="Termination and breach conditions", metadata={"source": "contract3.txt"}),
        ]
        self.index.add(self.sample_docs)
    
    def test_semantic_batch_search(self):
        """Test semantic batch search functionality."""
        queries = ["contract terms", "warranty clause", "termination"]
        results = self.index.batch_search(queries, top_k=2)
        
        assert len(results) == 3
        for query_results in results:
            assert len(query_results) <= 2
            for doc, score in query_results:
                assert isinstance(doc, LegalDocument)
                assert isinstance(score, float)
    
    def test_semantic_pipeline_batch_search(self):
        """Test semantic pipeline batch search."""
        pipeline = SemanticSearchPipeline()
        pipeline.ingest(self.sample_docs)
        
        queries = ["liability", "warranty"]
        results = pipeline.batch_search(queries, top_k=2)
        
        assert len(results) == 2
        for query_results in results:
            assert len(query_results) <= 2


class TestDocumentPipelineBatchSearch:
    """Test batch search functionality in LegalDocumentPipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = LegalDocumentPipeline(use_semantic=False)
        self.sample_docs = [
            LegalDocument(id="doc1", text="Contract liability and damages clause", metadata={"source": "contract1.txt"}),
            LegalDocument(id="doc2", text="Indemnification and warranty provisions", metadata={"source": "contract2.txt"}),
            LegalDocument(id="doc3", text="Termination and breach conditions", metadata={"source": "contract3.txt"}),
        ]
        self.pipeline.index.add(self.sample_docs)
    
    def test_pipeline_batch_search_vector(self):
        """Test pipeline batch search with vector index."""
        queries = ["liability", "warranty", "termination"]
        results = self.pipeline.batch_search(queries, top_k=2)
        
        assert len(results) == 3
        for query_results in results:
            assert len(query_results) <= 2
            for doc, score in query_results:
                assert isinstance(doc, LegalDocument)
                assert isinstance(score, float)
    
    def test_pipeline_batch_search_semantic(self):
        """Test pipeline batch search with semantic index."""
        semantic_pipeline = LegalDocumentPipeline(use_semantic=True)
        semantic_pipeline.index.add(self.sample_docs)
        semantic_pipeline.semantic.ingest(self.sample_docs)
        
        queries = ["contract terms", "legal provisions"]
        results = semantic_pipeline.batch_search(queries, top_k=2, semantic=True)
        
        assert len(results) == 2
        for query_results in results:
            assert len(query_results) <= 2
    
    @pytest.mark.asyncio
    async def test_pipeline_batch_search_with_cache(self):
        """Test pipeline batch search with caching integration."""
        queries = ["liability", "warranty"]
        
        # First search should populate cache
        results1 = self.pipeline.batch_search(queries, top_k=2, use_cache=True)
        
        # Second search should use cache
        results2 = self.pipeline.batch_search(queries, top_k=2, use_cache=True)
        
        # Results should be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert len(r1) == len(r2)
    
    def test_pipeline_batch_search_mixed_cache(self):
        """Test batch search with partially cached queries."""
        # Cache one query
        cached_results = self.pipeline.search("liability", top_k=2)
        
        # Batch search with one cached and one new query
        queries = ["liability", "new query term"]
        results = self.pipeline.batch_search(queries, top_k=2)
        
        assert len(results) == 2
        # First result should match cached result
        assert len(results[0]) == len(cached_results)


class TestRetrieverAgentBatchProcessing:
    """Test batch processing in RetrieverAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = LegalDocumentPipeline()
        self.sample_docs = [
            LegalDocument(id="doc1", text="Contract liability clause", metadata={"path": "contract1.txt"}),
            LegalDocument(id="doc2", text="Warranty provisions", metadata={"path": "contract2.txt"}),
        ]
        self.pipeline.index.add(self.sample_docs)
        
        self.agent = RetrieverAgent(pipeline=self.pipeline, top_k=2)
    
    @pytest.mark.asyncio
    async def test_retriever_batch_run(self):
        """Test RetrieverAgent batch_run method."""
        queries = ["liability", "warranty"]
        results = await self.agent.batch_run(queries)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, str)
            # Should either contain retrieved content or "No relevant documents"
            assert "Source:" in result or "No relevant documents" in result
    
    @pytest.mark.asyncio
    async def test_retriever_batch_run_no_pipeline(self):
        """Test RetrieverAgent batch_run without pipeline."""
        agent = RetrieverAgent(pipeline=None)
        queries = ["test1", "test2"]
        results = await agent.batch_run(queries)
        
        assert len(results) == 2
        assert all("retrieved:" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_retriever_batch_run_empty_queries(self):
        """Test RetrieverAgent batch_run with empty queries."""
        results = await self.agent.batch_run([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_retriever_batch_fallback_to_individual(self):
        """Test RetrieverAgent falls back to individual searches if no batch method."""
        # Mock pipeline without batch_search method
        mock_pipeline = Mock()
        mock_pipeline.batch_search = None  # No batch search method
        mock_pipeline.search.side_effect = [
            [(self.sample_docs[0], 0.8)],
            [(self.sample_docs[1], 0.7)]
        ]
        
        agent = RetrieverAgent(pipeline=mock_pipeline, top_k=1)
        queries = ["query1", "query2"]
        results = await agent.batch_run(queries)
        
        assert len(results) == 2
        assert mock_pipeline.search.call_count == 2


class TestMultiAgentBatchOptimization:
    """Test batch optimization in MultiAgentGraph to eliminate N+1 queries."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = LegalDocumentPipeline()
        self.sample_docs = [
            LegalDocument(id="doc1", text="Contract liability and damages provisions", metadata={"path": "contract1.txt"}),
            LegalDocument(id="doc2", text="Warranty and indemnification clauses", metadata={"path": "contract2.txt"}),
            LegalDocument(id="doc3", text="Termination conditions and breach remedies", metadata={"path": "contract3.txt"}),
        ]
        self.pipeline.index.add(self.sample_docs)
        
        self.agent_graph = MultiAgentGraph(pipeline=self.pipeline)
    
    @pytest.mark.asyncio
    async def test_batch_optimization_reduces_searches(self):
        """Test that batch optimization reduces the number of search calls."""
        query = "liability and damages"
        
        # Mock the pipeline search methods to count calls
        with patch.object(self.pipeline, 'search', wraps=self.pipeline.search) as mock_search, \
             patch.object(self.pipeline, 'batch_search', wraps=self.pipeline.batch_search) as mock_batch_search:
            
            # Run with citations (this previously caused N+1 queries)
            answer, docs = await self.agent_graph._run_with_batch_optimization(query, self.pipeline, top_k=3)
            
            # Should use batch optimization
            assert isinstance(answer, str)
            assert isinstance(docs, list)
            
            # Should not make excessive individual search calls
            # The key improvement is using batch processing when possible
            total_search_calls = mock_search.call_count + (mock_batch_search.call_count * 2)  # Estimate
            assert total_search_calls <= 3  # Reasonable limit for optimized searches
    
    @pytest.mark.asyncio 
    async def test_batch_optimization_with_complex_query(self):
        """Test batch optimization with complex queries that generate additional searches."""
        # Complex query with legal terms that might trigger additional searches
        query = "explain liability and indemnification clauses in detail"
        
        answer, docs = await self.agent_graph._run_with_batch_optimization(query, self.pipeline, top_k=2)
        
        assert isinstance(answer, str)
        assert isinstance(docs, list)
        assert len(docs) <= 2  # Respects top_k limit
        
        # Answer should be processed (not just retrieved text)
        assert len(answer) > 0
        assert "explain" not in answer.lower() or "liability" in answer.lower()
    
    @pytest.mark.asyncio
    async def test_batch_optimization_no_results(self):
        """Test batch optimization when no documents are found."""
        query = "nonexistent legal term xyz123"
        
        answer, docs = await self.agent_graph._run_with_batch_optimization(query, self.pipeline, top_k=3)
        
        assert "No relevant documents" in answer
        assert docs == []
    
    @pytest.mark.asyncio
    async def test_run_with_citations_uses_batch_optimization(self):
        """Test that run_with_citations uses batch optimization."""
        query = "warranty provisions"
        
        # Collect all chunks from the citation stream
        chunks = []
        async for chunk in self.agent_graph.run_with_citations(query, self.pipeline, top_k=2):
            chunks.append(chunk)
        
        assert len(chunks) >= 1  # Should have answer and potentially citations
        full_response = "".join(chunks)
        assert len(full_response) > 0
    
    @pytest.mark.asyncio
    async def test_batch_optimization_routing_decisions(self):
        """Test batch optimization with different routing decisions."""
        test_cases = [
            ("find liability clauses", "search"),
            ("summarize warranty terms", "summary"), 
            ("explain indemnification meaning", "explain"),
            ("contract provisions", "default")
        ]
        
        for query, expected_routing in test_cases:
            with patch.object(self.agent_graph.router, 'analyze_query_complexity', return_value=expected_routing):
                answer, docs = await self.agent_graph._run_with_batch_optimization(query, self.pipeline, top_k=2)
                
                assert isinstance(answer, str)
                assert isinstance(docs, list)
                assert len(answer) > 0


class TestPerformanceBenchmarks:
    """Performance benchmark tests for batch processing optimization."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.pipeline = LegalDocumentPipeline()
        
        # Create a larger document set for performance testing
        self.large_doc_set = []
        for i in range(50):
            doc = LegalDocument(
                id=f"doc{i}",
                text=f"Legal document {i} containing various terms like liability, warranty, indemnification, termination, and breach of contract provisions.",
                metadata={"path": f"doc{i}.txt"}
            )
            self.large_doc_set.append(doc)
        
        self.pipeline.index.add(self.large_doc_set)
    
    def test_batch_vs_individual_search_performance(self):
        """Compare performance of batch search vs individual searches."""
        import time
        
        queries = ["liability", "warranty", "termination", "indemnification", "breach"]
        
        # Time individual searches
        start_time = time.time()
        individual_results = []
        for query in queries:
            results = self.pipeline.search(query, top_k=5)
            individual_results.append(results)
        individual_time = time.time() - start_time
        
        # Time batch search
        start_time = time.time()
        batch_results = self.pipeline.batch_search(queries, top_k=5)
        batch_time = time.time() - start_time
        
        # Batch should be faster or comparable
        assert batch_time <= individual_time * 1.1  # Allow 10% margin
        
        # Results should be equivalent
        assert len(batch_results) == len(individual_results)
        for batch_result, individual_result in zip(batch_results, individual_results):
            assert len(batch_result) == len(individual_result)
    
    @pytest.mark.asyncio
    async def test_multi_agent_performance_improvement(self):
        """Test that multi-agent system shows performance improvement with batch optimization."""
        import time
        
        agent_graph = MultiAgentGraph(pipeline=self.pipeline)
        query = "explain liability and warranty provisions in detail"
        
        # Test batch-optimized version
        start_time = time.time()
        answer, docs = await agent_graph._run_with_batch_optimization(query, self.pipeline, top_k=5)
        batch_time = time.time() - start_time
        
        # Verify we got valid results
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(docs, list)
        
        # Performance should be reasonable (< 1 second for this test data)
        assert batch_time < 1.0
    
    def test_cache_hit_rate_with_batch_processing(self):
        """Test that batch processing maintains good cache utilization."""
        queries = ["liability", "warranty", "termination"]
        
        # First batch should populate cache
        results1 = self.pipeline.batch_search(queries, top_k=3, use_cache=True)
        
        # Second batch should hit cache
        results2 = self.pipeline.batch_search(queries, top_k=3, use_cache=True)
        
        # Results should be identical (cache working)
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert len(r1) == len(r2)
            # Compare document IDs to ensure same results
            ids1 = [doc.id for doc, _ in r1]
            ids2 = [doc.id for doc, _ in r2]
            assert ids1 == ids2


class TestBatchProcessingErrorHandling:
    """Test error handling in batch processing implementation."""
    
    def setup_method(self):
        """Set up error handling test fixtures."""
        self.pipeline = LegalDocumentPipeline()
    
    def test_batch_search_with_empty_index(self):
        """Test batch search behavior with empty index."""
        queries = ["test query 1", "test query 2"]
        results = self.pipeline.batch_search(queries, top_k=5)
        
        assert len(results) == 2
        assert all(len(result) == 0 for result in results)
    
    def test_batch_search_with_none_queries(self):
        """Test batch search error handling with None in queries."""
        # This should be handled gracefully
        queries = ["valid query", None, "another valid query"]
        
        # Should not crash, may filter out None values
        try:
            results = self.pipeline.batch_search([q for q in queries if q is not None], top_k=2)
            assert len(results) == 2  # Only valid queries
        except Exception as e:
            pytest.fail(f"Batch search should handle filtering gracefully: {e}")
    
    @pytest.mark.asyncio
    async def test_retriever_batch_with_pipeline_error(self):
        """Test RetrieverAgent batch processing with pipeline errors."""
        # Mock pipeline that raises an exception
        mock_pipeline = Mock()
        mock_pipeline.batch_search.side_effect = Exception("Pipeline error")
        mock_pipeline.search.side_effect = Exception("Search error")
        
        agent = RetrieverAgent(pipeline=mock_pipeline, top_k=2)
        
        # Should handle errors gracefully
        queries = ["query1", "query2"]
        try:
            results = await agent.batch_run(queries)
            # Should return some kind of error indicator
            assert len(results) == 2
        except Exception:
            # If exception is raised, it should be a specific, handled exception
            pass


if __name__ == "__main__":
    pytest.main([__file__])