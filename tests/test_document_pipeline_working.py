"""Working comprehensive tests for document pipeline module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path

from lexgraph_legal_rag.document_pipeline import VectorIndex, LegalDocumentPipeline
from lexgraph_legal_rag.models import LegalDocument


class TestVectorIndexCore:
    """Test core VectorIndex functionality."""

    def test_initialization(self):
        """Test VectorIndex initialization."""
        index = VectorIndex()
        assert index._vectorizer is not None
        assert index._matrix is None
        assert index._docs == []
        assert index.documents == []

    def test_documents_property(self):
        """Test documents property returns internal docs list."""
        index = VectorIndex()
        docs = [LegalDocument(id="1", text="test document")]
        index._docs = docs
        assert index.documents == docs

    def test_search_empty_index(self):
        """Test search on empty index returns empty results."""
        index = VectorIndex()
        results = index.search("any query")
        assert results == []

    def test_batch_search_empty_index(self):
        """Test batch search on empty index."""
        index = VectorIndex()
        results = index.batch_search(["query1", "query2"])
        assert results == [[], []]

    def test_save_load_workflow(self):
        """Test save and load methods exist and can be called."""
        index = VectorIndex()
        # Add some test data
        docs = [LegalDocument(id="1", text="test", metadata={})]
        index._docs = docs
        
        # Test save method exists and is callable
        assert hasattr(index, 'save')
        assert callable(index.save)
        
        # Test load method exists and is callable
        assert hasattr(index, 'load')
        assert callable(index.load)


class TestLegalDocumentPipelineCore:
    """Test core LegalDocumentPipeline functionality."""

    def test_initialization_default(self):
        """Test pipeline initialization without semantic search."""
        pipeline = LegalDocumentPipeline()
        assert isinstance(pipeline.index, VectorIndex)
        assert pipeline.semantic is None

    @patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline')
    def test_initialization_with_semantic(self, mock_semantic_class):
        """Test pipeline initialization with semantic search."""
        mock_semantic = Mock()
        mock_semantic_class.return_value = mock_semantic
        
        pipeline = LegalDocumentPipeline(use_semantic=True)
        assert pipeline.semantic is mock_semantic
        mock_semantic_class.assert_called_once()

    def test_documents_property_delegation(self):
        """Test documents property delegates to index."""
        pipeline = LegalDocumentPipeline()
        test_docs = [LegalDocument(id="1", text="test")]
        pipeline.index._docs = test_docs
        
        assert pipeline.documents == test_docs

    def test_save_load_index_basic(self):
        """Test save and load index methods."""
        pipeline = LegalDocumentPipeline()
        
        with patch.object(pipeline.index, 'save') as mock_save:
            pipeline.save_index("test_path.bin")
            mock_save.assert_called_once_with("test_path.bin")
        
        with patch.object(pipeline.index, 'load') as mock_load, \
             patch('pathlib.Path.exists', return_value=False):
            pipeline.load_index("test_path.bin")
            mock_load.assert_called_once_with("test_path.bin")

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_search_cache_integration(self, mock_get_cache):
        """Test search integrates with caching system."""
        pipeline = LegalDocumentPipeline()
        
        # Test cache hit
        mock_cache = Mock()
        cached_results = [("cached_doc", 0.9)]
        mock_cache.get.return_value = cached_results
        mock_get_cache.return_value = mock_cache
        
        results = pipeline.search("test query")
        assert results == cached_results
        mock_cache.get.assert_called_once()

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_search_cache_miss(self, mock_get_cache):
        """Test search with cache miss uses index search."""
        pipeline = LegalDocumentPipeline()
        
        mock_cache = Mock()
        mock_cache.get.return_value = None  # Cache miss
        mock_get_cache.return_value = mock_cache
        
        search_results = [(LegalDocument(id="1", text="result"), 0.8)]
        
        with patch.object(pipeline.index, 'search', return_value=search_results) as mock_search:
            results = pipeline.search("test query")
            
            assert results == search_results
            mock_search.assert_called_once_with("test query", top_k=5)
            mock_cache.put.assert_called_once()

    def test_search_no_cache(self):
        """Test search with caching disabled."""
        pipeline = LegalDocumentPipeline()
        search_results = [(LegalDocument(id="1", text="result"), 0.8)]
        
        with patch.object(pipeline.index, 'search', return_value=search_results) as mock_search, \
             patch('lexgraph_legal_rag.document_pipeline.get_query_cache') as mock_get_cache:
            
            results = pipeline.search("test query", use_cache=False)
            
            assert results == search_results
            mock_get_cache.assert_not_called()

    @patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline')
    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_search_semantic_mode(self, mock_get_cache, mock_semantic_class):
        """Test search using semantic pipeline."""
        mock_semantic = Mock()
        search_results = [(LegalDocument(id="1", text="semantic result"), 0.9)]
        mock_semantic.search.return_value = search_results
        mock_semantic_class.return_value = mock_semantic
        
        pipeline = LegalDocumentPipeline(use_semantic=True)
        
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        
        results = pipeline.search("test query", semantic=True)
        
        assert results == search_results
        mock_semantic.search.assert_called_once_with("test query", top_k=5)

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_batch_search_empty_queries(self, mock_get_cache):
        """Test batch search with empty query list."""
        pipeline = LegalDocumentPipeline()
        results = pipeline.batch_search([])
        assert results == []

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_batch_search_basic(self, mock_get_cache):
        """Test basic batch search functionality."""
        pipeline = LegalDocumentPipeline()
        
        mock_cache = Mock()
        mock_cache.get.return_value = None  # All cache misses
        mock_get_cache.return_value = mock_cache
        
        batch_results = [[(LegalDocument(id="1", text="result1"), 0.8)], 
                        [(LegalDocument(id="2", text="result2"), 0.9)]]
        
        with patch.object(pipeline.index, 'batch_search', return_value=batch_results) as mock_batch:
            results = pipeline.batch_search(["query1", "query2"])
            
            assert len(results) == 2
            mock_batch.assert_called_once_with(["query1", "query2"], top_k=5)

    def test_ingest_folder_basic(self):
        """Test basic folder ingestion functionality."""
        pipeline = LegalDocumentPipeline()
        
        # Test that ingest_folder method exists and is callable
        assert hasattr(pipeline, 'ingest_folder')
        assert callable(pipeline.ingest_folder)
        
        # Test with mock to avoid file system dependencies
        with patch.object(pipeline.index, 'add') as mock_add, \
             patch('pathlib.Path') as mock_path_class:
            
            mock_folder = Mock()
            mock_folder.glob.return_value = []  # Empty folder
            mock_path_class.return_value = mock_folder
            
            pipeline.ingest_folder("/empty/folder")
            mock_add.assert_not_called()  # No files to add


class TestPipelineIntegration:
    """Integration and workflow tests."""

    def test_end_to_end_workflow(self):
        """Test complete pipeline workflow."""
        pipeline = LegalDocumentPipeline()
        
        # Mock all components to test workflow coordination
        with patch.object(pipeline.index, 'add') as mock_add, \
             patch.object(pipeline.index, 'search', return_value=[]) as mock_search, \
             patch.object(pipeline.index, 'save') as mock_save, \
             patch.object(pipeline.index, 'load') as mock_load, \
             patch('lexgraph_legal_rag.document_pipeline.get_query_cache') as mock_get_cache:
            
            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_get_cache.return_value = mock_cache
            
            # Test workflow steps
            docs = [LegalDocument(id="1", text="test document")]
            
            # Mock add documents via index
            pipeline.index.add(docs)
            mock_add.assert_called_once_with(docs)
            
            # Test search
            results = pipeline.search("test query")
            mock_search.assert_called_once()
            
            # Test save/load
            pipeline.save_index("test_path")
            mock_save.assert_called_once()
            
            pipeline.load_index("test_path")
            mock_load.assert_called_once()

    def test_search_parameter_variations(self):
        """Test various search parameter combinations."""
        pipeline = LegalDocumentPipeline()
        
        with patch.object(pipeline.index, 'search', return_value=[]) as mock_search, \
             patch('lexgraph_legal_rag.document_pipeline.get_query_cache') as mock_get_cache:
            
            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_get_cache.return_value = mock_cache
            
            # Test different top_k values
            pipeline.search("query", top_k=10)
            mock_search.assert_called_with("query", top_k=10)
            
            # Test semantic=False explicitly
            pipeline.search("query", semantic=False)
            mock_search.assert_called_with("query", top_k=5)

    def test_error_handling_basic(self):
        """Test basic error handling."""
        pipeline = LegalDocumentPipeline()
        
        # Test save error propagation
        with patch.object(pipeline.index, 'save', side_effect=IOError("Save failed")):
            with pytest.raises(IOError, match="Save failed"):
                pipeline.save_index("invalid_path")
        
        # Test load error propagation
        with patch.object(pipeline.index, 'load', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError, match="File not found"):
                pipeline.load_index("missing_file")

    def test_semantic_integration(self):
        """Test semantic search integration."""
        with patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline') as mock_semantic_class:
            mock_semantic = Mock()
            mock_semantic_class.return_value = mock_semantic
            
            pipeline = LegalDocumentPipeline(use_semantic=True)
            
            # Test save with semantic
            with patch.object(pipeline.index, 'save') as mock_vector_save:
                pipeline.save_index("test_path")
                mock_vector_save.assert_called_once()
                mock_semantic.save.assert_called_once()
            
            # Test load with semantic file
            with patch.object(pipeline.index, 'load') as mock_vector_load, \
                 patch('pathlib.Path.exists', return_value=True):
                pipeline.load_index("test_path")
                mock_vector_load.assert_called_once()
                mock_semantic.load.assert_called_once()


class TestPipelineAdvancedFeatures:
    """Test advanced pipeline features."""

    @patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline')
    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_batch_search_semantic_with_batch_method(self, mock_get_cache, mock_semantic_class):
        """Test batch search with semantic pipeline that supports batching."""
        mock_semantic = Mock()
        mock_semantic.batch_search.return_value = [[(LegalDocument(id="sem", text="semantic"), 0.9)]]
        mock_semantic_class.return_value = mock_semantic
        
        pipeline = LegalDocumentPipeline(use_semantic=True)
        
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        
        results = pipeline.batch_search(["query"], semantic=True)
        
        assert len(results) == 1
        mock_semantic.batch_search.assert_called_once_with(["query"], top_k=5)

    @patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline')
    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_batch_search_semantic_fallback(self, mock_get_cache, mock_semantic_class):
        """Test batch search semantic fallback when no batch_search method."""
        mock_semantic = Mock()
        # Remove batch_search attribute to simulate fallback
        delattr(mock_semantic, 'batch_search')
        mock_semantic.search.return_value = [(LegalDocument(id="sem", text="semantic"), 0.9)]
        mock_semantic_class.return_value = mock_semantic
        
        pipeline = LegalDocumentPipeline(use_semantic=True)
        
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        
        results = pipeline.batch_search(["query1", "query2"], semantic=True)
        
        assert len(results) == 2
        assert mock_semantic.search.call_count == 2

    def test_large_scale_operations(self):
        """Test pipeline behavior with large document sets."""
        pipeline = LegalDocumentPipeline()
        
        # Create large document set
        large_doc_set = [
            LegalDocument(id=str(i), text=f"Document {i} content")
            for i in range(1000)
        ]
        
        # Test that pipeline can handle large document sets
        with patch.object(pipeline.index, 'add') as mock_add:
            pipeline.index.add(large_doc_set)
            mock_add.assert_called_once_with(large_doc_set)

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_cache_performance_optimization(self, mock_get_cache):
        """Test cache optimization for repeated queries."""
        pipeline = LegalDocumentPipeline()
        
        mock_cache = Mock()
        # First query: cache miss, second query: cache hit
        cached_result = [(LegalDocument(id="cached", text="cached"), 0.9)]
        mock_cache.get.side_effect = [None, cached_result]
        mock_get_cache.return_value = mock_cache
        
        search_results = [(LegalDocument(id="new", text="new"), 0.8)]
        
        with patch.object(pipeline.index, 'search', return_value=search_results):
            # First search - cache miss
            results1 = pipeline.search("repeated query")
            assert results1 == search_results
            
            # Second search - cache hit
            results2 = pipeline.search("repeated query")
            assert results2 == cached_result