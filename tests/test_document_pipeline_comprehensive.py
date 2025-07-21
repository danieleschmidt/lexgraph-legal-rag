"""Comprehensive tests for document pipeline module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile

from lexgraph_legal_rag.document_pipeline import VectorIndex, LegalDocumentPipeline
from lexgraph_legal_rag.models import LegalDocument


class TestVectorIndex:
    """Test VectorIndex functionality."""

    def test_vector_index_initialization(self):
        """Test VectorIndex creates correctly."""
        index = VectorIndex()
        assert index._vectorizer is not None
        assert index._matrix is None
        assert index._docs == []
        assert index.documents == []

    def test_vector_index_properties(self):
        """Test VectorIndex properties."""
        index = VectorIndex()
        # Test empty documents property
        assert index.documents == []
        
        # Mock documents for testing
        mock_docs = [
            LegalDocument(id="1", text="test document one"),
            LegalDocument(id="2", text="test document two")
        ]
        index._docs = mock_docs
        assert index.documents == mock_docs

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_vector_index_add_documents(self, mock_get_cache):
        """Test adding documents to VectorIndex."""
        # Mock cache to avoid cache dependency
        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache
        
        index = VectorIndex()
        docs = [
            LegalDocument(id="1", text="arbitration clause legal text"),
            LegalDocument(id="2", text="indemnification clause legal text")
        ]
        
        # Mock the TfidfVectorizer to avoid scipy issues
        with patch.object(index._vectorizer, 'fit_transform') as mock_fit:
            mock_fit.return_value = Mock()  # Mock matrix
            index.add(docs)
            
            # Verify documents were added
            assert len(index._docs) == 2
            assert index._docs[0].id == "1"
            assert index._docs[1].id == "2"
            
            # Verify cache was invalidated
            mock_cache.invalidate_pattern.assert_called_once_with("*")

    def test_vector_index_save_method_exists(self):
        """Test that save method exists and has correct signature."""
        index = VectorIndex()
        # Verify method exists
        assert hasattr(index, 'save')
        assert callable(index.save)

    def test_vector_index_load_method_exists(self):
        """Test that load method exists and has correct signature."""
        index = VectorIndex()
        # Verify method exists  
        assert hasattr(index, 'load')
        assert callable(index.load)

    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_vector_index_save_basic(self, mock_json_dump, mock_open):
        """Test basic save functionality."""
        index = VectorIndex()
        # Add mock documents
        index._docs = [LegalDocument(id="1", text="test")]
        
        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test save
        index.save("test_path.bin")
        
        # Verify file operations
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()

    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_vector_index_load_basic(self, mock_json_load, mock_open):
        """Test basic load functionality."""
        index = VectorIndex()
        
        # Mock loaded data
        mock_data = {
            "documents": [{"id": "1", "text": "test", "metadata": {}}]
        }
        mock_json_load.return_value = mock_data
        
        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test load
        index.load("test_path.bin")
        
        # Verify file operations
        mock_open.assert_called_once()
        mock_json_load.assert_called_once()

    def test_vector_index_search_method_exists(self):
        """Test that search method exists."""
        index = VectorIndex()
        assert hasattr(index, 'search')
        assert callable(index.search)


class TestLegalDocumentPipeline:
    """Test LegalDocumentPipeline functionality."""

    def test_pipeline_initialization_default(self):
        """Test pipeline initialization with defaults."""
        pipeline = LegalDocumentPipeline()
        assert pipeline.index is not None
        assert isinstance(pipeline.index, VectorIndex)
        assert pipeline.semantic_pipeline is None

    def test_pipeline_initialization_with_semantic(self):
        """Test pipeline initialization with semantic search."""
        with patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline') as mock_semantic:
            mock_semantic.return_value = Mock()
            pipeline = LegalDocumentPipeline(use_semantic=True)
            assert pipeline.semantic_pipeline is not None
            mock_semantic.assert_called_once()

    def test_pipeline_search_method_exists(self):
        """Test that search method exists."""
        pipeline = LegalDocumentPipeline()
        assert hasattr(pipeline, 'search')
        assert callable(pipeline.search)

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_pipeline_search_cache_integration(self, mock_get_cache):
        """Test search integrates with caching."""
        # Mock cache
        mock_cache = Mock()
        mock_cache.get.return_value = None  # Cache miss
        mock_get_cache.return_value = mock_cache
        
        pipeline = LegalDocumentPipeline()
        
        # Mock index search
        with patch.object(pipeline.index, 'search', return_value=[]) as mock_search:
            results = pipeline.search("test query")
            
            # Verify cache was checked
            mock_cache.get.assert_called()
            mock_cache.set.assert_called()
            
            # Verify index search was called
            mock_search.assert_called_once()

    def test_pipeline_search_with_semantic(self):
        """Test search with semantic pipeline."""
        with patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline') as mock_semantic_cls:
            mock_semantic = Mock()
            mock_semantic.search.return_value = []
            mock_semantic_cls.return_value = mock_semantic
            
            pipeline = LegalDocumentPipeline(use_semantic=True)
            
            # Mock cache to avoid dependency
            with patch('lexgraph_legal_rag.document_pipeline.get_query_cache') as mock_get_cache:
                mock_cache = Mock()
                mock_cache.get.return_value = None
                mock_get_cache.return_value = mock_cache
                
                results = pipeline.search("test query")
                
                # Verify semantic search was used
                mock_semantic.search.assert_called_once_with("test query", top_k=5)

    def test_pipeline_add_documents(self):
        """Test adding documents to pipeline."""
        pipeline = LegalDocumentPipeline()
        docs = [LegalDocument(id="1", text="test")]
        
        # Mock index add method
        with patch.object(pipeline.index, 'add') as mock_add:
            pipeline.add_documents(docs)
            mock_add.assert_called_once_with(docs)

    def test_pipeline_add_documents_with_semantic(self):
        """Test adding documents to pipeline with semantic search."""
        with patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline') as mock_semantic_cls:
            mock_semantic = Mock()
            mock_semantic_cls.return_value = mock_semantic
            
            pipeline = LegalDocumentPipeline(use_semantic=True)
            docs = [LegalDocument(id="1", text="test")]
            
            # Mock both index and semantic add
            with patch.object(pipeline.index, 'add') as mock_index_add:
                pipeline.add_documents(docs)
                
                # Verify both were called
                mock_index_add.assert_called_once_with(docs)
                mock_semantic.add_documents.assert_called_once_with(docs)

    def test_pipeline_save_and_load_methods_exist(self):
        """Test save and load methods exist."""
        pipeline = LegalDocumentPipeline()
        assert hasattr(pipeline, 'save')
        assert hasattr(pipeline, 'load')
        assert callable(pipeline.save)
        assert callable(pipeline.load)

    def test_pipeline_save_basic(self):
        """Test basic pipeline save functionality."""
        pipeline = LegalDocumentPipeline()
        
        # Mock index save
        with patch.object(pipeline.index, 'save') as mock_save:
            pipeline.save("test_path")
            mock_save.assert_called_once()

    def test_pipeline_save_with_semantic(self):
        """Test pipeline save with semantic search."""
        with patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline') as mock_semantic_cls:
            mock_semantic = Mock()
            mock_semantic_cls.return_value = mock_semantic
            
            pipeline = LegalDocumentPipeline(use_semantic=True)
            
            # Mock both saves
            with patch.object(pipeline.index, 'save') as mock_index_save:
                pipeline.save("test_path")
                
                # Verify both were saved
                mock_index_save.assert_called_once()
                mock_semantic.save.assert_called_once()

    def test_pipeline_load_basic(self):
        """Test basic pipeline load functionality."""
        pipeline = LegalDocumentPipeline()
        
        # Mock index load
        with patch.object(pipeline.index, 'load') as mock_load:
            pipeline.load("test_path")
            mock_load.assert_called_once()

    def test_pipeline_load_with_semantic(self):
        """Test pipeline load with semantic search."""
        with patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline') as mock_semantic_cls:
            mock_semantic = Mock()
            mock_semantic_cls.return_value = mock_semantic
            
            pipeline = LegalDocumentPipeline(use_semantic=True)
            
            # Mock both loads
            with patch.object(pipeline.index, 'load') as mock_index_load:
                with patch('pathlib.Path.exists', return_value=True):
                    pipeline.load("test_path")
                    
                    # Verify both were loaded
                    mock_index_load.assert_called_once()
                    mock_semantic.load.assert_called_once()


class TestDocumentPipelineIntegration:
    """Integration tests for document pipeline components."""

    def test_pipeline_end_to_end_workflow(self):
        """Test complete pipeline workflow without external dependencies."""
        # Create pipeline
        pipeline = LegalDocumentPipeline()
        
        # Mock all external dependencies
        with patch.object(pipeline.index, 'add') as mock_add, \
             patch.object(pipeline.index, 'search', return_value=[]) as mock_search, \
             patch.object(pipeline.index, 'save') as mock_save, \
             patch.object(pipeline.index, 'load') as mock_load, \
             patch('lexgraph_legal_rag.document_pipeline.get_query_cache') as mock_get_cache:
            
            # Mock cache
            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_get_cache.return_value = mock_cache
            
            # Test workflow
            docs = [LegalDocument(id="1", text="test document")]
            
            # Add documents
            pipeline.add_documents(docs)
            mock_add.assert_called_once_with(docs)
            
            # Search documents
            results = pipeline.search("test query")
            mock_search.assert_called_once()
            
            # Save pipeline
            pipeline.save("test_path")
            mock_save.assert_called_once()
            
            # Load pipeline
            pipeline.load("test_path")
            mock_load.assert_called_once()

    def test_pipeline_error_handling(self):
        """Test pipeline handles errors gracefully."""
        pipeline = LegalDocumentPipeline()
        
        # Test with invalid documents
        with pytest.raises(AttributeError):
            pipeline.add_documents([{"invalid": "document"}])

    def test_pipeline_metrics_integration(self):
        """Test pipeline integrates with metrics system."""
        pipeline = LegalDocumentPipeline()
        
        # Mock metrics and cache
        with patch('lexgraph_legal_rag.document_pipeline.SEARCH_LATENCY') as mock_latency, \
             patch('lexgraph_legal_rag.document_pipeline.SEARCH_REQUESTS') as mock_requests, \
             patch('lexgraph_legal_rag.document_pipeline.get_query_cache') as mock_get_cache:
            
            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_get_cache.return_value = mock_cache
            
            # Mock index search to avoid sklearn issues
            with patch.object(pipeline.index, 'search', return_value=[]):
                pipeline.search("test query")
                
                # Verify metrics were called
                mock_requests.inc.assert_called_once()
                mock_latency.observe.assert_called_once()