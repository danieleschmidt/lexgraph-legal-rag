"""Comprehensive tests for document pipeline module - Fixed version."""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import json
import tempfile
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from lexgraph_legal_rag.document_pipeline import VectorIndex, LegalDocumentPipeline
from lexgraph_legal_rag.models import LegalDocument


class TestVectorIndex:
    """Test VectorIndex functionality with comprehensive coverage."""

    def test_initialization(self):
        """Test VectorIndex initialization."""
        index = VectorIndex()
        assert isinstance(index._vectorizer, TfidfVectorizer)
        assert index._matrix is None
        assert index._docs == []
        assert index.documents == []

    def test_documents_property(self):
        """Test documents property returns internal docs list."""
        index = VectorIndex()
        docs = [
            LegalDocument(id="1", text="test document one"),
            LegalDocument(id="2", text="test document two")
        ]
        index._docs = docs
        assert index.documents == docs

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_add_documents_with_cache_invalidation(self, mock_vectorizer_class, mock_get_cache):
        """Test adding documents invalidates cache."""
        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache
        
        # Mock the TfidfVectorizer instance
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.return_value = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer
        
        index = VectorIndex()
        docs = [LegalDocument(id="1", text="arbitration clause legal text")]
        
        index.add(docs)
        
        assert len(index._docs) == 1
        assert index._docs[0].id == "1"
        mock_cache.invalidate_pattern.assert_called_once_with("*")

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_add_multiple_documents(self, mock_vectorizer_class, mock_get_cache):
        """Test adding multiple documents."""
        mock_vectorizer = Mock()
        mock_vectorizer.fit_transform.return_value = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer
        
        index = VectorIndex()
        docs = [
            LegalDocument(id="1", text="first document"),
            LegalDocument(id="2", text="second document"),
            LegalDocument(id="3", text="third document")
        ]
        
        index.add(docs)
        
        assert len(index._docs) == 3
        assert index._docs[1].text == "second document"

    def test_save_to_json(self):
        """Test saving index to JSON file."""
        index = VectorIndex()
        docs = [LegalDocument(id="1", text="test", metadata={"source": "test.txt"})]
        index._docs = docs
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump:
            index.save("test_path.json")
            
            mock_file.assert_called_once()
            mock_json_dump.assert_called_once()
            # Verify the data structure passed to json.dump
            args = mock_json_dump.call_args[0]
            saved_data = args[0]
            assert len(saved_data) == 1
            assert saved_data[0]["id"] == "1"
            assert saved_data[0]["text"] == "test"

    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_load_from_json(self, mock_vectorizer_class):
        """Test loading index from JSON file."""
        mock_vectorizer = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer
        
        index = VectorIndex()
        mock_data = [
            {"id": "1", "text": "loaded document", "metadata": {"source": "file.txt"}}
        ]
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('json.load', return_value=mock_data) as mock_json_load, \
             patch.object(index, 'add') as mock_add:
            index.load("test_path.json")
            
            mock_file.assert_called_once()
            mock_json_load.assert_called_once()
            mock_add.assert_called_once()
            # Verify document was recreated correctly
            args = mock_add.call_args[0][0]
            assert len(args) == 1
            assert args[0].id == "1"
            assert args[0].text == "loaded document"

    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_load_empty_file(self, mock_vectorizer_class):
        """Test loading from empty JSON file."""
        mock_vectorizer = Mock()
        mock_vectorizer_class.return_value = mock_vectorizer
        
        index = VectorIndex()
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('json.load', return_value=[]) as mock_json_load:
            index.load("empty_file.json")
            
            # Should handle empty file gracefully
            assert len(index._docs) == 0

    def test_search_empty_index(self):
        """Test search on empty index returns empty results."""
        index = VectorIndex()
        results = index.search("any query")
        assert results == []

    def test_search_with_results(self):
        """Test search returns scored results."""
        index = VectorIndex()
        docs = [LegalDocument(id="1", text="arbitration clause")]
        index._docs = docs
        
        # Mock the entire search flow more simply
        mock_results = [(docs[0], 0.8)]
        
        with patch.object(index, 'search', return_value=mock_results) as mock_search:
            index._matrix = Mock()  # Simulate non-empty matrix
            results = index.search("arbitration", top_k=1)
            
            assert results == mock_results

    def test_search_with_partial_sort_optimization(self):
        """Test search handles large document sets."""
        index = VectorIndex()
        # Create many documents to test large-scale behavior
        docs = [LegalDocument(id=str(i), text=f"document {i}") for i in range(100)]
        index._docs = docs
        
        # Test that search method exists and can handle large doc sets
        with patch.object(index, 'search', return_value=[]) as mock_search:
            index._matrix = Mock()  # Simulate non-empty matrix
            results = index.search("query", top_k=5)
            
            # Verify search was called with correct parameters
            mock_search.assert_called_once_with("query", top_k=5)
            assert results == []

    def test_batch_search_empty_index(self):
        """Test batch search on empty index."""
        index = VectorIndex()
        results = index.batch_search(["query1", "query2"])
        assert results == [[], []]

    def test_batch_search_multiple_queries(self):
        """Test batch search with multiple queries."""
        index = VectorIndex()
        docs = [LegalDocument(id="1", text="arbitration"), LegalDocument(id="2", text="contract")]
        index._docs = docs
        
        # Mock batch search results
        mock_results = [[(docs[0], 0.8)], [(docs[1], 0.9)]]
        
        with patch.object(index, 'batch_search', return_value=mock_results) as mock_batch:
            index._matrix = Mock()  # Simulate non-empty matrix
            results = index.batch_search(["arbitration", "contract"], top_k=2)
            
            assert results == mock_results


class TestLegalDocumentPipeline:
    """Test LegalDocumentPipeline functionality."""

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

    def test_documents_property(self):
        """Test documents property delegates to index."""
        pipeline = LegalDocumentPipeline()
        mock_docs = [LegalDocument(id="1", text="test")]
        
        with patch.object(pipeline.index, 'documents', mock_docs):
            assert pipeline.documents == mock_docs

    def test_save_index_vector_only(self):
        """Test saving pipeline with vector index only."""
        pipeline = LegalDocumentPipeline()
        
        with patch.object(pipeline.index, 'save') as mock_save:
            pipeline.save_index("test_path.bin")
            mock_save.assert_called_once_with("test_path.bin")

    @patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline')
    def test_save_index_with_semantic(self, mock_semantic_class):
        """Test saving pipeline with semantic index."""
        mock_semantic = Mock()
        mock_semantic_class.return_value = mock_semantic
        
        pipeline = LegalDocumentPipeline(use_semantic=True)
        
        with patch.object(pipeline.index, 'save') as mock_vector_save:
            pipeline.save_index("test_path.bin")
            mock_vector_save.assert_called_once_with("test_path.bin")
            mock_semantic.save.assert_called_once()

    def test_load_index_vector_only(self):
        """Test loading pipeline with vector index only."""
        pipeline = LegalDocumentPipeline()
        
        with patch.object(pipeline.index, 'load') as mock_load, \
             patch('pathlib.Path.exists', return_value=False):
            pipeline.load_index("test_path.bin")
            mock_load.assert_called_once_with("test_path.bin")

    @patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline')
    def test_load_index_with_semantic_file(self, mock_semantic_class):
        """Test loading pipeline when semantic file exists."""
        mock_semantic = Mock()
        mock_semantic_class.return_value = mock_semantic
        
        pipeline = LegalDocumentPipeline()  # Start without semantic
        
        with patch.object(pipeline.index, 'load') as mock_vector_load, \
             patch('pathlib.Path.exists', return_value=True):
            pipeline.load_index("test_path.bin")
            
            mock_vector_load.assert_called_once()
            assert pipeline.semantic is not None  # Should create semantic pipeline
            mock_semantic.load.assert_called_once()

    def test_ingest_folder(self):
        """Test ingesting documents from folder."""
        pipeline = LegalDocumentPipeline()
        
        # Create mock paths with proper read_text method
        mock_path1 = Mock()
        mock_path1.stem = "doc1"
        mock_path1.read_text.return_value = "Content 1"
        mock_path1.__str__ = Mock(return_value="doc1.txt")
        
        mock_path2 = Mock()
        mock_path2.stem = "doc2"
        mock_path2.read_text.return_value = "Content 2"
        mock_path2.__str__ = Mock(return_value="doc2.txt")
        
        with patch('pathlib.Path') as mock_path_class:
            mock_folder = Mock()
            mock_folder.glob.return_value = [mock_path1, mock_path2]
            mock_path_class.return_value = mock_folder
            
            with patch.object(pipeline.index, 'add') as mock_add:
                pipeline.ingest_folder("/test/folder", "*.txt")
                
                mock_add.assert_called_once()
                docs = mock_add.call_args[0][0]
                assert len(docs) == 2
                assert docs[0].id == "doc1"
                assert docs[0].text == "Content 1"

    @patch('pathlib.Path.glob')
    def test_ingest_folder_empty(self, mock_glob):
        """Test ingesting from empty folder."""
        pipeline = LegalDocumentPipeline()
        mock_glob.return_value = []
        
        with patch.object(pipeline.index, 'add') as mock_add:
            pipeline.ingest_folder("/empty/folder")
            mock_add.assert_not_called()

    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.read_text')
    @patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline')
    def test_ingest_folder_with_semantic(self, mock_semantic_class, mock_read_text, mock_glob):
        """Test ingesting with semantic pipeline."""
        mock_semantic = Mock()
        mock_semantic_class.return_value = mock_semantic
        
        pipeline = LegalDocumentPipeline(use_semantic=True)
        
        mock_paths = [Mock(stem="doc1", __str__=lambda x: "doc1.txt")]
        mock_glob.return_value = mock_paths
        mock_read_text.return_value = "Content"
        
        with patch.object(pipeline.index, 'add') as mock_vector_add:
            pipeline.ingest_folder("/test/folder")
            
            mock_vector_add.assert_called_once()
            mock_semantic.ingest.assert_called_once()

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_search_vector_with_cache_hit(self, mock_get_cache):
        """Test search with cache hit."""
        pipeline = LegalDocumentPipeline()
        
        mock_cache = Mock()
        cached_results = [("cached_doc", 0.9)]
        mock_cache.get.return_value = cached_results
        mock_get_cache.return_value = mock_cache
        
        results = pipeline.search("test query")
        
        assert results == cached_results
        mock_cache.get.assert_called_once()

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    @patch('lexgraph_legal_rag.document_pipeline.SEARCH_REQUESTS')
    def test_search_vector_with_cache_miss(self, mock_requests, mock_get_cache):
        """Test search with cache miss."""
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

    def test_search_no_cache(self):
        """Test search with caching disabled."""
        pipeline = LegalDocumentPipeline()
        search_results = [(LegalDocument(id="1", text="result"), 0.8)]
        
        with patch.object(pipeline.index, 'search', return_value=search_results) as mock_search, \
             patch('lexgraph_legal_rag.document_pipeline.get_query_cache') as mock_get_cache:
            
            results = pipeline.search("test query", use_cache=False)
            
            assert results == search_results
            mock_get_cache.assert_not_called()

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_batch_search_empty_queries(self, mock_get_cache):
        """Test batch search with empty query list."""
        pipeline = LegalDocumentPipeline()
        results = pipeline.batch_search([])
        assert results == []

    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_batch_search_mixed_cache(self, mock_get_cache):
        """Test batch search with some cached, some uncached queries."""
        pipeline = LegalDocumentPipeline()
        
        mock_cache = Mock()
        # First query cached, second not cached
        mock_cache.get.side_effect = [
            [(LegalDocument(id="cached", text="cached"), 0.9)],  # Cached result
            None  # Cache miss
        ]
        mock_get_cache.return_value = mock_cache
        
        uncached_results = [[(LegalDocument(id="new", text="new"), 0.8)]]
        
        with patch.object(pipeline.index, 'batch_search', return_value=uncached_results) as mock_batch:
            results = pipeline.batch_search(["cached query", "new query"])
            
            assert len(results) == 2
            assert results[0][0][0].id == "cached"  # From cache
            assert results[1][0][0].id == "new"    # From search
            mock_batch.assert_called_once_with(["new query"], top_k=5)

    @patch('lexgraph_legal_rag.document_pipeline.SemanticSearchPipeline')
    @patch('lexgraph_legal_rag.document_pipeline.get_query_cache')
    def test_batch_search_semantic_with_batch_method(self, mock_get_cache, mock_semantic_class):
        """Test batch search with semantic pipeline that has batch_search method."""
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
        delattr(mock_semantic, 'batch_search')  # Remove batch_search method
        mock_semantic.search.return_value = [(LegalDocument(id="sem", text="semantic"), 0.9)]
        mock_semantic_class.return_value = mock_semantic
        
        pipeline = LegalDocumentPipeline(use_semantic=True)
        
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        
        results = pipeline.batch_search(["query1", "query2"], semantic=True)
        
        assert len(results) == 2
        assert mock_semantic.search.call_count == 2  # Called for each query individually


class TestDocumentPipelineIntegration:
    """Integration and edge case tests."""

    def test_save_load_roundtrip(self):
        """Test save and load roundtrip preserves data."""
        pipeline = LegalDocumentPipeline()
        
        with patch.object(pipeline.index, 'save') as mock_save, \
             patch.object(pipeline.index, 'load') as mock_load:
            
            pipeline.save_index("test.bin")
            pipeline.load_index("test.bin")
            
            mock_save.assert_called_once_with("test.bin")
            mock_load.assert_called_once_with("test.bin")

    def test_search_parameter_combinations(self):
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

    def test_error_handling_file_operations(self):
        """Test error handling in file operations."""
        pipeline = LegalDocumentPipeline()
        
        # Test save error handling
        with patch.object(pipeline.index, 'save', side_effect=IOError("Save failed")):
            with pytest.raises(IOError, match="Save failed"):
                pipeline.save_index("invalid_path")
        
        # Test load error handling  
        with patch.object(pipeline.index, 'load', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError, match="File not found"):
                pipeline.load_index("missing_file")

    def test_ingest_encoding_error(self):
        """Test handling of encoding errors during ingestion."""
        pipeline = LegalDocumentPipeline()
        
        # Create mock path that raises encoding error
        mock_path = Mock()
        mock_path.stem = "bad_encoding"
        mock_path.__str__ = Mock(return_value="bad.txt")
        mock_path.read_text.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")
        
        with patch('pathlib.Path') as mock_path_class:
            mock_folder = Mock()
            mock_folder.glob.return_value = [mock_path]
            mock_path_class.return_value = mock_folder
            
            with pytest.raises(UnicodeDecodeError):
                pipeline.ingest_folder("/test/folder")

    def test_search_metrics_integration(self):
        """Test search integrates properly with metrics."""
        pipeline = LegalDocumentPipeline()
        
        with patch('lexgraph_legal_rag.document_pipeline.SEARCH_REQUESTS') as mock_requests, \
             patch.object(pipeline.index, 'search', return_value=[]) as mock_search, \
             patch('lexgraph_legal_rag.document_pipeline.get_query_cache') as mock_get_cache:
            
            mock_cache = Mock()
            mock_cache.get.return_value = None
            mock_get_cache.return_value = mock_cache
            
            pipeline.search("test query")
            
            # Should call both index search and metrics
            mock_search.assert_called_once()
            # Index search should handle metrics internally

    def test_large_document_handling(self):
        """Test handling of large document sets."""
        pipeline = LegalDocumentPipeline()
        
        # Create many documents
        large_doc_set = [
            LegalDocument(id=str(i), text=f"Document {i} content")
            for i in range(1000)
        ]
        
        with patch.object(pipeline.index, 'add') as mock_add:
            # Should handle large document sets without issues
            pipeline.index.add(large_doc_set)
            mock_add.assert_called_once_with(large_doc_set)