"""Test coverage for models module."""

import pytest
from lexgraph_legal_rag.models import LegalDocument


def test_legal_document_creation():
    """Test LegalDocument model creation."""
    doc = LegalDocument(id="doc1", text="This is a legal document.")
    assert doc.id == "doc1"
    assert doc.text == "This is a legal document."
    assert doc.metadata == {}


def test_legal_document_with_metadata():
    """Test LegalDocument model with metadata."""
    metadata = {"source": "contract", "date": "2024-01-01"}
    doc = LegalDocument(
        id="doc2", 
        text="Contract terms", 
        metadata=metadata
    )
    assert doc.id == "doc2"
    assert doc.text == "Contract terms"
    assert doc.metadata == metadata


def test_legal_document_default_metadata():
    """Test LegalDocument model with default metadata."""
    doc = LegalDocument(id="doc3", text="Test document")
    assert doc.metadata == {}
    
    # Test that metadata is mutable
    doc.metadata["key"] = "value"
    assert doc.metadata["key"] == "value"