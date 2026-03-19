"""LexGraph Legal RAG — FAISS-based legal document retrieval for DocGraph."""

from __future__ import annotations

from .document_pipeline import LegalDocumentPipeline
from .faiss_index import FaissVectorIndex
from .indexer import LegalDocumentIndexer
from .logging_config import configure_logging
from .models import LegalDocument
from .multi_agent import CitationAgent
from .multi_agent import MultiAgentGraph
from .multi_agent import RouterAgent
from .retriever import LegalRetriever
from .retriever import MultiAgentRetriever


__version__ = "2.0.0"

__all__ = [
    "CitationAgent",
    "FaissVectorIndex",
    "LegalDocument",
    "LegalDocumentIndexer",
    "LegalDocumentPipeline",
    "LegalRetriever",
    "MultiAgentGraph",
    "MultiAgentRetriever",
    "RouterAgent",
    "configure_logging",
]
