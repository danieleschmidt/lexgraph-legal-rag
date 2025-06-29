"""LexGraph Legal Retrieval Augmented Generation."""

from .multi_agent import CitationAgent, MultiAgentGraph, RouterAgent
from .document_pipeline import LegalDocumentPipeline
from .semantic_search import EmbeddingModel, SemanticSearchPipeline
from .faiss_index import FaissVectorIndex
from .context_reasoning import ContextAwareReasoner
from .models import LegalDocument
from .logging_config import configure_logging

__version__ = "1.0.0"

__all__ = [
    "MultiAgentGraph",
    "RouterAgent",
    "CitationAgent",
    "LegalDocument",
    "LegalDocumentPipeline",
    "EmbeddingModel",
    "SemanticSearchPipeline",
    "FaissVectorIndex",
    "ContextAwareReasoner",
    "configure_logging",
]
