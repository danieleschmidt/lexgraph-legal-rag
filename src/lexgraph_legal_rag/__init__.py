"""LexGraph Legal Retrieval Augmented Generation."""

from .multi_agent import CitationAgent, MultiAgentGraph, RouterAgent
from .document_pipeline import LegalDocumentPipeline
from .semantic_search import EmbeddingModel, SemanticSearchPipeline
from .context_reasoning import ContextAwareReasoner
from .models import LegalDocument

__version__ = "0.1.0"

__all__ = [
    "MultiAgentGraph",
    "RouterAgent",
    "CitationAgent",
    "LegalDocument",
    "LegalDocumentPipeline",
    "EmbeddingModel",
    "SemanticSearchPipeline",
    "ContextAwareReasoner",
]
