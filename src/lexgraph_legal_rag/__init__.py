"""LexGraph Legal Retrieval Augmented Generation."""

import logging

from .multi_agent import CitationAgent, MultiAgentGraph, RouterAgent
from .document_pipeline import LegalDocumentPipeline
from .semantic_search import EmbeddingModel, SemanticSearchPipeline
from .context_reasoning import ContextAwareReasoner
from .models import LegalDocument

logging.basicConfig(level=logging.INFO)

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
