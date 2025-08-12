"""LexGraph Legal Retrieval Augmented Generation with Bioneural Olfactory Fusion."""

from .multi_agent import CitationAgent, MultiAgentGraph, RouterAgent
from .document_pipeline import LegalDocumentPipeline
from .semantic_search import EmbeddingModel, SemanticSearchPipeline
from .faiss_index import FaissVectorIndex
from .context_reasoning import ContextAwareReasoner
from .models import LegalDocument
from .logging_config import configure_logging
from .bioneuro_olfactory_fusion import (
    BioneuroOlfactoryFusionEngine,
    DocumentScentProfile,
    OlfactoryReceptorType,
    get_fusion_engine,
    analyze_document_scent
)
from .multisensory_legal_processor import (
    MultiSensoryLegalProcessor,
    MultiSensoryAnalysis,
    SensoryChannel,
    get_multisensory_processor,
    analyze_document_multisensory
)

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
    "BioneuroOlfactoryFusionEngine",
    "DocumentScentProfile",
    "OlfactoryReceptorType",
    "get_fusion_engine",
    "analyze_document_scent",
    "MultiSensoryLegalProcessor",
    "MultiSensoryAnalysis",
    "SensoryChannel",
    "get_multisensory_processor",
    "analyze_document_multisensory",
]
