"""LexGraph Legal Retrieval Augmented Generation with Bioneural Olfactory Fusion."""

from .bioneuro_olfactory_fusion import BioneuroOlfactoryFusionEngine
from .bioneuro_olfactory_fusion import DocumentScentProfile
from .bioneuro_olfactory_fusion import OlfactoryReceptorType
from .bioneuro_olfactory_fusion import analyze_document_scent
from .bioneuro_olfactory_fusion import get_fusion_engine
from .context_reasoning import ContextAwareReasoner
from .document_pipeline import LegalDocumentPipeline
from .faiss_index import FaissVectorIndex
from .logging_config import configure_logging
from .models import LegalDocument
from .multi_agent import CitationAgent
from .multi_agent import MultiAgentGraph
from .multi_agent import RouterAgent
from .multisensory_legal_processor import MultiSensoryAnalysis
from .multisensory_legal_processor import MultiSensoryLegalProcessor
from .multisensory_legal_processor import SensoryChannel
from .multisensory_legal_processor import analyze_document_multisensory
from .multisensory_legal_processor import get_multisensory_processor
from .semantic_search import EmbeddingModel
from .semantic_search import SemanticSearchPipeline


__version__ = "1.0.0"

__all__ = [
    "BioneuroOlfactoryFusionEngine",
    "CitationAgent",
    "ContextAwareReasoner",
    "DocumentScentProfile",
    "EmbeddingModel",
    "FaissVectorIndex",
    "LegalDocument",
    "LegalDocumentPipeline",
    "MultiAgentGraph",
    "MultiSensoryAnalysis",
    "MultiSensoryLegalProcessor",
    "OlfactoryReceptorType",
    "RouterAgent",
    "SemanticSearchPipeline",
    "SensoryChannel",
    "analyze_document_multisensory",
    "analyze_document_scent",
    "configure_logging",
    "get_fusion_engine",
    "get_multisensory_processor",
]
