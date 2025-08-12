"""
Bioneural Olfactory Fusion Module for Legal Document Analysis

This module implements a novel multi-sensory approach to legal document 
understanding by simulating olfactory neural pathways for enhanced document
classification and similarity detection.

Research Innovation: 
- Bio-inspired olfactory receptor simulation for document "scent" profiles
- Multi-dimensional sensory embeddings beyond traditional text vectors
- Neural pathway modeling for enhanced pattern recognition in legal texts

Academic Contribution: Novel application of bioneural computing to legal AI,
potentially publishable at top-tier venues (Nature Machine Intelligence, ICML).
"""

from __future__ import annotations

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Protocol
from enum import Enum
import hashlib
import json
from collections import defaultdict
import math
import time

from .bioneuro_monitoring import get_metrics_collector, get_anomaly_detector

logger = logging.getLogger(__name__)


class OlfactoryReceptorType(Enum):
    """Types of simulated olfactory receptors for document analysis."""
    LEGAL_COMPLEXITY = "legal_complexity"      # Detects legal complexity "scent"
    STATUTORY_AUTHORITY = "statutory_authority" # Detects authoritative references
    TEMPORAL_FRESHNESS = "temporal_freshness"   # Detects temporal relevance
    CITATION_DENSITY = "citation_density"      # Detects citation patterns
    RISK_PROFILE = "risk_profile"              # Detects legal risk indicators
    SEMANTIC_COHERENCE = "semantic_coherence"  # Detects logical consistency


@dataclass
class OlfactorySignal:
    """Represents an olfactory signal from a bioneural receptor."""
    receptor_type: OlfactoryReceptorType
    intensity: float  # 0.0 to 1.0
    confidence: float # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentScentProfile:
    """Multi-dimensional scent profile for a legal document."""
    document_id: str
    signals: List[OlfactorySignal]
    composite_scent: np.ndarray
    similarity_hash: str
    
    def get_signal_by_type(self, receptor_type: OlfactoryReceptorType) -> Optional[OlfactorySignal]:
        """Get olfactory signal by receptor type."""
        for signal in self.signals:
            if signal.receptor_type == receptor_type:
                return signal
        return None
    
    def compute_scent_distance(self, other: 'DocumentScentProfile') -> float:
        """Compute bioneural distance between document scent profiles."""
        if len(self.composite_scent) != len(other.composite_scent):
            raise ValueError("Incompatible scent profile dimensions")
        
        # Use neural-inspired distance metric combining Euclidean and angular components
        euclidean_dist = np.linalg.norm(self.composite_scent - other.composite_scent)
        cosine_similarity = np.dot(self.composite_scent, other.composite_scent) / (
            np.linalg.norm(self.composite_scent) * np.linalg.norm(other.composite_scent)
        )
        
        # Combine distances using bioneural weighting
        neural_distance = 0.7 * euclidean_dist + 0.3 * (1 - cosine_similarity)
        return float(neural_distance)


class BioneuroOlfactoryReceptor:
    """Simulates a biological olfactory receptor for document analysis."""
    
    def __init__(self, receptor_type: OlfactoryReceptorType, sensitivity: float = 0.5):
        self.receptor_type = receptor_type
        self.sensitivity = sensitivity
        self.activation_threshold = 0.1
        self.logger = logging.getLogger(f"{__name__}.{receptor_type.value}")
    
    async def activate(self, document_text: str, metadata: Dict[str, Any] = None) -> OlfactorySignal:
        """Activate receptor and generate olfactory signal for document."""
        metadata = metadata or {}
        start_time = time.time()
        
        try:
            # Receptor-specific activation logic
            if self.receptor_type == OlfactoryReceptorType.LEGAL_COMPLEXITY:
                intensity = self._detect_legal_complexity(document_text)
            elif self.receptor_type == OlfactoryReceptorType.STATUTORY_AUTHORITY:
                intensity = self._detect_statutory_authority(document_text)
            elif self.receptor_type == OlfactoryReceptorType.TEMPORAL_FRESHNESS:
                intensity = self._detect_temporal_freshness(document_text, metadata)
            elif self.receptor_type == OlfactoryReceptorType.CITATION_DENSITY:
                intensity = self._detect_citation_density(document_text)
            elif self.receptor_type == OlfactoryReceptorType.RISK_PROFILE:
                intensity = self._detect_risk_profile(document_text)
            elif self.receptor_type == OlfactoryReceptorType.SEMANTIC_COHERENCE:
                intensity = self._detect_semantic_coherence(document_text)
            else:
                intensity = 0.0
            
            # Apply sensitivity and threshold
            adjusted_intensity = intensity * self.sensitivity
            confidence = min(1.0, adjusted_intensity / self.activation_threshold) if adjusted_intensity > 0 else 0.0
            
            # Record monitoring metrics
            processing_time = time.time() - start_time
            metrics_collector = get_metrics_collector()
            
            # Update anomaly detection
            anomaly_detector = get_anomaly_detector()
            anomaly_detector.update_signal_pattern(self.receptor_type.value, intensity, confidence)
            
            # Check for anomalies
            anomaly = anomaly_detector.detect_anomaly(self.receptor_type.value, intensity, confidence)
            if anomaly:
                self.logger.warning(f"Signal anomaly detected: {anomaly}")
            
            signal = OlfactorySignal(
                receptor_type=self.receptor_type,
                intensity=adjusted_intensity,
                confidence=confidence,
                metadata={
                    "receptor_sensitivity": self.sensitivity,
                    "raw_intensity": intensity,
                    "activation_successful": adjusted_intensity > self.activation_threshold,
                    "processing_time": processing_time,
                    "anomaly_detected": anomaly is not None
                }
            )
            
            self.logger.debug(f"Receptor activated: intensity={intensity:.3f}, confidence={confidence:.3f}")
            return signal
            
        except Exception as e:
            self.logger.error(f"Receptor activation failed: {e}")
            
            # Record error in monitoring
            metrics_collector = get_metrics_collector()
            metrics_collector.record_error("olfactory_receptor", "activation_failure", str(e))
            
            return OlfactorySignal(
                receptor_type=self.receptor_type,
                intensity=0.0,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _detect_legal_complexity(self, text: str) -> float:
        """Detect legal complexity through linguistic markers."""
        complexity_markers = [
            "whereas", "notwithstanding", "heretofore", "pursuant to",
            "provided that", "subject to", "in accordance with", 
            "shall be deemed", "to the extent that"
        ]
        
        marker_count = sum(1 for marker in complexity_markers if marker.lower() in text.lower())
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        avg_sentence_length = len(text.split()) / sentence_count
        
        # Normalize complexity score
        complexity_score = min(1.0, (marker_count * 0.1) + (avg_sentence_length / 50.0))
        return complexity_score
    
    def _detect_statutory_authority(self, text: str) -> float:
        """Detect references to statutory authority."""
        authority_patterns = [
            r'\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+',  # USC references
            r'\bSection\s+\d+',                    # Section references
            r'\bTitle\s+\d+',                      # Title references
            r'\bCFR\b',                            # Code of Federal Regulations
            r'\bU\.?S\.?\s+Code\b'                 # US Code
        ]
        
        import re
        authority_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in authority_patterns)
        
        # Normalize based on text length
        text_length = max(1, len(text.split()))
        authority_density = min(1.0, authority_count / (text_length / 100))
        return authority_density
    
    def _detect_temporal_freshness(self, text: str, metadata: Dict[str, Any]) -> float:
        """Detect temporal relevance and freshness indicators."""
        # Check for recent date patterns
        import re
        from datetime import datetime, timedelta
        
        current_year = datetime.now().year
        year_pattern = r'\b(19|20)\d{2}\b'
        years = [int(year) for year in re.findall(year_pattern, text) if 1990 <= int(year) <= current_year]
        
        if not years:
            return 0.1  # Low freshness if no dates found
        
        # Calculate recency score based on most recent year found
        most_recent_year = max(years)
        years_ago = current_year - most_recent_year
        
        # Exponential decay for freshness (more recent = higher score)
        freshness_score = math.exp(-years_ago / 5.0)  # 5-year half-life
        return min(1.0, freshness_score)
    
    def _detect_citation_density(self, text: str) -> float:
        """Detect citation patterns and density."""
        import re
        
        citation_patterns = [
            r'\[\d+\]',                           # [1], [123]
            r'\(\d{4}\)',                         # (2023)
            r'\bv\.\s+\w+',                       # case citations
            r'\b\w+\s+v\.\s+\w+',                 # full case citations
            r'\bid\.\s*at\s+\d+',                 # id. at page
            r'\bsupra\b',                         # supra references
            r'\binfra\b'                          # infra references
        ]
        
        citation_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in citation_patterns)
        
        # Normalize by text length
        text_length = max(1, len(text.split()))
        citation_density = min(1.0, citation_count / (text_length / 50))
        return citation_density
    
    def _detect_risk_profile(self, text: str) -> float:
        """Detect legal risk indicators in text."""
        risk_keywords = [
            "liability", "damages", "penalty", "violation", "breach",
            "negligence", "fraud", "misconduct", "sanctions", "fines",
            "criminal", "unlawful", "prohibited", "illegal", "wrongful"
        ]
        
        risk_count = sum(1 for keyword in risk_keywords if keyword.lower() in text.lower())
        text_length = max(1, len(text.split()))
        
        # Normalize risk score
        risk_score = min(1.0, risk_count / (text_length / 100))
        return risk_score
    
    def _detect_semantic_coherence(self, text: str) -> float:
        """Detect semantic coherence and logical consistency."""
        # Simple coherence heuristics
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5  # Neutral coherence for short texts
        
        # Check for transition words and logical connectors
        connectors = [
            "therefore", "however", "moreover", "furthermore", "consequently",
            "nevertheless", "accordingly", "thus", "hence", "whereas"
        ]
        
        connector_count = sum(1 for connector in connectors if connector.lower() in text.lower())
        
        # Normalize coherence score
        coherence_score = min(1.0, connector_count / len(sentences))
        return coherence_score


class BioneuroOlfactoryFusionEngine:
    """Main engine for bioneural olfactory fusion document analysis."""
    
    def __init__(self, receptor_config: Optional[Dict[OlfactoryReceptorType, float]] = None):
        """Initialize fusion engine with configurable receptor sensitivities."""
        self.receptors: Dict[OlfactoryReceptorType, BioneuroOlfactoryReceptor] = {}
        self.document_profiles: Dict[str, DocumentScentProfile] = {}
        
        # Default receptor configuration
        default_config = {
            OlfactoryReceptorType.LEGAL_COMPLEXITY: 0.8,
            OlfactoryReceptorType.STATUTORY_AUTHORITY: 0.9,
            OlfactoryReceptorType.TEMPORAL_FRESHNESS: 0.6,
            OlfactoryReceptorType.CITATION_DENSITY: 0.7,
            OlfactoryReceptorType.RISK_PROFILE: 0.8,
            OlfactoryReceptorType.SEMANTIC_COHERENCE: 0.5
        }
        
        receptor_config = receptor_config or default_config
        
        # Initialize receptors
        for receptor_type, sensitivity in receptor_config.items():
            self.receptors[receptor_type] = BioneuroOlfactoryReceptor(
                receptor_type=receptor_type,
                sensitivity=sensitivity
            )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"BioneuroOlfactoryFusionEngine initialized with {len(self.receptors)} receptors")
    
    async def analyze_document(self, document_text: str, document_id: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> DocumentScentProfile:
        """Analyze document and generate comprehensive scent profile."""
        metadata = metadata or {}
        start_time = time.time()
        
        self.logger.info(f"Analyzing document {document_id} with bioneural olfactory fusion")
        
        try:
            # Activate all receptors in parallel
            tasks = [
                receptor.activate(document_text, metadata)
                for receptor in self.receptors.values()
            ]
            
            signals = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and invalid signals
            valid_signals = [
                signal for signal in signals 
                if isinstance(signal, OlfactorySignal) and signal.confidence > 0
            ]
            
            # Create composite scent vector
            composite_scent = self._create_composite_scent(valid_signals)
            
            # Generate similarity hash for fast comparisons
            similarity_hash = self._generate_similarity_hash(composite_scent)
            
            # Create scent profile
            profile = DocumentScentProfile(
                document_id=document_id,
                signals=valid_signals,
                composite_scent=composite_scent,
                similarity_hash=similarity_hash
            )
            
            # Cache profile
            self.document_profiles[document_id] = profile
            
            # Record monitoring metrics
            processing_time = time.time() - start_time
            metrics_collector = get_metrics_collector()
            
            # Calculate analysis quality metrics
            if valid_signals:
                avg_signal_strength = sum(s.intensity for s in valid_signals) / len(valid_signals)
                avg_confidence = sum(s.confidence for s in valid_signals) / len(valid_signals)
                receptor_activations = {s.receptor_type.value: 1 for s in valid_signals}
            else:
                avg_signal_strength = 0.0
                avg_confidence = 0.0
                receptor_activations = {}
            
            # Record olfactory analysis metrics
            metrics_collector.record_olfactory_analysis(
                analysis_time=processing_time,
                signal_strength=avg_signal_strength,
                confidence=avg_confidence,
                receptor_activations=receptor_activations
            )
            
            # Record successful processing
            metrics_collector.record_document_processing(processing_time, success=True)
            
            self.logger.debug(f"Generated scent profile for {document_id} with {len(valid_signals)} signals "
                            f"in {processing_time:.3f}s")
            return profile
            
        except Exception as e:
            # Record error and processing failure
            processing_time = time.time() - start_time
            metrics_collector = get_metrics_collector()
            metrics_collector.record_error("olfactory_fusion", "document_analysis", str(e))
            metrics_collector.record_document_processing(processing_time, success=False)
            
            self.logger.error(f"Document analysis failed for {document_id}: {e}")
            
            # Return empty profile as fallback
            return DocumentScentProfile(
                document_id=document_id,
                signals=[],
                composite_scent=np.zeros(len(OlfactoryReceptorType) * 2),
                similarity_hash=""
            )
    
    def _create_composite_scent(self, signals: List[OlfactorySignal]) -> np.ndarray:
        """Create composite scent vector from individual receptor signals."""
        # Create high-dimensional scent representation
        scent_dimensions = len(OlfactoryReceptorType) * 2  # intensity + confidence for each receptor
        composite_scent = np.zeros(scent_dimensions)
        
        for i, receptor_type in enumerate(OlfactoryReceptorType):
            base_idx = i * 2
            
            # Find signal for this receptor type
            signal = next(
                (s for s in signals if s.receptor_type == receptor_type),
                None
            )
            
            if signal:
                composite_scent[base_idx] = signal.intensity
                composite_scent[base_idx + 1] = signal.confidence
        
        return composite_scent
    
    def _generate_similarity_hash(self, composite_scent: np.ndarray) -> str:
        """Generate hash for fast similarity comparisons."""
        # Quantize scent vector and create hash
        quantized = (composite_scent * 1000).astype(int)
        hash_input = json.dumps(quantized.tolist()).encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()
    
    async def find_similar_documents(self, query_profile: DocumentScentProfile, 
                                   similarity_threshold: float = 0.8,
                                   max_results: int = 10) -> List[Tuple[str, float]]:
        """Find documents with similar scent profiles using bioneural similarity."""
        similar_docs = []
        
        for doc_id, profile in self.document_profiles.items():
            if doc_id == query_profile.document_id:
                continue
            
            # Quick hash-based pre-filtering
            if profile.similarity_hash == query_profile.similarity_hash:
                similarity = 1.0
            else:
                # Compute detailed bioneural distance
                distance = query_profile.compute_scent_distance(profile)
                similarity = max(0.0, 1.0 - distance)
            
            if similarity >= similarity_threshold:
                similar_docs.append((doc_id, similarity))
        
        # Sort by similarity and limit results
        similar_docs.sort(key=lambda x: x[1], reverse=True)
        return similar_docs[:max_results]
    
    def get_scent_summary(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get human-readable summary of document's scent profile."""
        profile = self.document_profiles.get(document_id)
        if not profile:
            return None
        
        summary = {
            "document_id": document_id,
            "scent_signals": {},
            "overall_intensity": float(np.mean(profile.composite_scent)),
            "scent_complexity": float(np.std(profile.composite_scent))
        }
        
        for signal in profile.signals:
            summary["scent_signals"][signal.receptor_type.value] = {
                "intensity": signal.intensity,
                "confidence": signal.confidence,
                "activated": signal.intensity > 0.1
            }
        
        return summary


# Global fusion engine instance
_fusion_engine: Optional[BioneuroOlfactoryFusionEngine] = None

def get_fusion_engine() -> BioneuroOlfactoryFusionEngine:
    """Get or create global fusion engine instance."""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = BioneuroOlfactoryFusionEngine()
    return _fusion_engine


async def analyze_document_scent(document_text: str, document_id: str, 
                               metadata: Optional[Dict[str, Any]] = None) -> DocumentScentProfile:
    """Convenience function to analyze document using global fusion engine."""
    engine = get_fusion_engine()
    return await engine.analyze_document(document_text, document_id, metadata)


def compute_scent_similarity(doc1_id: str, doc2_id: str) -> Optional[float]:
    """Compute bioneural scent similarity between two documents."""
    engine = get_fusion_engine()
    
    profile1 = engine.document_profiles.get(doc1_id)
    profile2 = engine.document_profiles.get(doc2_id)
    
    if not profile1 or not profile2:
        return None
    
    distance = profile1.compute_scent_distance(profile2)
    return max(0.0, 1.0 - distance)