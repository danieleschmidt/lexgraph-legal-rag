"""
Enhanced Bioneural Olfactory Fusion - Generation 1: Make It Work (Simple)

This module implements immediate performance improvements to the bioneural 
olfactory fusion system for legal document analysis.

Generation 1 Enhancements:
- Optimized receptor activation pipeline
- Enhanced error handling and recovery
- Improved caching for repeated document analysis
- Streamlined scent profile generation
- Basic performance monitoring
"""

from __future__ import annotations

import asyncio
import logging
import numpy as np
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict
import functools

# Import base classes from existing system
from src.lexgraph_legal_rag.bioneuro_olfactory_fusion import (
    OlfactoryReceptorType, 
    OlfactorySignal, 
    DocumentScentProfile,
    BioneuroOlfactoryReceptor
)

logger = logging.getLogger(__name__)


class EnhancedBioneuroCache:
    """Simple but effective caching system for bioneural analysis."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_count = defaultdict(int)
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if still valid."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.access_count[key] += 1
                return value
            else:
                del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with automatic eviction."""
        # Evict least recently used items if at capacity
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = (value, time.time())
        self.access_count[key] = 1


class EnhancedBioneuroReceptor(BioneuroOlfactoryReceptor):
    """Enhanced version with Generation 1 improvements."""
    
    def __init__(self, receptor_type: OlfactoryReceptorType, sensitivity: float = 0.5):
        super().__init__(receptor_type, sensitivity)
        self.cache = EnhancedBioneuroCache(max_size=500)
        self.performance_metrics = {
            "activations": 0,
            "cache_hits": 0,
            "average_time": 0.0,
            "errors": 0
        }
    
    @functools.lru_cache(maxsize=128)
    def _get_document_hash(self, text: str) -> str:
        """Generate stable hash for document caching."""
        return hashlib.md5(text.encode('utf-8'), usedforsecurity=False).hexdigest()
    
    async def activate(self, document_text: str, metadata: Dict[str, Any] = None) -> OlfactorySignal:
        """Enhanced activation with caching and performance tracking."""
        metadata = metadata or {}
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{self.receptor_type.value}:{self._get_document_hash(document_text)}"
        cached_signal = self.cache.get(cache_key)
        
        if cached_signal:
            self.performance_metrics["cache_hits"] += 1
            self.performance_metrics["activations"] += 1
            logger.debug(f"Cache hit for {self.receptor_type.value}")
            return cached_signal
        
        try:
            # Enhanced receptor activation with optimized patterns
            intensity = await self._enhanced_detect(document_text, metadata)
            
            # Apply sensitivity and threshold
            adjusted_intensity = intensity * self.sensitivity
            confidence = min(1.0, adjusted_intensity / self.activation_threshold) if adjusted_intensity > 0 else 0.0
            
            # Create enhanced signal
            processing_time = time.time() - start_time
            signal = OlfactorySignal(
                receptor_type=self.receptor_type,
                intensity=adjusted_intensity,
                confidence=confidence,
                metadata={
                    "receptor_sensitivity": self.sensitivity,
                    "raw_intensity": intensity,
                    "activation_successful": adjusted_intensity > self.activation_threshold,
                    "processing_time": processing_time,
                    "cache_miss": True,
                    "generation": "G1_enhanced"
                }
            )
            
            # Cache the result
            self.cache.set(cache_key, signal)
            
            # Update metrics
            self.performance_metrics["activations"] += 1
            self.performance_metrics["average_time"] = (
                (self.performance_metrics["average_time"] * (self.performance_metrics["activations"] - 1) + processing_time) 
                / self.performance_metrics["activations"]
            )
            
            logger.debug(f"Enhanced receptor {self.receptor_type.value} activated: intensity={intensity:.3f}")
            return signal
            
        except Exception as e:
            self.performance_metrics["errors"] += 1
            logger.error(f"Enhanced receptor activation failed: {e}")
            
            # Graceful fallback
            return OlfactorySignal(
                receptor_type=self.receptor_type,
                intensity=0.1,  # Minimal fallback
                confidence=0.2,
                metadata={"error": str(e), "fallback_mode": True, "generation": "G1_enhanced"}
            )
    
    async def _enhanced_detect(self, text: str, metadata: Dict[str, Any]) -> float:
        """Enhanced detection with optimized pattern matching."""
        if self.receptor_type == OlfactoryReceptorType.LEGAL_COMPLEXITY:
            return self._enhanced_detect_legal_complexity(text)
        elif self.receptor_type == OlfactoryReceptorType.STATUTORY_AUTHORITY:
            return self._enhanced_detect_statutory_authority(text)
        elif self.receptor_type == OlfactoryReceptorType.TEMPORAL_FRESHNESS:
            return self._enhanced_detect_temporal_freshness(text, metadata)
        elif self.receptor_type == OlfactoryReceptorType.CITATION_DENSITY:
            return self._enhanced_detect_citation_density(text)
        elif self.receptor_type == OlfactoryReceptorType.RISK_PROFILE:
            return self._enhanced_detect_risk_profile(text)
        elif self.receptor_type == OlfactoryReceptorType.SEMANTIC_COHERENCE:
            return self._enhanced_detect_semantic_coherence(text)
        else:
            return 0.0
    
    def _enhanced_detect_legal_complexity(self, text: str) -> float:
        """Enhanced legal complexity detection with weighted markers."""
        # Weighted complexity markers
        high_complexity = ["notwithstanding", "heretofore", "whereas", "provided that"]
        medium_complexity = ["pursuant to", "subject to", "in accordance with"]
        low_complexity = ["shall", "will", "must", "may"]
        
        text_lower = text.lower()
        high_score = sum(2.0 for marker in high_complexity if marker in text_lower)
        medium_score = sum(1.5 for marker in medium_complexity if marker in text_lower)
        low_score = sum(1.0 for marker in low_complexity if marker in text_lower)
        
        # Sentence complexity
        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        avg_sentence_length = len(text.split()) / sentences
        length_complexity = min(2.0, avg_sentence_length / 25.0)
        
        total_score = (high_score + medium_score + low_score + length_complexity) / 10.0
        return min(1.0, total_score)
    
    def _enhanced_detect_statutory_authority(self, text: str) -> float:
        """Enhanced statutory authority detection."""
        import re
        
        # Optimized regex patterns with scores
        patterns = {
            r'\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+': 3.0,  # USC references - high weight
            r'\bSection\s+\d+': 2.0,                    # Section references - medium weight
            r'\bTitle\s+\d+': 2.0,                      # Title references - medium weight
            r'\bCFR\b': 2.5,                            # CFR - high weight
            r'\b\d+\s+F\.?\s*\d*d?\s+\d+': 2.0,        # Federal reporter citations
        }
        
        total_score = 0.0
        for pattern, weight in patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            total_score += matches * weight
        
        # Normalize by text length
        words = max(1, len(text.split()))
        normalized_score = min(1.0, total_score / (words / 50))
        return normalized_score
    
    def _enhanced_detect_temporal_freshness(self, text: str, metadata: Dict[str, Any]) -> float:
        """Enhanced temporal freshness detection."""
        import re
        from datetime import datetime
        
        current_year = datetime.now().year
        
        # Extract years with better patterns
        year_patterns = [
            r'\b(20[0-2]\d)\b',  # 2000-2029
            r'\b(19[89]\d)\b',   # 1980-1999
        ]
        
        years = []
        for pattern in year_patterns:
            years.extend([int(year) for year in re.findall(pattern, text)])
        
        if not years:
            return 0.3  # Default moderate freshness
        
        # Calculate recency score
        max_year = max(years)
        years_ago = current_year - max_year
        
        if years_ago <= 1:
            return 1.0  # Very fresh
        elif years_ago <= 3:
            return 0.8  # Recent
        elif years_ago <= 5:
            return 0.6  # Somewhat recent
        elif years_ago <= 10:
            return 0.4  # Older
        else:
            return 0.2  # Old
    
    def _enhanced_detect_citation_density(self, text: str) -> float:
        """Enhanced citation density detection."""
        import re
        
        # Citation patterns with weights
        citation_patterns = {
            r'\bv\.\s+[A-Z][a-zA-Z\s]+': 2.0,          # Case citations
            r'\b\d+\s+F\.\s*\d*d?\s+\d+': 2.5,         # Federal reporters
            r'\bId\.\s+at\s+\d+': 1.5,                  # Id. citations
            r'\bSee\s+[A-Z]': 1.0,                      # See citations
            r'\bCf\.\s+[A-Z]': 1.0,                     # Compare citations
        }
        
        total_citations = 0.0
        for pattern, weight in citation_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            total_citations += matches * weight
        
        # Normalize by paragraphs
        paragraphs = max(1, text.count('\n\n') + 1)
        citation_density = min(1.0, total_citations / paragraphs)
        return citation_density
    
    def _enhanced_detect_risk_profile(self, text: str) -> float:
        """Enhanced risk profile detection."""
        # Risk indicators with weights
        high_risk = ["liability", "penalty", "violation", "breach", "damages"]
        medium_risk = ["obligation", "requirement", "compliance", "indemnify"]
        low_risk = ["should", "recommend", "suggest", "consider"]
        
        text_lower = text.lower()
        high_score = sum(3.0 for term in high_risk if term in text_lower)
        medium_score = sum(2.0 for term in medium_risk if term in text_lower)
        low_score = sum(1.0 for term in low_risk if term in text_lower)
        
        total_score = (high_score + medium_score + low_score) / 15.0
        return min(1.0, total_score)
    
    def _enhanced_detect_semantic_coherence(self, text: str) -> float:
        """Enhanced semantic coherence detection."""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5  # Single sentence - moderate coherence
        
        # Check for transition words
        transitions = ["however", "therefore", "furthermore", "moreover", "additionally"]
        transition_count = sum(1 for sent in sentences for trans in transitions if trans in sent.lower())
        
        # Check for pronoun references
        pronouns = ["this", "that", "these", "those", "it", "they"]
        pronoun_count = sum(1 for sent in sentences for pron in pronouns if pron in sent.lower().split())
        
        # Calculate coherence score
        coherence_score = (transition_count + pronoun_count) / max(1, len(sentences))
        return min(1.0, coherence_score)


class EnhancedBioneuroFusionEngine:
    """Enhanced fusion engine with Generation 1 improvements."""
    
    def __init__(self, receptor_sensitivities: Optional[Dict[str, float]] = None):
        self.receptor_sensitivities = receptor_sensitivities or {
            "legal_complexity": 0.7,
            "statutory_authority": 0.8,
            "temporal_freshness": 0.6,
            "citation_density": 0.7,
            "risk_profile": 0.8,
            "semantic_coherence": 0.5
        }
        
        # Initialize enhanced receptors
        self.receptors = {
            receptor_type: EnhancedBioneuroReceptor(
                receptor_type, 
                self.receptor_sensitivities.get(receptor_type.value, 0.5)
            )
            for receptor_type in OlfactoryReceptorType
        }
        
        self.analysis_cache = EnhancedBioneuroCache(max_size=200)
        self.performance_stats = {
            "documents_analyzed": 0,
            "total_analysis_time": 0.0,
            "cache_hit_rate": 0.0
        }
    
    async def analyze_document_scent(self, document_text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> DocumentScentProfile:
        """Enhanced document scent analysis with performance optimization."""
        start_time = time.time()
        metadata = metadata or {}
        
        # Check analysis cache
        cache_key = f"analysis:{hashlib.md5(document_text.encode('utf-8'), usedforsecurity=False).hexdigest()}"
        cached_profile = self.analysis_cache.get(cache_key)
        
        if cached_profile:
            self.performance_stats["cache_hit_rate"] = (
                self.performance_stats["cache_hit_rate"] * self.performance_stats["documents_analyzed"] + 1.0
            ) / (self.performance_stats["documents_analyzed"] + 1)
            self.performance_stats["documents_analyzed"] += 1
            logger.debug(f"Cache hit for document analysis: {document_id}")
            return cached_profile
        
        try:
            # Parallel receptor activation for better performance
            activation_tasks = [
                receptor.activate(document_text, metadata)
                for receptor in self.receptors.values()
            ]
            
            signals = await asyncio.gather(*activation_tasks, return_exceptions=True)
            
            # Filter out exceptions and create valid signals list
            valid_signals = []
            for signal in signals:
                if isinstance(signal, Exception):
                    logger.warning(f"Receptor activation failed: {signal}")
                    # Create fallback signal
                    valid_signals.append(OlfactorySignal(
                        receptor_type=OlfactoryReceptorType.LEGAL_COMPLEXITY,  # Default fallback
                        intensity=0.1,
                        confidence=0.1,
                        metadata={"fallback": True, "error": str(signal)}
                    ))
                else:
                    valid_signals.append(signal)
            
            # Generate composite scent vector
            composite_scent = self._generate_enhanced_composite_scent(valid_signals)
            
            # Create similarity hash
            similarity_hash = hashlib.md5(
                json.dumps([s.intensity for s in valid_signals], sort_keys=True).encode('utf-8'),
                usedforsecurity=False
            ).hexdigest()[:16]
            
            # Create enhanced scent profile
            profile = DocumentScentProfile(
                document_id=document_id,
                signals=valid_signals,
                composite_scent=composite_scent,
                similarity_hash=similarity_hash
            )
            
            # Cache the result
            self.analysis_cache.set(cache_key, profile)
            
            # Update performance statistics
            analysis_time = time.time() - start_time
            self.performance_stats["documents_analyzed"] += 1
            self.performance_stats["total_analysis_time"] += analysis_time
            
            logger.info(f"Enhanced document analysis completed for {document_id} in {analysis_time:.3f}s")
            return profile
            
        except Exception as e:
            logger.error(f"Enhanced document analysis failed for {document_id}: {e}")
            
            # Return minimal fallback profile
            fallback_signals = [
                OlfactorySignal(
                    receptor_type=receptor_type,
                    intensity=0.1,
                    confidence=0.1,
                    metadata={"fallback": True, "error": str(e)}
                )
                for receptor_type in OlfactoryReceptorType
            ]
            
            return DocumentScentProfile(
                document_id=document_id,
                signals=fallback_signals,
                composite_scent=np.array([0.1] * (len(OlfactoryReceptorType) * 2)),
                similarity_hash="fallback"
            )
    
    def _generate_enhanced_composite_scent(self, signals: List[OlfactorySignal]) -> np.ndarray:
        """Generate enhanced composite scent vector."""
        # Create enhanced feature vector with intensity and confidence
        features = []
        for signal in signals:
            features.append(signal.intensity)
            features.append(signal.confidence)
        
        # Add derived features for better discrimination
        if len(signals) > 1:
            intensities = [s.intensity for s in signals]
            features.append(np.mean(intensities))  # Mean intensity
            features.append(np.std(intensities))   # Intensity variance
            features.append(np.max(intensities))   # Peak intensity
        
        return np.array(features, dtype=np.float32)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        receptor_metrics = {}
        for receptor_type, receptor in self.receptors.items():
            receptor_metrics[receptor_type.value] = receptor.performance_metrics.copy()
        
        return {
            "engine_stats": self.performance_stats.copy(),
            "receptor_stats": receptor_metrics,
            "cache_stats": {
                "engine_cache_size": len(self.analysis_cache.cache),
                "total_cache_accesses": sum(self.analysis_cache.access_count.values())
            }
        }


# Enhanced convenience functions
async def analyze_document_scent_enhanced(document_text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> DocumentScentProfile:
    """Enhanced document scent analysis with Generation 1 improvements."""
    engine = EnhancedBioneuroFusionEngine()
    return await engine.analyze_document_scent(document_text, document_id, metadata)


async def batch_analyze_documents_enhanced(documents: List[Tuple[str, str]], metadata: Optional[Dict[str, Any]] = None) -> List[DocumentScentProfile]:
    """Enhanced batch document analysis with parallel processing."""
    engine = EnhancedBioneuroFusionEngine()
    
    tasks = [
        engine.analyze_document_scent(doc_text, doc_id, metadata)
        for doc_text, doc_id in documents
    ]
    
    profiles = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_profiles = []
    for i, profile in enumerate(profiles):
        if isinstance(profile, Exception):
            logger.error(f"Document analysis failed for {documents[i][1]}: {profile}")
            # Create fallback profile
            fallback_profile = DocumentScentProfile(
                document_id=documents[i][1],
                signals=[],
                composite_scent=np.array([0.1] * 12),
                similarity_hash="error"
            )
            valid_profiles.append(fallback_profile)
        else:
            valid_profiles.append(profile)
    
    return valid_profiles


if __name__ == "__main__":
    # Example usage
    async def demo_enhanced_bioneural():
        print("🧬 ENHANCED BIONEURAL OLFACTORY FUSION - GENERATION 1")
        print("=" * 60)
        
        sample_doc = """
        WHEREAS, the parties hereto agree pursuant to 15 U.S.C. § 1681,
        the Contractor shall indemnify Company from any liability, damages,
        or penalties arising from breach of this agreement.
        """
        
        print("📄 Analyzing sample legal document...")
        start = time.time()
        
        profile = await analyze_document_scent_enhanced(sample_doc, "contract_001")
        
        analysis_time = time.time() - start
        print(f"✅ Analysis completed in {analysis_time:.3f}s")
        print(f"🔬 Signals detected: {len(profile.signals)}")
        print(f"📊 Composite scent dimensions: {len(profile.composite_scent)}")
        
        for signal in profile.signals:
            if signal.intensity > 0.1:
                print(f"   {signal.receptor_type.value}: {signal.intensity:.3f} (conf: {signal.confidence:.3f})")
        
        print("\n🚀 Generation 1 enhancements active:")
        print("   ✓ Enhanced caching system")
        print("   ✓ Optimized pattern matching")
        print("   ✓ Parallel receptor activation")
        print("   ✓ Graceful error handling")
        print("   ✓ Performance monitoring")
    
    # Run demo
    asyncio.run(demo_enhanced_bioneural())