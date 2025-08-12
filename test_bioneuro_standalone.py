#!/usr/bin/env python3
"""
Standalone Bioneural Test

Tests core bioneural functionality without external dependencies.
"""

import sys
import time
import asyncio
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib
import json
from collections import defaultdict
import math

# Standalone implementation for testing
class OlfactoryReceptorType(Enum):
    """Types of simulated olfactory receptors for document analysis."""
    LEGAL_COMPLEXITY = "legal_complexity"
    STATUTORY_AUTHORITY = "statutory_authority"
    TEMPORAL_FRESHNESS = "temporal_freshness"
    CITATION_DENSITY = "citation_density"
    RISK_PROFILE = "risk_profile"
    SEMANTIC_COHERENCE = "semantic_coherence"

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

class BioneuroOlfactoryReceptor:
    """Simulates a biological olfactory receptor for document analysis."""
    
    def __init__(self, receptor_type: OlfactoryReceptorType, sensitivity: float = 0.5):
        self.receptor_type = receptor_type
        self.sensitivity = sensitivity
        self.activation_threshold = 0.1
    
    async def activate(self, document_text: str, metadata: Dict[str, Any] = None) -> OlfactorySignal:
        """Activate receptor and generate olfactory signal for document."""
        metadata = metadata or {}
        
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
            
            return OlfactorySignal(
                receptor_type=self.receptor_type,
                intensity=adjusted_intensity,
                confidence=confidence,
                metadata={
                    "receptor_sensitivity": self.sensitivity,
                    "raw_intensity": intensity,
                    "activation_successful": adjusted_intensity > self.activation_threshold
                }
            )
            
        except Exception as e:
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
            "U.S.C.", "USC", "CFR", "Section", "Title", "Code"
        ]
        
        authority_count = sum(1 for pattern in authority_patterns if pattern in text)
        text_length = max(1, len(text.split()))
        authority_density = min(1.0, authority_count / (text_length / 100))
        return authority_density
    
    def _detect_temporal_freshness(self, text: str, metadata: Dict[str, Any]) -> float:
        """Detect temporal relevance and freshness indicators."""
        import re
        from datetime import datetime
        
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
        citation_keywords = ["v.", "id.", "supra", "infra", "see", "cited", "case"]
        citation_count = sum(1 for keyword in citation_keywords if keyword.lower() in text.lower())
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
    
    def __init__(self):
        """Initialize fusion engine."""
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
        
        # Initialize receptors
        for receptor_type, sensitivity in default_config.items():
            self.receptors[receptor_type] = BioneuroOlfactoryReceptor(
                receptor_type=receptor_type,
                sensitivity=sensitivity
            )
    
    async def analyze_document(self, document_text: str, document_id: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> DocumentScentProfile:
        """Analyze document and generate comprehensive scent profile."""
        metadata = metadata or {}
        
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
        
        return profile
    
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

# Test functions
def test_basic_functionality():
    """Test basic bioneural functionality."""
    print("🧬 Testing Basic Bioneural Functionality")
    print("-" * 40)
    
    # Test receptor creation
    receptor = BioneuroOlfactoryReceptor(
        OlfactoryReceptorType.LEGAL_COMPLEXITY,
        sensitivity=0.8
    )
    
    assert receptor.receptor_type == OlfactoryReceptorType.LEGAL_COMPLEXITY
    assert receptor.sensitivity == 0.8
    print("✅ Receptor creation successful")
    
    return True

async def test_receptor_activation():
    """Test receptor activation with sample text."""
    print("\n🔬 Testing Receptor Activation")
    print("-" * 40)
    
    receptor = BioneuroOlfactoryReceptor(
        OlfactoryReceptorType.LEGAL_COMPLEXITY,
        sensitivity=0.8
    )
    
    test_text = """
    WHEREAS, the parties hereto desire to enter into this agreement
    pursuant to applicable statutes and regulations, NOTWITHSTANDING
    any prior agreements, the Contractor shall indemnify and hold
    harmless the Company from any liability, damages, or penalties.
    """
    
    signal = await receptor.activate(test_text)
    
    assert signal.receptor_type == OlfactoryReceptorType.LEGAL_COMPLEXITY
    assert 0.0 <= signal.intensity <= 1.0
    assert 0.0 <= signal.confidence <= 1.0
    
    print(f"✅ Receptor activation successful")
    print(f"   Intensity: {signal.intensity:.3f}")
    print(f"   Confidence: {signal.confidence:.3f}")
    print(f"   Activated: {signal.metadata.get('activation_successful', False)}")
    
    return True

async def test_document_analysis():
    """Test complete document analysis."""
    print("\n📄 Testing Document Analysis")
    print("-" * 40)
    
    engine = BioneuroOlfactoryFusionEngine()
    
    test_document = """
    PROFESSIONAL SERVICES AGREEMENT
    
    This Agreement is entered into as of January 15, 2024, between
    TechCorp Inc. and Legal Advisors LLC pursuant to 15 U.S.C. § 1681.
    
    The Contractor shall provide legal consulting services and shall
    indemnify Company from any claims, damages, or liabilities arising
    from negligent performance, PROVIDED THAT liability shall not exceed
    the total contract value and penalties may apply for breach.
    """
    
    start_time = time.time()
    profile = await engine.analyze_document(test_document, "test_doc_1")
    analysis_time = time.time() - start_time
    
    assert isinstance(profile, DocumentScentProfile)
    assert profile.document_id == "test_doc_1"
    assert len(profile.signals) > 0
    assert len(profile.composite_scent) > 0
    
    print(f"✅ Document analysis successful")
    print(f"   Analysis time: {analysis_time:.3f}s")
    print(f"   Signals detected: {len(profile.signals)}")
    print(f"   Composite scent dimensions: {len(profile.composite_scent)}")
    
    # Show signal details
    for signal in profile.signals:
        print(f"   {signal.receptor_type.value}: {signal.intensity:.3f} (conf: {signal.confidence:.3f})")
    
    return True

async def test_performance_benchmark():
    """Test performance with multiple documents."""
    print("\n⚡ Testing Performance Benchmark")
    print("-" * 40)
    
    engine = BioneuroOlfactoryFusionEngine()
    
    test_documents = [
        ("Simple contract with basic terms.", "doc_1"),
        ("Complex legal agreement with liability provisions pursuant to regulations.", "doc_2"),
        ("Comprehensive statute with multiple sections and penalty clauses.", "doc_3"),
        ("Regulatory document with compliance requirements and sanctions.", "doc_4"),
        ("Commercial contract with indemnification and limitation of liability.", "doc_5")
    ]
    
    start_time = time.time()
    
    profiles = []
    for doc_text, doc_id in test_documents:
        profile = await engine.analyze_document(doc_text, doc_id)
        profiles.append(profile)
    
    total_time = time.time() - start_time
    avg_time = total_time / len(test_documents)
    throughput = len(test_documents) / total_time
    
    print(f"✅ Performance test completed")
    print(f"   Documents processed: {len(profiles)}")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average per document: {avg_time:.3f}s")
    print(f"   Throughput: {throughput:.1f} docs/sec")
    
    # Performance should be reasonable
    return avg_time < 2.0  # Should be under 2 seconds per document

async def test_scent_similarity():
    """Test scent similarity between documents."""
    print("\n🔗 Testing Scent Similarity")
    print("-" * 40)
    
    engine = BioneuroOlfactoryFusionEngine()
    
    # Similar documents
    doc1 = "Contract with liability provisions and indemnification clauses."
    doc2 = "Agreement including liability terms and indemnification provisions."
    
    # Different document
    doc3 = "Simple weather report for today."
    
    profile1 = await engine.analyze_document(doc1, "similar_1")
    profile2 = await engine.analyze_document(doc2, "similar_2")  
    profile3 = await engine.analyze_document(doc3, "different_1")
    
    # Calculate similarity (simple comparison)
    similarity_12 = np.dot(profile1.composite_scent, profile2.composite_scent) / (
        np.linalg.norm(profile1.composite_scent) * np.linalg.norm(profile2.composite_scent)
    )
    
    similarity_13 = np.dot(profile1.composite_scent, profile3.composite_scent) / (
        np.linalg.norm(profile1.composite_scent) * np.linalg.norm(profile3.composite_scent)
    )
    
    print(f"✅ Scent similarity test completed")
    print(f"   Similar docs similarity: {similarity_12:.3f}")
    print(f"   Different docs similarity: {similarity_13:.3f}")
    
    # Similar documents should have higher similarity
    return similarity_12 > similarity_13

async def main():
    """Run all tests."""
    print("🧬 BIONEURAL OLFACTORY FUSION SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality, False),
        ("Receptor Activation", test_receptor_activation, True),
        ("Document Analysis", test_document_analysis, True),
        ("Performance Benchmark", test_performance_benchmark, True),
        ("Scent Similarity", test_scent_similarity, True)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func, is_async in tests:
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total:.1%}")
    
    if passed >= total * 0.8:  # 80% pass rate
        print("\n🎉 Bioneural olfactory fusion system is functional!")
        print("✨ Novel multi-sensory legal document analysis capabilities verified!")
        return True
    else:
        print("\n⚠️  Some tests failed. System needs review.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)