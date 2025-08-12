"""
Comprehensive tests for Bioneural Olfactory Fusion module.

Tests cover:
- Individual olfactory receptor functionality
- Scent profile generation and comparison
- Fusion engine operations
- Error handling and edge cases
- Performance characteristics
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

from lexgraph_legal_rag.bioneuro_olfactory_fusion import (
    BioneuroOlfactoryFusionEngine,
    BioneuroOlfactoryReceptor,
    DocumentScentProfile,
    OlfactorySignal,
    OlfactoryReceptorType,
    get_fusion_engine,
    analyze_document_scent,
    compute_scent_similarity
)


class TestBioneuroOlfactoryReceptor:
    """Test individual olfactory receptor functionality."""
    
    def test_receptor_initialization(self):
        """Test receptor initialization with different parameters."""
        receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.LEGAL_COMPLEXITY,
            sensitivity=0.8
        )
        
        assert receptor.receptor_type == OlfactoryReceptorType.LEGAL_COMPLEXITY
        assert receptor.sensitivity == 0.8
        assert receptor.activation_threshold == 0.1
    
    @pytest.mark.asyncio
    async def test_legal_complexity_detection(self):
        """Test legal complexity receptor activation."""
        receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.LEGAL_COMPLEXITY,
            sensitivity=1.0
        )
        
        # Test with complex legal text
        complex_text = """
        WHEREAS, the parties hereto desire to enter into this agreement 
        pursuant to the provisions set forth herein, and NOTWITHSTANDING 
        any prior agreements, the Contractor shall perform services in 
        accordance with applicable regulations, PROVIDED THAT all terms 
        remain subject to the conditions specified below.
        """
        
        signal = await receptor.activate(complex_text)
        
        assert signal.receptor_type == OlfactoryReceptorType.LEGAL_COMPLEXITY
        assert signal.intensity > 0.2  # Should detect complexity
        assert signal.confidence > 0.0
        assert signal.metadata["activation_successful"] is True
    
    @pytest.mark.asyncio
    async def test_statutory_authority_detection(self):
        """Test statutory authority receptor activation."""
        receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.STATUTORY_AUTHORITY,
            sensitivity=1.0
        )
        
        # Test with statutory references
        statutory_text = """
        According to 15 U.S.C. § 1681 and Title 12 CFR 225.4,
        financial institutions must comply with Section 5 of
        the U.S. Code provisions.
        """
        
        signal = await receptor.activate(statutory_text)
        
        assert signal.intensity > 0.1  # Should detect authority references
        assert signal.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_temporal_freshness_detection(self):
        """Test temporal freshness receptor activation."""
        receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.TEMPORAL_FRESHNESS,
            sensitivity=1.0
        )
        
        # Test with recent dates
        recent_text = """
        This regulation becomes effective January 1, 2024.
        The updated provisions apply to all contracts signed in 2023.
        """
        
        signal = await receptor.activate(recent_text)
        
        assert signal.intensity > 0.0
        assert signal.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_citation_density_detection(self):
        """Test citation density receptor activation."""
        receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.CITATION_DENSITY,
            sensitivity=1.0
        )
        
        # Test with citations
        cited_text = """
        See Smith v. Jones (2020) [1], Johnson v. Williams (2021) [2],
        and Brown v. Davis, id. at 123. Supra note 1; infra Section 5.
        """
        
        signal = await receptor.activate(cited_text)
        
        assert signal.intensity > 0.1  # Should detect citations
        assert signal.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_risk_profile_detection(self):
        """Test risk profile receptor activation."""
        receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.RISK_PROFILE,
            sensitivity=1.0
        )
        
        # Test with risk keywords
        risky_text = """
        The contractor shall be liable for any damages, penalties,
        or sanctions resulting from breach of contract, negligence,
        fraud, or other unlawful conduct.
        """
        
        signal = await receptor.activate(risky_text)
        
        assert signal.intensity > 0.1  # Should detect risk indicators
        assert signal.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_semantic_coherence_detection(self):
        """Test semantic coherence receptor activation."""
        receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.SEMANTIC_COHERENCE,
            sensitivity=1.0
        )
        
        # Test with coherent text
        coherent_text = """
        Therefore, the parties agree to the following terms. However,
        certain provisions may be modified. Moreover, all amendments
        must be in writing. Consequently, oral modifications are invalid.
        """
        
        signal = await receptor.activate(coherent_text)
        
        assert signal.intensity > 0.0
        assert signal.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_receptor_error_handling(self):
        """Test receptor error handling with invalid input."""
        receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.LEGAL_COMPLEXITY,
            sensitivity=1.0
        )
        
        # Test with empty text
        signal = await receptor.activate("")
        assert signal.intensity == 0.0
        assert signal.confidence == 0.0
        
        # Test with None input (should handle gracefully)
        with patch.object(receptor, '_detect_legal_complexity', side_effect=Exception("Test error")):
            signal = await receptor.activate("test text")
            assert signal.intensity == 0.0
            assert signal.confidence == 0.0
            assert "error" in signal.metadata
    
    @pytest.mark.asyncio
    async def test_sensitivity_adjustment(self):
        """Test that receptor sensitivity affects output."""
        low_sensitivity_receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.LEGAL_COMPLEXITY,
            sensitivity=0.1
        )
        
        high_sensitivity_receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.LEGAL_COMPLEXITY,
            sensitivity=1.0
        )
        
        test_text = "WHEREAS the parties hereto agree pursuant to these terms."
        
        low_signal = await low_sensitivity_receptor.activate(test_text)
        high_signal = await high_sensitivity_receptor.activate(test_text)
        
        # High sensitivity should produce stronger signal
        assert high_signal.intensity > low_signal.intensity


class TestDocumentScentProfile:
    """Test DocumentScentProfile functionality."""
    
    def test_scent_profile_creation(self):
        """Test creation of document scent profile."""
        signals = [
            OlfactorySignal(OlfactoryReceptorType.LEGAL_COMPLEXITY, 0.8, 0.9),
            OlfactorySignal(OlfactoryReceptorType.STATUTORY_AUTHORITY, 0.6, 0.7)
        ]
        
        composite_scent = np.array([0.8, 0.9, 0.6, 0.7, 0.0, 0.0])
        
        profile = DocumentScentProfile(
            document_id="test_doc",
            signals=signals,
            composite_scent=composite_scent,
            similarity_hash="abc123"
        )
        
        assert profile.document_id == "test_doc"
        assert len(profile.signals) == 2
        assert len(profile.composite_scent) == 6
        assert profile.similarity_hash == "abc123"
    
    def test_get_signal_by_type(self):
        """Test retrieving signals by receptor type."""
        signals = [
            OlfactorySignal(OlfactoryReceptorType.LEGAL_COMPLEXITY, 0.8, 0.9),
            OlfactorySignal(OlfactoryReceptorType.RISK_PROFILE, 0.3, 0.4)
        ]
        
        profile = DocumentScentProfile(
            document_id="test_doc",
            signals=signals,
            composite_scent=np.zeros(6),
            similarity_hash="test"
        )
        
        complexity_signal = profile.get_signal_by_type(OlfactoryReceptorType.LEGAL_COMPLEXITY)
        assert complexity_signal is not None
        assert complexity_signal.intensity == 0.8
        
        missing_signal = profile.get_signal_by_type(OlfactoryReceptorType.CITATION_DENSITY)
        assert missing_signal is None
    
    def test_scent_distance_computation(self):
        """Test bioneural distance computation between scent profiles."""
        profile1 = DocumentScentProfile(
            document_id="doc1",
            signals=[],
            composite_scent=np.array([0.8, 0.9, 0.6, 0.7]),
            similarity_hash="hash1"
        )
        
        profile2 = DocumentScentProfile(
            document_id="doc2",
            signals=[],
            composite_scent=np.array([0.7, 0.8, 0.5, 0.6]),
            similarity_hash="hash2"
        )
        
        distance = profile1.compute_scent_distance(profile2)
        
        assert isinstance(distance, float)
        assert 0.0 <= distance <= 2.0  # Valid distance range
    
    def test_scent_distance_identical_profiles(self):
        """Test distance computation for identical profiles."""
        scent_vector = np.array([0.8, 0.9, 0.6, 0.7])
        
        profile1 = DocumentScentProfile(
            document_id="doc1",
            signals=[],
            composite_scent=scent_vector.copy(),
            similarity_hash="same"
        )
        
        profile2 = DocumentScentProfile(
            document_id="doc2",
            signals=[],
            composite_scent=scent_vector.copy(),
            similarity_hash="same"
        )
        
        distance = profile1.compute_scent_distance(profile2)
        
        assert distance == pytest.approx(0.0, abs=1e-6)
    
    def test_scent_distance_dimension_mismatch(self):
        """Test error handling for mismatched scent vector dimensions."""
        profile1 = DocumentScentProfile(
            document_id="doc1",
            signals=[],
            composite_scent=np.array([0.8, 0.9]),
            similarity_hash="hash1"
        )
        
        profile2 = DocumentScentProfile(
            document_id="doc2",
            signals=[],
            composite_scent=np.array([0.7, 0.8, 0.5]),
            similarity_hash="hash2"
        )
        
        with pytest.raises(ValueError, match="Incompatible scent profile dimensions"):
            profile1.compute_scent_distance(profile2)


class TestBioneuroOlfactoryFusionEngine:
    """Test the main fusion engine functionality."""
    
    def test_engine_initialization(self):
        """Test fusion engine initialization."""
        engine = BioneuroOlfactoryFusionEngine()
        
        assert len(engine.receptors) == 6  # All receptor types
        assert all(isinstance(receptor, BioneuroOlfactoryReceptor) 
                  for receptor in engine.receptors.values())
        assert isinstance(engine.document_profiles, dict)
    
    def test_custom_receptor_configuration(self):
        """Test engine initialization with custom receptor configuration."""
        custom_config = {
            OlfactoryReceptorType.LEGAL_COMPLEXITY: 0.9,
            OlfactoryReceptorType.RISK_PROFILE: 0.7
        }
        
        engine = BioneuroOlfactoryFusionEngine(receptor_config=custom_config)
        
        assert len(engine.receptors) == 2
        assert OlfactoryReceptorType.LEGAL_COMPLEXITY in engine.receptors
        assert OlfactoryReceptorType.RISK_PROFILE in engine.receptors
        assert engine.receptors[OlfactoryReceptorType.LEGAL_COMPLEXITY].sensitivity == 0.9
    
    @pytest.mark.asyncio
    async def test_document_analysis(self):
        """Test comprehensive document analysis."""
        engine = BioneuroOlfactoryFusionEngine()
        
        test_text = """
        WHEREAS, the parties agree to this contract pursuant to 15 U.S.C. § 1681,
        the Contractor shall be liable for any damages resulting from breach.
        This agreement is effective January 1, 2024.
        """
        
        profile = await engine.analyze_document(test_text, "test_doc_1")
        
        assert isinstance(profile, DocumentScentProfile)
        assert profile.document_id == "test_doc_1"
        assert len(profile.signals) > 0
        assert len(profile.composite_scent) == 12  # 6 receptors × 2 dimensions
        assert profile.similarity_hash is not None
        
        # Check that profile was cached
        assert "test_doc_1" in engine.document_profiles
    
    @pytest.mark.asyncio
    async def test_document_analysis_with_metadata(self):
        """Test document analysis with metadata."""
        engine = BioneuroOlfactoryFusionEngine()
        
        metadata = {
            "creation_date": "2024-01-01T00:00:00Z",
            "document_type": "contract"
        }
        
        profile = await engine.analyze_document("Test document", "test_doc_2", metadata)
        
        assert profile.document_id == "test_doc_2"
        # Metadata should be passed to receptors
    
    def test_composite_scent_creation(self):
        """Test creation of composite scent vectors."""
        engine = BioneuroOlfactoryFusionEngine()
        
        signals = [
            OlfactorySignal(OlfactoryReceptorType.LEGAL_COMPLEXITY, 0.8, 0.9),
            OlfactorySignal(OlfactoryReceptorType.RISK_PROFILE, 0.5, 0.6)
        ]
        
        composite_scent = engine._create_composite_scent(signals)
        
        assert len(composite_scent) == 12  # 6 receptor types × 2 dimensions
        assert composite_scent[0] == 0.8  # LEGAL_COMPLEXITY intensity
        assert composite_scent[1] == 0.9  # LEGAL_COMPLEXITY confidence
        assert composite_scent[10] == 0.5  # RISK_PROFILE intensity (index 5*2)
        assert composite_scent[11] == 0.6  # RISK_PROFILE confidence
    
    def test_similarity_hash_generation(self):
        """Test similarity hash generation."""
        engine = BioneuroOlfactoryFusionEngine()
        
        scent_vector = np.array([0.123, 0.456, 0.789])
        hash1 = engine._generate_similarity_hash(scent_vector)
        hash2 = engine._generate_similarity_hash(scent_vector)
        
        assert hash1 == hash2  # Same input should produce same hash
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length
        
        # Different vectors should produce different hashes
        different_vector = np.array([0.124, 0.456, 0.789])
        hash3 = engine._generate_similarity_hash(different_vector)
        assert hash1 != hash3
    
    @pytest.mark.asyncio
    async def test_find_similar_documents(self):
        """Test finding similar documents using scent profiles."""
        engine = BioneuroOlfactoryFusionEngine()
        
        # Create two similar documents
        doc1_text = "Contract with liability and damages clauses pursuant to statute."
        doc2_text = "Agreement including liability provisions and damage limitations under law."
        doc3_text = "Simple statement with no legal complexity."
        
        profile1 = await engine.analyze_document(doc1_text, "doc1")
        profile2 = await engine.analyze_document(doc2_text, "doc2")
        profile3 = await engine.analyze_document(doc3_text, "doc3")
        
        # Find documents similar to doc1
        similar_docs = await engine.find_similar_documents(profile1, similarity_threshold=0.1)
        
        assert isinstance(similar_docs, list)
        # Should find doc2 as similar (both have legal complexity)
        doc_ids = [doc_id for doc_id, similarity in similar_docs]
        assert "doc2" in doc_ids or "doc3" in doc_ids  # At least one should be found
    
    def test_scent_summary_generation(self):
        """Test human-readable scent summary generation."""
        engine = BioneuroOlfactoryFusionEngine()
        
        # Manually create a profile for testing
        signals = [
            OlfactorySignal(OlfactoryReceptorType.LEGAL_COMPLEXITY, 0.8, 0.9),
            OlfactorySignal(OlfactoryReceptorType.RISK_PROFILE, 0.3, 0.4)
        ]
        
        profile = DocumentScentProfile(
            document_id="summary_test",
            signals=signals,
            composite_scent=np.array([0.8, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4]),
            similarity_hash="test_hash"
        )
        
        engine.document_profiles["summary_test"] = profile
        
        summary = engine.get_scent_summary("summary_test")
        
        assert summary is not None
        assert summary["document_id"] == "summary_test"
        assert "scent_signals" in summary
        assert "overall_intensity" in summary
        assert "scent_complexity" in summary
        
        # Check signal details
        assert "legal_complexity" in summary["scent_signals"]
        assert summary["scent_signals"]["legal_complexity"]["intensity"] == 0.8
        assert summary["scent_signals"]["legal_complexity"]["activated"] is True
    
    def test_scent_summary_missing_document(self):
        """Test scent summary for non-existent document."""
        engine = BioneuroOlfactoryFusionEngine()
        
        summary = engine.get_scent_summary("non_existent_doc")
        
        assert summary is None


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_fusion_engine_singleton(self):
        """Test that get_fusion_engine returns singleton instance."""
        engine1 = get_fusion_engine()
        engine2 = get_fusion_engine()
        
        assert engine1 is engine2
        assert isinstance(engine1, BioneuroOlfactoryFusionEngine)
    
    @pytest.mark.asyncio
    async def test_analyze_document_scent_convenience(self):
        """Test convenience function for document scent analysis."""
        profile = await analyze_document_scent("Test document", "convenience_test")
        
        assert isinstance(profile, DocumentScentProfile)
        assert profile.document_id == "convenience_test"
    
    def test_compute_scent_similarity_convenience(self):
        """Test convenience function for scent similarity computation."""
        # First analyze some documents to populate profiles
        async def setup_docs():
            await analyze_document_scent("Document one with legal content", "sim_doc1")
            await analyze_document_scent("Document two with similar content", "sim_doc2")
        
        asyncio.run(setup_docs())
        
        similarity = compute_scent_similarity("sim_doc1", "sim_doc2")
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_compute_scent_similarity_missing_documents(self):
        """Test similarity computation with missing documents."""
        similarity = compute_scent_similarity("missing_doc1", "missing_doc2")
        
        assert similarity is None


class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_document_handling(self):
        """Test handling of empty documents."""
        engine = BioneuroOlfactoryFusionEngine()
        
        profile = await engine.analyze_document("", "empty_doc")
        
        assert profile.document_id == "empty_doc"
        assert len(profile.signals) >= 0  # May have some signals with 0 intensity
        assert len(profile.composite_scent) == 12
    
    @pytest.mark.asyncio
    async def test_very_long_document_handling(self):
        """Test handling of very long documents."""
        engine = BioneuroOlfactoryFusionEngine()
        
        # Create a very long document
        long_text = "This is a legal document. " * 10000  # Very long text
        
        profile = await engine.analyze_document(long_text, "long_doc")
        
        assert profile.document_id == "long_doc"
        assert len(profile.composite_scent) == 12
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self):
        """Test handling of documents with special characters."""
        engine = BioneuroOlfactoryFusionEngine()
        
        special_text = """
        Contract with special chars: @#$%^&*()
        Legal symbols: § ¶ © ® ™
        Unicode: 中文 العربية русский
        """
        
        profile = await engine.analyze_document(special_text, "special_doc")
        
        assert profile.document_id == "special_doc"
        # Should handle special characters without errors
    
    @pytest.mark.asyncio
    async def test_concurrent_document_analysis(self):
        """Test concurrent analysis of multiple documents."""
        engine = BioneuroOlfactoryFusionEngine()
        
        documents = {
            f"concurrent_doc_{i}": f"Legal document number {i} with various clauses and provisions."
            for i in range(10)
        }
        
        # Analyze all documents concurrently
        tasks = [
            engine.analyze_document(text, doc_id)
            for doc_id, text in documents.items()
        ]
        
        profiles = await asyncio.gather(*tasks)
        
        assert len(profiles) == 10
        assert all(isinstance(profile, DocumentScentProfile) for profile in profiles)
        assert len(set(profile.document_id for profile in profiles)) == 10  # All unique IDs
    
    @pytest.mark.asyncio
    async def test_receptor_failure_resilience(self):
        """Test system resilience when individual receptors fail."""
        engine = BioneuroOlfactoryFusionEngine()
        
        # Mock one receptor to always fail
        with patch.object(
            engine.receptors[OlfactoryReceptorType.LEGAL_COMPLEXITY],
            'activate',
            side_effect=Exception("Simulated receptor failure")
        ):
            profile = await engine.analyze_document("Test document", "resilience_test")
            
            # Should still produce a profile with other receptors
            assert profile.document_id == "resilience_test"
            assert len(profile.composite_scent) == 12
            # Should have fewer valid signals due to one failure
            assert len(profile.signals) < 6


class TestIntegrationScenarios:
    """Test integration scenarios and realistic use cases."""
    
    @pytest.mark.asyncio
    async def test_contract_analysis_scenario(self):
        """Test realistic contract analysis scenario."""
        engine = BioneuroOlfactoryFusionEngine()
        
        contract_text = """
        PROFESSIONAL SERVICES AGREEMENT
        
        This Agreement is entered into as of January 15, 2024, between
        TechCorp Inc. ("Company") and Legal Advisors LLC ("Contractor").
        
        WHEREAS, Company desires to retain Contractor to provide legal
        consulting services pursuant to the terms set forth herein;
        
        NOW THEREFORE, in consideration of the mutual covenants contained
        herein, the parties agree as follows:
        
        1. SCOPE OF SERVICES. Contractor shall provide comprehensive legal
        analysis and regulatory compliance consulting in accordance with
        applicable federal and state laws, including but not limited to
        15 U.S.C. § 78 and related SEC regulations.
        
        2. COMPENSATION. Company shall pay Contractor $300 per hour for
        services performed hereunder.
        
        3. INDEMNIFICATION. Contractor agrees to indemnify and hold harmless
        Company from any claims, damages, or liabilities arising from
        Contractor's negligent performance of services.
        
        4. LIMITATION OF LIABILITY. In no event shall either party's liability
        exceed the total amount paid under this Agreement.
        
        This Agreement shall be governed by the laws of Delaware.
        """
        
        profile = await engine.analyze_document(contract_text, "professional_services_contract")
        
        # Verify key aspects were detected
        complexity_signal = profile.get_signal_by_type(OlfactoryReceptorType.LEGAL_COMPLEXITY)
        authority_signal = profile.get_signal_by_type(OlfactoryReceptorType.STATUTORY_AUTHORITY)
        risk_signal = profile.get_signal_by_type(OlfactoryReceptorType.RISK_PROFILE)
        temporal_signal = profile.get_signal_by_type(OlfactoryReceptorType.TEMPORAL_FRESHNESS)
        
        # Contract should show complexity
        assert complexity_signal.intensity > 0.3
        
        # Should detect statutory references
        assert authority_signal.intensity > 0.1
        
        # Should detect indemnification risk
        assert risk_signal.intensity > 0.2
        
        # Should detect recent date (2024)
        assert temporal_signal.intensity > 0.5
    
    @pytest.mark.asyncio
    async def test_regulatory_document_analysis(self):
        """Test analysis of regulatory document."""
        engine = BioneuroOlfactoryFusionEngine()
        
        regulation_text = """
        FEDERAL REGISTER
        Vol. 89, No. 45
        Wednesday, March 6, 2024
        
        DEPARTMENT OF TREASURY
        Financial Crimes Enforcement Network
        
        31 CFR Part 1010
        RIN 1506-AB89
        
        Anti-Money Laundering Program Requirements for Investment Advisers
        
        AGENCY: Financial Crimes Enforcement Network (FinCEN), Treasury.
        
        ACTION: Final rule.
        
        SUMMARY: The Financial Crimes Enforcement Network (FinCEN) is issuing
        this final rule to establish anti-money laundering (AML) program
        requirements for registered investment advisers under the Bank Secrecy Act.
        
        DATES: This rule is effective May 1, 2024.
        
        The rule requires investment advisers to:
        (1) Establish and maintain an AML program;
        (2) Report suspicious activities;
        (3) Maintain appropriate records;
        (4) Verify customer identity.
        
        Violations may result in civil penalties up to $250,000 per violation
        and criminal sanctions including imprisonment.
        """
        
        profile = await engine.analyze_document(regulation_text, "aml_final_rule")
        
        # Verify detection of regulatory characteristics
        authority_signal = profile.get_signal_by_type(OlfactoryReceptorType.STATUTORY_AUTHORITY)
        temporal_signal = profile.get_signal_by_type(OlfactoryReceptorType.TEMPORAL_FRESHNESS)
        risk_signal = profile.get_signal_by_type(OlfactoryReceptorType.RISK_PROFILE)
        
        # Should detect strong regulatory authority
        assert authority_signal.intensity > 0.2
        
        # Should detect recent effective date
        assert temporal_signal.intensity > 0.7
        
        # Should detect penalty risks
        assert risk_signal.intensity > 0.3
    
    @pytest.mark.asyncio
    async def test_document_similarity_classification(self):
        """Test document similarity for classification purposes."""
        engine = BioneuroOlfactoryFusionEngine()
        
        # Analyze different types of legal documents
        documents = {
            "contract1": """Service Agreement between parties with liability clauses
                          and indemnification provisions effective 2024.""",
            "contract2": """Professional Services Contract including compensation terms
                          and limitation of liability provisions dated 2024.""",
            "statute1": """Title 15 U.S.C. Section 1681 establishes consumer reporting
                         requirements and penalties for violations.""",
            "statute2": """15 USC 78 Securities Exchange Act provisions governing
                         trading activities and regulatory compliance.""",
            "case1": """In Smith v. Jones (1995), the court held that contractual
                      liability limitations must be conspicuous and clear."""
        }
        
        profiles = {}
        for doc_id, text in documents.items():
            profiles[doc_id] = await engine.analyze_document(text, doc_id)
        
        # Test similarity between contracts
        contract_similarity = profiles["contract1"].compute_scent_distance(profiles["contract2"])
        
        # Test similarity between statutes
        statute_similarity = profiles["statute1"].compute_scent_distance(profiles["statute2"])
        
        # Test dissimilarity between contract and case
        cross_type_similarity = profiles["contract1"].compute_scent_distance(profiles["case1"])
        
        # Contracts should be more similar to each other than to cases
        # Note: Lower distance means higher similarity
        assert contract_similarity < cross_type_similarity
        assert statute_similarity < cross_type_similarity