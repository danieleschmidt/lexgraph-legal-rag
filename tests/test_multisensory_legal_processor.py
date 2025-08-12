"""
Comprehensive tests for Multi-Sensory Legal Processor module.

Tests cover:
- Individual sensory channel processors
- Multi-sensory integration and fusion
- Performance characteristics
- Error handling and resilience
- Real-world document analysis scenarios
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any

from lexgraph_legal_rag.multisensory_legal_processor import (
    MultiSensoryLegalProcessor,
    TextualSensoryProcessor,
    VisualSensoryProcessor,
    TemporalSensoryProcessor,
    SensorySignal,
    SensoryChannel,
    MultiSensoryAnalysis,
    get_multisensory_processor,
    analyze_document_multisensory
)


class TestTextualSensoryProcessor:
    """Test textual sensory channel processor."""
    
    def test_processor_initialization(self):
        """Test textual processor initialization."""
        processor = TextualSensoryProcessor()
        assert processor.logger is not None
    
    @pytest.mark.asyncio
    async def test_basic_textual_processing(self):
        """Test basic textual feature extraction."""
        processor = TextualSensoryProcessor()
        
        test_text = """
        This is a contract between two parties. The agreement includes liability
        provisions and indemnification clauses. Both parties agree to comply
        with all applicable regulations and statutes.
        """
        
        signal = await processor.process(test_text, {})
        
        assert isinstance(signal, SensorySignal)
        assert signal.channel == SensoryChannel.TEXTUAL
        assert signal.strength > 0.0
        assert signal.confidence > 0.0
        
        # Check features
        features = signal.features
        assert "word_count" in features
        assert "sentence_count" in features
        assert "avg_sentence_length" in features
        assert "lexical_diversity" in features
        assert "legal_term_density" in features
        
        assert features["word_count"] > 20
        assert features["sentence_count"] >= 2
        assert features["legal_term_density"] > 0.0  # Should detect legal terms
    
    @pytest.mark.asyncio
    async def test_legal_term_density_calculation(self):
        """Test legal terminology density calculation."""
        processor = TextualSensoryProcessor()
        
        legal_text = """
        The contract establishes liability for damages. The plaintiff seeks
        compensation from the defendant. The court's jurisdiction extends
        to all breach of contract claims under applicable statute.
        """
        
        signal = await processor.process(legal_text, {})
        
        # Should detect high legal term density
        assert signal.features["legal_term_density"] > 0.1
        assert signal.strength > 0.3
    
    @pytest.mark.asyncio
    async def test_lexical_diversity_calculation(self):
        """Test lexical diversity calculation."""
        processor = TextualSensoryProcessor()
        
        # High diversity text
        diverse_text = "The comprehensive agreement establishes various complex provisions."
        
        # Low diversity text (repeated words)
        repetitive_text = "The agreement agreement establishes establishes provisions provisions."
        
        diverse_signal = await processor.process(diverse_text, {})
        repetitive_signal = await processor.process(repetitive_text, {})
        
        # Diverse text should have higher lexical diversity
        assert (diverse_signal.features["lexical_diversity"] > 
                repetitive_signal.features["lexical_diversity"])
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Test handling of empty text."""
        processor = TextualSensoryProcessor()
        
        signal = await processor.process("", {})
        
        assert signal.channel == SensoryChannel.TEXTUAL
        assert signal.strength == 0.0
        assert signal.confidence == 0.0
        assert signal.features["word_count"] == 0
    
    @pytest.mark.asyncio
    async def test_textual_processing_error_handling(self):
        """Test error handling in textual processing."""
        processor = TextualSensoryProcessor()
        
        # Mock a processing error
        with patch.object(processor, '_calculate_legal_term_density', side_effect=Exception("Test error")):
            signal = await processor.process("Test text", {})
            
            assert signal.channel == SensoryChannel.TEXTUAL
            assert signal.strength == 0.0
            assert signal.confidence == 0.0
            assert "error" in signal.features


class TestVisualSensoryProcessor:
    """Test visual/structural sensory channel processor."""
    
    def test_processor_initialization(self):
        """Test visual processor initialization."""
        processor = VisualSensoryProcessor()
        assert processor.logger is not None
    
    @pytest.mark.asyncio
    async def test_structure_complexity_analysis(self):
        """Test structural complexity analysis."""
        processor = VisualSensoryProcessor()
        
        structured_text = """
        # Main Contract Agreement
        
        ## Section 1: Definitions
        1. "Agreement" means this contract
        2. "Party" means signatory entity
        
        ## Section 2: Terms
        ### Subsection 2.1: Payment Terms
        • Monthly payments due
        • Late fees applicable
        
        Chapter 3: Termination
        """
        
        signal = await processor.process(structured_text, {})
        
        assert signal.channel == SensoryChannel.VISUAL
        assert signal.strength > 0.2  # Should detect structure
        assert signal.features["structure_complexity"] > 0.0
    
    @pytest.mark.asyncio
    async def test_formatting_indicators_detection(self):
        """Test detection of formatting indicators."""
        processor = VisualSensoryProcessor()
        
        formatted_text = """
        **IMPORTANT NOTICE**
        
        This agreement contains the following _key provisions_:
        
        • Liability limitations
        • Indemnification clauses
        • TERMINATION PROCEDURES
        
        **ALL CAPS SECTION** regarding penalties.
        """
        
        signal = await processor.process(formatted_text, {})
        
        formatting = signal.features["formatting_indicators"]
        assert formatting["bold_indicators"] > 0
        assert formatting["italic_indicators"] > 0
        assert formatting["bullet_points"] > 0
        assert formatting["all_caps_words"] > 0
    
    @pytest.mark.asyncio
    async def test_list_structure_analysis(self):
        """Test analysis of list structures."""
        processor = VisualSensoryProcessor()
        
        list_text = """
        The agreement includes:
        
        1. Payment terms
        2. Liability provisions
        3. Termination clauses
        
        Additional items:
        • Quality standards
        • Delivery requirements
          - Timing specifications
          - Location details
        """
        
        signal = await processor.process(list_text, {})
        
        list_structure = signal.features["list_structure"]
        assert list_structure["numbered_lists"] > 0
        assert list_structure["bulleted_lists"] > 0
        assert list_structure["total_list_items"] > 4
    
    @pytest.mark.asyncio
    async def test_whitespace_pattern_analysis(self):
        """Test whitespace usage pattern analysis."""
        processor = VisualSensoryProcessor()
        
        well_spaced_text = """
        Section 1: Introduction
        
        This agreement establishes terms.
        
        
        Section 2: Details
        
        The following provisions apply.
        """
        
        signal = await processor.process(well_spaced_text, {})
        
        whitespace = signal.features["whitespace_patterns"]
        assert "whitespace_ratio" in whitespace
        assert "paragraph_separation" in whitespace
        assert whitespace["whitespace_ratio"] > 0.1
    
    @pytest.mark.asyncio
    async def test_visual_processing_error_handling(self):
        """Test error handling in visual processing."""
        processor = VisualSensoryProcessor()
        
        with patch.object(processor, '_analyze_structure_complexity', side_effect=Exception("Test error")):
            signal = await processor.process("Test text", {})
            
            assert signal.channel == SensoryChannel.VISUAL
            assert signal.strength == 0.0
            assert signal.confidence == 0.0
            assert "error" in signal.features


class TestTemporalSensoryProcessor:
    """Test temporal sensory channel processor."""
    
    def test_processor_initialization(self):
        """Test temporal processor initialization."""
        processor = TemporalSensoryProcessor()
        assert processor.logger is not None
    
    @pytest.mark.asyncio
    async def test_temporal_reference_extraction(self):
        """Test extraction of temporal references."""
        processor = TemporalSensoryProcessor()
        
        temporal_text = """
        This agreement is effective January 15, 2024. The contract was
        modified on 03/20/2023 and will expire in 2025. Monthly payments
        are due throughout the term. The parties must provide notice
        within 30 days before termination.
        """
        
        signal = await processor.process(temporal_text, {})
        
        assert signal.channel == SensoryChannel.TEMPORAL
        assert signal.strength > 0.0
        
        temporal_refs = signal.features["temporal_references"]
        assert len(temporal_refs) > 0
        
        # Should find various types of temporal references
        ref_types = [ref["type"] for ref in temporal_refs]
        assert "date" in ref_types or "year" in ref_types
    
    @pytest.mark.asyncio
    async def test_chronological_order_analysis(self):
        """Test chronological order analysis."""
        processor = TemporalSensoryProcessor()
        
        # Text with chronological progression
        chronological_text = """
        The company was founded in 1990. In 1995, it expanded operations.
        By 2000, revenues had doubled. In 2010, the merger was completed.
        Current operations as of 2024 include global expansion.
        """
        
        signal = await processor.process(chronological_text, {})
        
        # Should detect good chronological order
        assert signal.features["chronological_order"] > 0.5
    
    @pytest.mark.asyncio
    async def test_recency_indicators_detection(self):
        """Test detection of recency indicators."""
        processor = TemporalSensoryProcessor()
        
        recent_text = """
        This is the latest updated version of the contract, reflecting
        current regulations effective 2024. The modern provisions include
        recent compliance requirements and new standards.
        """
        
        signal = await processor.process(recent_text, {})
        
        recency = signal.features["recency_indicators"]
        assert recency["recency_keyword_count"] > 0
        assert recency["recency_score"] > 0.0
    
    @pytest.mark.asyncio
    async def test_temporal_context_from_metadata(self):
        """Test temporal context analysis from metadata."""
        processor = TemporalSensoryProcessor()
        
        metadata = {
            "creation_date": "2024-01-15T10:30:00Z",
            "modification_date": "2024-01-20T15:45:00Z"
        }
        
        signal = await processor.process("Test document", metadata)
        
        temporal_context = signal.features["temporal_context"]
        assert temporal_context["creation_date"] is not None
        assert temporal_context["document_age_days"] is not None
        assert isinstance(temporal_context["is_recent"], bool)
    
    @pytest.mark.asyncio
    async def test_temporal_density_calculation(self):
        """Test temporal density calculation."""
        processor = TemporalSensoryProcessor()
        
        # High temporal density text
        dense_text = """
        January 2024 contract effective February 1, 2024. March deadline
        for compliance. April 15, 2024 review date. May 2024 renewal.
        """
        
        # Low temporal density text
        sparse_text = """
        This is a general contract with standard terms and conditions
        that apply to business relationships and operational procedures.
        """
        
        dense_signal = await processor.process(dense_text, {})
        sparse_signal = await processor.process(sparse_text, {})
        
        # Dense text should have higher temporal density
        assert (dense_signal.features["temporal_density"] > 
                sparse_signal.features["temporal_density"])
    
    @pytest.mark.asyncio
    async def test_temporal_processing_error_handling(self):
        """Test error handling in temporal processing."""
        processor = TemporalSensoryProcessor()
        
        with patch.object(processor, '_extract_temporal_references', side_effect=Exception("Test error")):
            signal = await processor.process("Test text", {})
            
            assert signal.channel == SensoryChannel.TEMPORAL
            assert signal.strength == 0.0
            assert signal.confidence == 0.0
            assert "error" in signal.features


class TestMultiSensoryLegalProcessor:
    """Test the main multi-sensory processor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        assert processor.enable_olfactory is True
        assert processor.textual_processor is not None
        assert processor.visual_processor is not None
        assert processor.temporal_processor is not None
        assert processor.olfactory_engine is not None
    
    def test_processor_initialization_without_olfactory(self):
        """Test processor initialization without olfactory."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=False)
        
        assert processor.enable_olfactory is False
        assert not hasattr(processor, 'olfactory_engine')
    
    @pytest.mark.asyncio
    async def test_comprehensive_document_processing(self):
        """Test comprehensive multi-sensory document processing."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        test_document = """
        # PROFESSIONAL SERVICES AGREEMENT
        
        **Effective Date:** January 15, 2024
        
        This Agreement establishes terms between the parties pursuant to
        applicable regulations including 15 U.S.C. § 1681.
        
        ## Section 1: Scope of Services
        
        The Contractor shall provide:
        1. Legal analysis and research
        2. Regulatory compliance consulting
        3. Risk assessment services
        
        ## Section 2: Liability and Indemnification
        
        Contractor agrees to indemnify Company against any damages,
        penalties, or sanctions resulting from negligent performance.
        
        **IMPORTANT:** This agreement contains liability limitations.
        """
        
        metadata = {
            "source": "legal_department",
            "document_type": "contract",
            "creation_date": "2024-01-15T10:00:00Z"
        }
        
        analysis = await processor.process_document(test_document, "comprehensive_test", metadata)
        
        assert isinstance(analysis, MultiSensoryAnalysis)
        assert analysis.document_id == "comprehensive_test"
        assert len(analysis.sensory_signals) > 0
        assert len(analysis.fusion_vector) > 0
        assert analysis.primary_sensory_channel in SensoryChannel
        assert 0.0 <= analysis.analysis_confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_sensory_channel_detection(self):
        """Test that all sensory channels are detected."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        # Document designed to trigger all sensory channels
        multi_sensory_document = """
        # CONTRACT AGREEMENT (Visual: structure)
        
        **Effective 2024** (Temporal: recent date)
        
        WHEREAS, the parties agree pursuant to 15 U.S.C. § 1681 (Textual: legal terms)
        with liability and indemnification provisions (Olfactory: legal complexity)
        
        ## Terms and Conditions
        
        1. Payment schedule: monthly
        2. Performance standards
        • Quality metrics
        • Delivery timelines
        
        Risk factors include penalties and sanctions.
        """
        
        analysis = await processor.process_document(multi_sensory_document, "multi_sensory_test")
        
        # Check that multiple channels detected signals
        channel_strengths = analysis.get_channel_strengths()
        
        assert SensoryChannel.TEXTUAL in channel_strengths
        assert SensoryChannel.VISUAL in channel_strengths
        assert SensoryChannel.TEMPORAL in channel_strengths
        assert SensoryChannel.OLFACTORY in channel_strengths
        
        # At least some channels should have non-zero strength
        non_zero_channels = sum(1 for strength in channel_strengths.values() if strength > 0.0)
        assert non_zero_channels >= 2
    
    @pytest.mark.asyncio
    async def test_primary_channel_determination(self):
        """Test determination of primary sensory channel."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        # Document with strong visual structure
        visual_heavy_document = """
        # TITLE
        ## Section 1
        ### Subsection 1.1
        #### Item 1.1.1
        
        1. First item
        2. Second item
        3. Third item
        
        • Bullet one
        • Bullet two
        
        **Bold text** and _italic text_
        
        ALL CAPS SECTION
        """
        
        analysis = await processor.process_document(visual_heavy_document, "visual_test")
        
        # Primary channel determination should be reasonable
        assert analysis.primary_sensory_channel in SensoryChannel
        
        # For this document, visual or textual should likely be primary
        assert analysis.primary_sensory_channel in [SensoryChannel.VISUAL, SensoryChannel.TEXTUAL]
    
    def test_fusion_vector_creation(self):
        """Test creation of fusion vectors from sensory signals."""
        processor = MultiSensoryLegalProcessor()
        
        signals = [
            SensorySignal(SensoryChannel.TEXTUAL, 0.8, 0.9),
            SensorySignal(SensoryChannel.VISUAL, 0.6, 0.7),
            SensorySignal(SensoryChannel.TEMPORAL, 0.4, 0.5)
        ]
        
        fusion_vector = processor._create_fusion_vector(signals)
        
        # Should create vector with 2 dimensions per channel
        expected_size = len(SensoryChannel) * 2
        assert len(fusion_vector) == expected_size
        
        # Check specific values
        assert fusion_vector[0] == 0.8  # TEXTUAL strength
        assert fusion_vector[1] == 0.9  # TEXTUAL confidence
        assert fusion_vector[2] == 0.6  # VISUAL strength (assuming order)
        assert fusion_vector[3] == 0.7  # VISUAL confidence
    
    def test_analysis_confidence_calculation(self):
        """Test calculation of overall analysis confidence."""
        processor = MultiSensoryLegalProcessor()
        
        signals = [
            SensorySignal(SensoryChannel.TEXTUAL, 0.8, 0.9),
            SensorySignal(SensoryChannel.VISUAL, 0.6, 0.7),
            SensorySignal(SensoryChannel.TEMPORAL, 0.2, 0.3)
        ]
        
        confidence = processor._calculate_analysis_confidence(signals)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident
    
    def test_analysis_confidence_with_empty_signals(self):
        """Test confidence calculation with no signals."""
        processor = MultiSensoryLegalProcessor()
        
        confidence = processor._calculate_analysis_confidence([])
        
        assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_processor_resilience_to_failures(self):
        """Test processor resilience when individual processors fail."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        # Mock textual processor to fail
        with patch.object(processor.textual_processor, 'process', 
                         side_effect=Exception("Textual processing failed")):
            analysis = await processor.process_document("Test document", "resilience_test")
            
            # Should still produce analysis with other processors
            assert isinstance(analysis, MultiSensoryAnalysis)
            assert analysis.document_id == "resilience_test"
            # Should have fewer sensory signals due to failure
            assert len(analysis.sensory_signals) < 4
    
    @pytest.mark.asyncio
    async def test_olfactory_integration(self):
        """Test integration with olfactory fusion engine."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        legal_document = """
        Complex legal contract with indemnification clauses pursuant to
        statutory requirements under 15 U.S.C. § 1681, establishing
        liability limitations and penalty provisions.
        """
        
        analysis = await processor.process_document(legal_document, "olfactory_integration_test")
        
        # Should have olfactory analysis
        assert analysis.scent_profile is not None
        
        # Should have olfactory signal
        olfactory_signal = analysis.get_signal_by_channel(SensoryChannel.OLFACTORY)
        assert olfactory_signal is not None
        assert olfactory_signal.strength >= 0.0
        assert olfactory_signal.confidence >= 0.0


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_multisensory_processor_singleton(self):
        """Test that get_multisensory_processor returns consistent instance."""
        processor1 = get_multisensory_processor(enable_olfactory=True)
        processor2 = get_multisensory_processor(enable_olfactory=True)
        
        assert processor1 is processor2
        assert isinstance(processor1, MultiSensoryLegalProcessor)
    
    @pytest.mark.asyncio
    async def test_analyze_document_multisensory_convenience(self):
        """Test convenience function for multi-sensory analysis."""
        analysis = await analyze_document_multisensory(
            "Test legal document with various provisions.",
            "convenience_test",
            {"source": "test"}
        )
        
        assert isinstance(analysis, MultiSensoryAnalysis)
        assert analysis.document_id == "convenience_test"


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""
    
    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self):
        """Test concurrent processing of multiple documents."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        documents = {
            f"concurrent_doc_{i}": f"Legal document {i} with contract provisions and clauses."
            for i in range(5)
        }
        
        # Process all documents concurrently
        tasks = [
            processor.process_document(text, doc_id)
            for doc_id, text in documents.items()
        ]
        
        analyses = await asyncio.gather(*tasks)
        
        assert len(analyses) == 5
        assert all(isinstance(analysis, MultiSensoryAnalysis) for analysis in analyses)
        
        # All should have unique document IDs
        doc_ids = [analysis.document_id for analysis in analyses]
        assert len(set(doc_ids)) == 5
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self):
        """Test processing of large documents."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        # Create a large document
        large_document = """
        # COMPREHENSIVE LEGAL AGREEMENT
        
        This agreement contains extensive provisions and clauses.
        """ + "Standard contractual language with liability provisions. " * 1000
        
        analysis = await processor.process_document(large_document, "large_doc_test")
        
        assert isinstance(analysis, MultiSensoryAnalysis)
        assert analysis.document_id == "large_doc_test"
        # Should handle large documents without errors
    
    @pytest.mark.asyncio
    async def test_empty_document_handling(self):
        """Test handling of empty documents."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        analysis = await processor.process_document("", "empty_doc_test")
        
        assert isinstance(analysis, MultiSensoryAnalysis)
        assert analysis.document_id == "empty_doc_test"
        # Should handle empty documents gracefully


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_contract_analysis_workflow(self):
        """Test complete contract analysis workflow."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        contract = """
        # SOFTWARE DEVELOPMENT AGREEMENT
        
        **Effective Date:** March 1, 2024
        **Parties:** TechCorp Inc. and DevStudio LLC
        
        ## 1. SCOPE OF WORK
        
        Developer shall create custom software pursuant to specifications
        in Exhibit A, in compliance with applicable regulations including
        but not limited to 15 U.S.C. § 7001 (E-SIGN Act).
        
        ### 1.1 Deliverables
        1. Software application
        2. Documentation
        3. Source code
        4. User training
        
        ## 2. COMPENSATION
        
        Total contract value: $150,000
        Payment schedule:
        • 25% upon signing
        • 50% upon beta delivery
        • 25% upon final acceptance
        
        ## 3. LIABILITY AND INDEMNIFICATION
        
        **IMPORTANT NOTICE:** Developer's liability is limited to contract value.
        
        Developer shall indemnify Client against third-party claims arising
        from intellectual property infringement, PROVIDED THAT Client
        promptly notifies Developer of any such claims.
        
        Penalties for breach may include liquidated damages up to $50,000.
        
        ## 4. TERMINATION
        
        Either party may terminate with 30 days written notice.
        Upon termination, all work product becomes Client property.
        
        This agreement shall be governed by Delaware law.
        """
        
        metadata = {
            "document_type": "software_development_agreement",
            "creation_date": "2024-03-01T09:00:00Z",
            "parties": ["TechCorp Inc.", "DevStudio LLC"],
            "contract_value": 150000
        }
        
        analysis = await processor.process_document(contract, "software_dev_contract", metadata)
        
        # Validate comprehensive analysis
        assert analysis.document_id == "software_dev_contract"
        assert analysis.analysis_confidence > 0.5
        
        # Check all sensory channels detected content
        channel_strengths = analysis.get_channel_strengths()
        
        # Should detect strong textual content (legal terms)
        textual_signal = analysis.get_signal_by_channel(SensoryChannel.TEXTUAL)
        assert textual_signal.strength > 0.4
        assert textual_signal.features["legal_term_density"] > 0.05
        
        # Should detect strong visual structure (headings, lists)
        visual_signal = analysis.get_signal_by_channel(SensoryChannel.VISUAL)
        assert visual_signal.strength > 0.3
        assert visual_signal.features["structure_complexity"] > 0.2
        
        # Should detect temporal freshness (2024 date)
        temporal_signal = analysis.get_signal_by_channel(SensoryChannel.TEMPORAL)
        assert temporal_signal.strength > 0.5
        
        # Should detect legal complexity through olfactory analysis
        olfactory_signal = analysis.get_signal_by_channel(SensoryChannel.OLFACTORY)
        assert olfactory_signal.strength > 0.3
        
        # Scent profile should indicate contract characteristics
        if analysis.scent_profile:
            scent_summary = get_multisensory_processor().olfactory_engine.get_scent_summary("software_dev_contract")
            if scent_summary:
                # Should detect legal complexity and risk profile
                assert scent_summary["scent_signals"]["legal_complexity"]["activated"]
                assert scent_summary["scent_signals"]["risk_profile"]["activated"]
    
    @pytest.mark.asyncio
    async def test_regulatory_document_workflow(self):
        """Test regulatory document analysis workflow."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        regulation = """
        DEPARTMENT OF COMMERCE
        Bureau of Industry and Security
        15 CFR Parts 730, 732, 734, 736, 738, 740, 742, 744, 746, 748, 750, 752, 758, 762, 772, 774
        
        RIN 0694-AI23
        
        Export Administration Regulations: Updates to License Requirements
        
        AGENCY: Bureau of Industry and Security, Commerce.
        
        ACTION: Final rule.
        
        SUMMARY: The Bureau of Industry and Security (BIS) amends the Export
        Administration Regulations (EAR) to update license requirements for
        certain dual-use items and technology transfers.
        
        DATES: This rule is effective April 15, 2024.
        
        SUPPLEMENTARY INFORMATION:
        
        I. Background
        
        The EAR controls the export of dual-use items for national security
        and foreign policy reasons. This rule updates regulations to address
        emerging technologies and security concerns.
        
        II. Summary of Changes
        
        1. License Requirements
           - New controls on quantum computing technology
           - Updated semiconductor manufacturing equipment controls
           - Enhanced end-user verification requirements
        
        2. Penalties
           
           Violations may result in:
           • Civil penalties up to $300,000 per violation
           • Criminal penalties including imprisonment
           • Denial of export privileges
        
        3. Compliance Deadlines
           
           - Existing licensees: 90 days to comply
           - New applications: Effective immediately
           - Industry guidance: Available within 30 days
        
        III. Economic Impact
        
        BIS estimates compliance costs of $2.5 million annually across
        affected industries. Benefits include enhanced national security
        and reduced technology transfer risks.
        
        This rule shall take effect notwithstanding any other provision.
        """
        
        metadata = {
            "document_type": "federal_regulation",
            "agency": "Bureau of Industry and Security",
            "effective_date": "2024-04-15",
            "cfr_parts": ["730", "732", "734", "736", "738", "740", "742", "744", "746", "748", "750", "752", "758", "762", "772", "774"]
        }
        
        analysis = await processor.process_document(regulation, "export_control_regulation", metadata)
        
        # Validate regulatory document characteristics
        assert analysis.document_id == "export_control_regulation"
        
        # Should detect strong statutory authority
        olfactory_signal = analysis.get_signal_by_channel(SensoryChannel.OLFACTORY)
        if analysis.scent_profile:
            authority_signal = analysis.scent_profile.get_signal_by_type(
                get_multisensory_processor().olfactory_engine.receptors[
                    list(get_multisensory_processor().olfactory_engine.receptors.keys())[1]
                ].receptor_type  # Statutory authority receptor
            )
            # Note: This is a simplified test - in practice we'd import the enum
            
        # Should detect recent temporal content
        temporal_signal = analysis.get_signal_by_channel(SensoryChannel.TEMPORAL)
        assert temporal_signal.strength > 0.6  # 2024 effective date
        
        # Should detect structured visual content
        visual_signal = analysis.get_signal_by_channel(SensoryChannel.VISUAL)
        assert visual_signal.strength > 0.4  # Well-structured document
        
        # Should detect high penalty/risk content
        if analysis.scent_profile:
            scent_summary = get_multisensory_processor().olfactory_engine.get_scent_summary("export_control_regulation")
            if scent_summary:
                # Should detect risk profile due to penalties
                assert scent_summary["scent_signals"]["risk_profile"]["activated"]
    
    @pytest.mark.asyncio
    async def test_document_classification_accuracy(self):
        """Test document classification accuracy using multi-sensory analysis."""
        processor = MultiSensoryLegalProcessor(enable_olfactory=True)
        
        test_documents = {
            "simple_contract": {
                "text": "Basic service agreement between two parties with standard terms.",
                "expected_complexity": "low",
                "expected_type": "contract"
            },
            "complex_contract": {
                "text": """
                WHEREAS the parties hereto desire to enter into this comprehensive
                agreement pursuant to applicable statutes including 15 U.S.C. § 1681,
                NOTWITHSTANDING any prior agreements, the Contractor shall indemnify
                and hold harmless the Company from any claims, damages, liabilities,
                penalties, or sanctions arising from negligent performance, PROVIDED THAT
                such indemnification shall not exceed the total contract value.
                """,
                "expected_complexity": "high",
                "expected_type": "contract"
            },
            "recent_statute": {
                "text": """
                Effective January 1, 2024, the Consumer Data Protection Act establishes
                new requirements for data processing and includes penalties up to
                $100,000 per violation for non-compliance with privacy regulations.
                """,
                "expected_complexity": "medium",
                "expected_type": "statute"
            }
        }
        
        classifications = {}
        
        for doc_id, doc_data in test_documents.items():
            analysis = await processor.process_document(doc_data["text"], doc_id)
            
            # Classify based on sensory signals
            textual_signal = analysis.get_signal_by_channel(SensoryChannel.TEXTUAL)
            olfactory_signal = analysis.get_signal_by_channel(SensoryChannel.OLFACTORY)
            
            # Simple classification logic
            if textual_signal and olfactory_signal:
                complexity = "high" if olfactory_signal.strength > 0.5 else "medium" if olfactory_signal.strength > 0.2 else "low"
                doc_type = "contract" if textual_signal.features.get("legal_term_density", 0) > 0.1 else "statute"
            else:
                complexity = "unknown"
                doc_type = "unknown"
            
            classifications[doc_id] = {
                "predicted_complexity": complexity,
                "predicted_type": doc_type,
                "expected_complexity": doc_data["expected_complexity"],
                "expected_type": doc_data["expected_type"]
            }
        
        # Evaluate classification accuracy
        correct_predictions = 0
        total_predictions = 0
        
        for doc_id, classification in classifications.items():
            if classification["predicted_complexity"] == classification["expected_complexity"]:
                correct_predictions += 1
            total_predictions += 1
            
            # Note: Type classification is simplified for this test
            
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Should achieve reasonable accuracy with multi-sensory approach
        assert accuracy >= 0.5  # At least 50% accuracy as baseline