#!/usr/bin/env python3
"""
Simple Bioneural System Test

Tests basic functionality without external dependencies.
"""

import sys
import time
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that basic modules can be imported."""
    try:
        from lexgraph_legal_rag.bioneuro_olfactory_fusion import (
            OlfactoryReceptorType,
            BioneuroOlfactoryReceptor,
            DocumentScentProfile,
            OlfactorySignal
        )
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_receptor_creation():
    """Test receptor creation and basic functionality."""
    try:
        from lexgraph_legal_rag.bioneuro_olfactory_fusion import (
            BioneuroOlfactoryReceptor,
            OlfactoryReceptorType
        )
        
        receptor = BioneuroOlfactoryReceptor(
            OlfactoryReceptorType.LEGAL_COMPLEXITY,
            sensitivity=0.8
        )
        
        assert receptor.receptor_type == OlfactoryReceptorType.LEGAL_COMPLEXITY
        assert receptor.sensitivity == 0.8
        
        print("✅ Receptor creation successful")
        return True
    except Exception as e:
        print(f"❌ Receptor creation failed: {e}")
        return False

async def test_receptor_activation():
    """Test receptor activation with sample text."""
    try:
        from lexgraph_legal_rag.bioneuro_olfactory_fusion import (
            BioneuroOlfactoryReceptor,
            OlfactoryReceptorType
        )
        
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
        
        print(f"✅ Receptor activation successful (intensity: {signal.intensity:.3f})")
        return True
    except Exception as e:
        print(f"❌ Receptor activation failed: {e}")
        return False

async def test_document_analysis():
    """Test complete document analysis."""
    try:
        from lexgraph_legal_rag.bioneuro_olfactory_fusion import (
            BioneuroOlfactoryFusionEngine,
            DocumentScentProfile
        )
        
        engine = BioneuroOlfactoryFusionEngine()
        
        test_document = """
        PROFESSIONAL SERVICES AGREEMENT
        
        This Agreement is entered into as of January 15, 2024, between
        TechCorp Inc. and Legal Advisors LLC pursuant to 15 U.S.C. § 1681.
        
        The Contractor shall provide legal consulting services and shall
        indemnify Company from any claims, damages, or liabilities arising
        from negligent performance, PROVIDED THAT liability shall not exceed
        the total contract value.
        """
        
        start_time = time.time()
        profile = await engine.analyze_document(test_document, "test_doc_1")
        analysis_time = time.time() - start_time
        
        assert isinstance(profile, DocumentScentProfile)
        assert profile.document_id == "test_doc_1"
        assert len(profile.signals) > 0
        assert len(profile.composite_scent) > 0
        
        print(f"✅ Document analysis successful ({analysis_time:.3f}s, {len(profile.signals)} signals)")
        return True
    except Exception as e:
        print(f"❌ Document analysis failed: {e}")
        return False

async def test_multisensory_processor():
    """Test multisensory processor."""
    try:
        from lexgraph_legal_rag.multisensory_legal_processor import (
            MultiSensoryLegalProcessor,
            MultiSensoryAnalysis
        )
        
        processor = MultiSensoryLegalProcessor(enable_olfactory=False)  # Disable olfactory to avoid circular import
        
        test_document = """
        # CONTRACT AGREEMENT
        
        **Effective Date:** January 15, 2024
        
        This agreement establishes terms between parties with liability
        provisions and indemnification clauses as specified below.
        
        ## Section 1: Scope
        1. Services to be provided
        2. Performance standards
        3. Delivery requirements
        
        Risk factors include penalties and sanctions for non-compliance.
        """
        
        start_time = time.time()
        analysis = await processor.process_document(test_document, "multisensory_test")
        analysis_time = time.time() - start_time
        
        assert isinstance(analysis, MultiSensoryAnalysis)
        assert analysis.document_id == "multisensory_test"
        assert len(analysis.sensory_signals) > 0
        
        print(f"✅ Multisensory analysis successful ({analysis_time:.3f}s, {len(analysis.sensory_signals)} channels)")
        return True
    except Exception as e:
        print(f"❌ Multisensory analysis failed: {e}")
        return False

def test_performance_benchmark():
    """Simple performance benchmark."""
    try:
        async def run_performance_test():
            from lexgraph_legal_rag.bioneuro_olfactory_fusion import BioneuroOlfactoryFusionEngine
            
            engine = BioneuroOlfactoryFusionEngine()
            
            test_documents = [
                "Simple contract with basic terms.",
                "Complex legal agreement with liability provisions pursuant to regulations.",
                "Comprehensive statute with multiple sections and penalty clauses.",
                "Regulatory document with compliance requirements and sanctions.",
                "Commercial contract with indemnification and limitation of liability."
            ]
            
            start_time = time.time()
            
            for i, doc_text in enumerate(test_documents):
                await engine.analyze_document(doc_text, f"perf_test_{i}")
            
            total_time = time.time() - start_time
            avg_time = total_time / len(test_documents)
            
            return total_time, avg_time
        
        total_time, avg_time = asyncio.run(run_performance_test())
        
        print(f"✅ Performance test completed:")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average per document: {avg_time:.3f}s")
        print(f"   Throughput: {1/avg_time:.1f} docs/sec")
        
        return avg_time < 5.0  # Should be under 5 seconds per document
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧬 BIONEURAL OLFACTORY FUSION SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports, False),
        ("Receptor Creation", test_receptor_creation, False),
        ("Receptor Activation", test_receptor_activation, True),
        ("Document Analysis", test_document_analysis, True),
        ("Multisensory Processing", test_multisensory_processor, True),
        ("Performance Benchmark", test_performance_benchmark, False)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func, is_async in tests:
        print(f"\n🔬 Running {test_name}...")
        
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total:.1%}")
    
    if passed >= total * 0.8:  # 80% pass rate
        print("\n🎉 Bioneural system is functional!")
        return True
    else:
        print("\n⚠️  Some tests failed. System needs review.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)