#!/usr/bin/env python3
"""Comprehensive test suite for bioneural olfactory fusion system."""

import sys
import os
import json
import time
import unittest
from pathlib import Path
from typing import Dict, List, Any
import hashlib

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / 'src'))

class TestBioneuroOlfactoryFusion(unittest.TestCase):
    """Test suite for bioneural olfactory fusion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_legal_texts = {
            'contract': '''
            This Agreement is entered into between Company and Client.
            Company shall provide services and Client shall pay fees.
            Either party may terminate upon breach. Limitation of liability applies.
            ''',
            'statute': '''
            15 U.S.C. § 1681 requires disclosure of consumer information.
            Violations may result in damages under 15 U.S.C. § 1681n.
            Civil penalties apply per 15 U.S.C. § 1681a(d).
            ''',
            'simple_text': '''
            This is a simple document with no legal content.
            It contains regular words and sentences.
            Nothing complex or specialized here.
            '''
        }
    
    def test_legal_text_analysis_basic(self):
        """Test basic legal text analysis functionality."""
        from minimal_working_demo import simple_legal_text_analysis
        
        # Test contract analysis
        result = simple_legal_text_analysis(self.sample_legal_texts['contract'])
        
        self.assertIn('categories', result)
        self.assertIn('risk_score', result)
        self.assertIn('complexity_score', result)
        self.assertIn('word_count', result)
        
        # Should detect contractual elements
        categories = [cat['category'] for cat in result['categories']]
        self.assertIn('contractual', categories)
        
        # Risk score should be reasonable
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 1)
    
    def test_bioneural_scent_simulation(self):
        """Test bioneural olfactory scent simulation."""
        from minimal_working_demo import bioneural_scent_simulation
        
        result = bioneural_scent_simulation(self.sample_legal_texts['statute'])
        
        # Verify structure
        self.assertIn('primary_scent', result)
        self.assertIn('intensity', result) 
        self.assertIn('receptor_activations', result)
        self.assertIn('document_signature', result)
        
        # Verify receptor types
        expected_receptors = [
            'legal_complexity', 'statutory_authority', 'temporal_freshness',
            'citation_density', 'risk_profile', 'semantic_coherence'
        ]
        
        for receptor in expected_receptors:
            self.assertIn(receptor, result['receptor_activations'])
        
        # Statutory text should have reasonable statutory authority  
        self.assertGreater(result['receptor_activations']['statutory_authority'], 0.4)
    
    def test_document_classification_accuracy(self):
        """Test document classification accuracy."""
        from minimal_working_demo import bioneural_scent_simulation
        
        results = {}
        for doc_type, content in self.sample_legal_texts.items():
            results[doc_type] = bioneural_scent_simulation(content)
        
        # Contract should have high legal complexity
        self.assertGreater(results['contract']['receptor_activations']['legal_complexity'], 0.7)
        
        # Statute should have reasonable statutory authority
        self.assertGreater(results['statute']['receptor_activations']['statutory_authority'], 0.4)
        
        # Simple text should have lower overall intensity
        self.assertLess(results['simple_text']['intensity'], 0.6)
    
    def test_performance_consistency(self):
        """Test performance and consistency of analysis."""
        from minimal_working_demo import bioneural_scent_simulation
        
        # Run multiple times on same text
        text = self.sample_legal_texts['contract']
        results = []
        
        start_time = time.time()
        for _ in range(100):
            result = bioneural_scent_simulation(text)
            results.append(result)
        end_time = time.time()
        
        # Should complete quickly (< 1 second for 100 runs)
        self.assertLess(end_time - start_time, 1.0)
        
        # Results should be consistent
        for i in range(1, len(results)):
            self.assertEqual(results[0]['document_signature'], results[i]['document_signature'])
            self.assertEqual(results[0]['primary_scent'], results[i]['primary_scent'])
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        from minimal_working_demo import simple_legal_text_analysis, bioneural_scent_simulation
        
        # Empty text
        result = simple_legal_text_analysis("")
        self.assertEqual(result['word_count'], 0)
        self.assertGreaterEqual(result['risk_score'], 0)
        
        # Very short text
        result = bioneural_scent_simulation("Contract.")
        self.assertIn('primary_scent', result)
        
        # Very long text (simulate)
        long_text = "This is a legal agreement. " * 1000
        result = bioneural_scent_simulation(long_text)
        self.assertIn('intensity', result)
        self.assertGreater(result['intensity'], 0)

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_demo_execution(self):
        """Test that the main demo executes successfully."""
        from minimal_working_demo import demonstrate_system
        
        # Should complete without errors
        results = demonstrate_system()
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check result structure
        for result in results:
            self.assertIn('document', result)
            self.assertIn('traditional', result)
            self.assertIn('bioneural', result)
    
    def test_output_file_creation(self):
        """Test that output files are created properly."""
        from minimal_working_demo import demonstrate_system
        
        # Clean up any existing file
        output_file = Path('bioneural_demo_results.json')
        if output_file.exists():
            output_file.unlink()
        
        # Run demo
        demonstrate_system()
        
        # Verify file creation
        self.assertTrue(output_file.exists())
        
        # Verify file content
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('timestamp', data)
        self.assertIn('results', data)
        self.assertIn('performance', data)
        
        # Clean up
        output_file.unlink()

class TestPerformanceValidation(unittest.TestCase):
    """Validate performance claims and benchmarks."""
    
    def test_processing_speed_benchmark(self):
        """Benchmark processing speed claims."""
        from minimal_working_demo import bioneural_scent_simulation
        
        # Test with medium-sized legal document
        test_doc = '''
        This Software License Agreement governs the use of software products.
        The license is granted subject to compliance with all terms herein.
        Breach of this agreement may result in termination and damages.
        User warrants that use will not infringe third party rights.
        Company disclaims all warranties except as required by law.
        Limitation of liability applies to maximum extent permitted.
        This agreement shall be governed by state law.
        ''' * 10  # Simulate larger document
        
        # Measure processing time
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            bioneural_scent_simulation(test_doc)
        
        end_time = time.time()
        processing_time = end_time - start_time
        docs_per_second = iterations / processing_time
        
        # Should process at reasonable speed (>100 docs/sec minimum)
        self.assertGreater(docs_per_second, 100)
        print(f"Processing speed: {docs_per_second:.0f} docs/sec")
    
    def test_accuracy_validation(self):
        """Validate classification accuracy improvements."""
        from minimal_working_demo import simple_legal_text_analysis, bioneural_scent_simulation
        
        # Test documents with known characteristics
        test_cases = [
            {
                'text': 'This contract shall indemnify party from liability and damages.',
                'expected_high_risk': True,
                'expected_contractual': True
            },
            {
                'text': '15 U.S.C. § 1681 provides statutory requirements for disclosures.',
                'expected_statutory': True,
                'expected_citations': True
            },
            {
                'text': 'This document contains warranty disclaimers and limitations.',
                'expected_warranty': True,
                'expected_risk': True
            }
        ]
        
        correct_classifications = 0
        total_tests = 0
        
        for case in test_cases:
            traditional = simple_legal_text_analysis(case['text'])
            bioneural = bioneural_scent_simulation(case['text'])
            
            # Test traditional analysis
            if case.get('expected_high_risk') and traditional['risk_score'] > 0.5:
                correct_classifications += 1
            if case.get('expected_contractual'):
                categories = [cat['category'] for cat in traditional['categories']]
                if 'contractual' in categories:
                    correct_classifications += 1
            
            # Test bioneural analysis
            if case.get('expected_statutory') and bioneural['receptor_activations']['statutory_authority'] > 0.5:
                correct_classifications += 1
            if case.get('expected_risk') and bioneural['receptor_activations']['risk_profile'] > 0.3:
                correct_classifications += 1
            
            total_tests += 2  # Two tests per case
        
        accuracy = correct_classifications / total_tests if total_tests > 0 else 0
        self.assertGreater(accuracy, 0.5)  # Should achieve >50% accuracy in basic tests
        print(f"Classification accuracy: {accuracy:.1%}")

def run_comprehensive_tests():
    """Run all test suites and generate report."""
    
    print("🧪 Bioneural Olfactory Fusion - Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBioneuroOlfactoryFusion,
        TestSystemIntegration,
        TestPerformanceValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Generate test report
    report = {
        'timestamp': time.time(),
        'test_duration': end_time - start_time,
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        'details': {
            'failures': [str(failure) for failure, _ in result.failures],
            'errors': [str(error) for error, _ in result.errors]
        }
    }
    
    # Save test report
    report_file = Path('test_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Test Report Summary:")
    print(f"   Tests run: {report['tests_run']}")
    print(f"   Failures: {report['failures']}")
    print(f"   Errors: {report['errors']}")
    print(f"   Success rate: {report['success_rate']:.1%}")
    print(f"   Duration: {report['test_duration']:.2f}s")
    print(f"   Report saved to: {report_file}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    
    if success:
        print("\n✅ All tests passed! System is robust and validated.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Review the test report for details.")
        sys.exit(1)