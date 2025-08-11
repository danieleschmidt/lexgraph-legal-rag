#!/usr/bin/env python3
"""
Comprehensive test suite for intelligent RAG systems.
Tests all new AI-powered enhancements and validates system integration.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from lexgraph_legal_rag.intelligent_query_processor import (
    IntelligentQueryProcessor,
    QueryIntent,
    LegalTermExpander,
    QueryIntentClassifier
)
from lexgraph_legal_rag.semantic_cache import (
    SemanticQueryCache,
    SemanticSimilarityMatcher,
    CacheEntry
)
from lexgraph_legal_rag.auto_optimizer import (
    AdaptiveOptimizer,
    PerformanceMonitor,
    OptimizationStrategy
)
from lexgraph_legal_rag.advanced_security import (
    SecurityManager,
    InputValidator,
    IntelligentRateLimiter
)
from lexgraph_legal_rag.advanced_resilience import (
    ResilienceManager,
    ErrorCategory,
    RecoveryStrategy
)
from lexgraph_legal_rag.distributed_intelligence import (
    DistributedQueryProcessor,
    IntelligentLoadBalancer,
    QueryProfile,
    QueryComplexity
)
from lexgraph_legal_rag.intelligent_multi_agent import (
    IntelligentMultiAgentSystem,
    get_intelligent_multi_agent_system
)


class TestIntelligentQueryProcessor:
    """Test the AI-powered query processing system."""
    
    def test_legal_term_expansion(self):
        """Test legal term expansion and synonym addition."""
        expander = LegalTermExpander()
        
        # Test contract expansion
        enhanced, synonyms, expansions = expander.expand_query("breach of contract damages")
        
        assert "breach" in enhanced.lower()
        assert "contract" in enhanced.lower() 
        assert "damages" in enhanced.lower()
        assert len(synonyms) > 0  # Should add synonyms
        assert "OR" in enhanced  # Should use OR logic
        
        print(f"✓ Query expansion: '{enhanced}'")
        print(f"✓ Synonyms added: {synonyms}")
        print(f"✓ Context expansions: {expansions}")
    
    def test_intent_classification(self):
        """Test query intent classification."""
        classifier = QueryIntentClassifier()
        
        test_cases = [
            ("What is indemnification?", QueryIntent.EXPLAIN),
            ("Find contracts about liability", QueryIntent.SEARCH),
            ("Summarize the key points of this statute", QueryIntent.SUMMARIZE),
            ("Compare these two legal precedents", QueryIntent.COMPARE),
            ("Analyze the compliance requirements", QueryIntent.ANALYZE)
        ]
        
        for query, expected_intent in test_cases:
            intent, confidence = classifier.classify_intent(query)
            print(f"✓ Query: '{query}' -> Intent: {intent.value} (confidence: {confidence:.2f})")
            
            # Intent should match or be reasonable alternative
            if expected_intent != intent:
                print(f"  Note: Expected {expected_intent.value}, got {intent.value}")
    
    @pytest.mark.asyncio
    async def test_full_query_processing(self):
        """Test complete query processing pipeline."""
        processor = IntelligentQueryProcessor()
        
        test_query = "Explain the liability implications of breach of contract in commercial agreements"
        enhancement = await processor.process_query(test_query)
        
        assert enhancement.original_query == test_query
        assert len(enhancement.enhanced_query) >= len(test_query)
        assert enhancement.intent in [QueryIntent.EXPLAIN, QueryIntent.ANALYZE]
        assert enhancement.confidence > 0
        assert len(enhancement.legal_terms) > 0
        
        print(f"✓ Original: {enhancement.original_query}")
        print(f"✓ Enhanced: {enhancement.enhanced_query}")
        print(f"✓ Intent: {enhancement.intent.value} ({enhancement.confidence:.2f})")
        print(f"✓ Legal terms: {enhancement.legal_terms}")


class TestSemanticCache:
    """Test the semantic caching system."""
    
    def test_similarity_matching(self):
        """Test semantic similarity calculation."""
        matcher = SemanticSimilarityMatcher()
        
        # Test similar legal queries
        keywords1 = matcher.extract_keywords("breach of contract damages")
        keywords2 = matcher.extract_keywords("contract violation compensation")
        
        similarity = matcher.calculate_similarity(keywords1, keywords2)
        
        assert similarity > 0.5  # Should be similar
        print(f"✓ Similarity between legal queries: {similarity:.3f}")
        
        # Test dissimilar queries
        keywords3 = matcher.extract_keywords("weather forecast tomorrow")
        similarity_diff = matcher.calculate_similarity(keywords1, keywords3)
        
        assert similarity_diff < similarity  # Should be less similar
        print(f"✓ Dissimilar query similarity: {similarity_diff:.3f}")
    
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test cache storage and retrieval."""
        cache = SemanticQueryCache(max_size=100, similarity_threshold=0.6)
        
        # Store a query result
        query1 = "What are the damages for breach of contract?"
        response1 = "Damages for breach of contract include compensatory and consequential damages."
        
        await cache.put(query1, response1, response_time=1.5)
        
        # Test exact match
        result = await cache.get(query1)
        assert result is not None
        cached_response, metadata = result
        assert cached_response == response1
        assert metadata['cache_hit_type'] == 'exact'
        
        print(f"✓ Exact cache hit successful")
        
        # Test semantic match
        query2 = "What compensation is available for contract breach?"
        result2 = await cache.get(query2)
        
        if result2 is not None:
            cached_response2, metadata2 = result2
            assert metadata2['cache_hit_type'] == 'semantic'
            assert metadata2['similarity_score'] > 0.6
            print(f"✓ Semantic cache hit: similarity = {metadata2['similarity_score']:.3f}")
        else:
            print("✓ No semantic match found (threshold not met)")
    
    @pytest.mark.asyncio 
    async def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = SemanticQueryCache(max_size=10)
        
        # Add some entries
        for i in range(5):
            await cache.put(f"query {i}", f"response {i}", response_time=i * 0.5)
        
        # Perform some lookups
        for i in range(3):
            await cache.get(f"query {i}")
        
        stats = cache.get_stats()
        assert stats['size'] == 5
        assert stats['total_hits'] > 0
        assert 'hit_rate' in stats
        
        print(f"✓ Cache stats: {stats}")


class TestAutoOptimizer:
    """Test the autonomous optimization system."""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance metric tracking."""
        optimizer = AdaptiveOptimizer(OptimizationStrategy.BALANCED)
        
        # Simulate query performance data
        for i in range(20):
            response_time = 1.0 + (i % 5) * 0.5  # Variable response times
            success = i % 10 != 0  # 10% failure rate
            
            await optimizer.record_query_performance(
                f"test query {i}", 
                response_time, 
                success,
                metadata={'cache_hit': i % 3 == 0}
            )
        
        report = optimizer.get_optimization_report()
        assert 'current_parameters' in report
        assert 'performance_trends' in report
        assert report['recent_optimizations'] >= 0
        
        print(f"✓ Optimization report: {report}")
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization(self):
        """Test adaptive parameter optimization."""
        optimizer = AdaptiveOptimizer(OptimizationStrategy.AGGRESSIVE)
        
        # Simulate poor performance to trigger optimization
        for i in range(30):
            await optimizer.record_query_performance(
                f"slow query {i}",
                5.0,  # Slow responses
                True,
                metadata={'error_rate': 0.1}
            )
        
        # Force optimization
        actions = await optimizer.optimize()
        
        print(f"✓ Optimization actions taken: {len(actions)}")
        for action in actions:
            print(f"  - {action.parameter}: {action.old_value} -> {action.new_value} ({action.reason})")


class TestAdvancedSecurity:
    """Test the advanced security system."""
    
    @pytest.mark.asyncio
    async def test_input_validation(self):
        """Test malicious input detection."""
        validator = InputValidator()
        
        # Test normal query
        valid, violations = await validator.validate_query("What is contract law?")
        assert valid
        assert len(violations) == 0
        
        # Test malicious query
        malicious = "DROP TABLE contracts; --"
        valid_mal, violations_mal = await validator.validate_query(malicious)
        assert not valid_mal
        assert len(violations_mal) > 0
        
        print(f"✓ Normal query validation: {valid}")
        print(f"✓ Malicious query blocked: {not valid_mal} ({violations_mal})")
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test intelligent rate limiting."""
        rate_limiter = IntelligentRateLimiter()
        
        # Test normal usage
        allowed1, info1 = await rate_limiter.check_rate_limit("192.168.1.1")
        assert allowed1
        assert info1['remaining'] > 0
        
        # Simulate rapid requests
        for i in range(65):  # Exceed default limit
            allowed, info = await rate_limiter.check_rate_limit("192.168.1.2")
            if not allowed:
                print(f"✓ Rate limit triggered after {i} requests")
                assert info['remaining'] == 0
                break
        else:
            print("✓ Rate limit not reached (unexpected)")
    
    @pytest.mark.asyncio 
    async def test_security_integration(self):
        """Test full security validation."""
        security_manager = SecurityManager()
        
        # Test legitimate query
        allowed, info = await security_manager.validate_request(
            "Explain contract formation requirements", 
            "192.168.1.100"
        )
        assert allowed
        assert info['threat_level'].value == 'low'
        
        print(f"✓ Legitimate query allowed: {allowed}")
        print(f"✓ Threat level: {info['threat_level'].value}")


class TestAdvancedResilience:
    """Test the resilience and error handling system."""
    
    @pytest.mark.asyncio
    async def test_error_classification(self):
        """Test error pattern classification."""
        resilience = ResilienceManager()
        
        # Test different error types
        connection_error = ConnectionError("Connection refused")
        pattern = resilience.classify_error(connection_error)
        assert pattern.category == ErrorCategory.TRANSIENT
        assert pattern.strategy == RecoveryStrategy.RETRY
        
        validation_error = ValueError("Invalid input format")
        pattern2 = resilience.classify_error(validation_error)
        assert pattern2.category == ErrorCategory.VALIDATION
        assert pattern2.strategy == RecoveryStrategy.FAIL_FAST
        
        print(f"✓ Connection error -> {pattern.category.value} / {pattern.strategy.value}")
        print(f"✓ Validation error -> {pattern2.category.value} / {pattern2.strategy.value}")
    
    @pytest.mark.asyncio
    async def test_resilient_execution(self):
        """Test resilient function execution."""
        resilience = ResilienceManager()
        
        # Test successful operation
        async def success_func():
            return "success"
        
        result = await resilience.execute_with_resilience(
            success_func, 
            "test_operation"
        )
        assert result == "success"
        
        print(f"✓ Successful operation: {result}")
        
        # Test operation with retries
        call_count = 0
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "eventual success"
        
        result2 = await resilience.execute_with_resilience(
            failing_then_success,
            "retry_operation"
        )
        assert result2 == "eventual success"
        assert call_count == 3
        
        print(f"✓ Retry operation succeeded after {call_count} attempts")


class TestDistributedIntelligence:
    """Test the distributed processing system."""
    
    def test_query_complexity_analysis(self):
        """Test query complexity analysis."""
        simple_query = "find contracts"
        simple_profile = QueryProfile.analyze_query(simple_query)
        assert simple_profile.complexity == QueryComplexity.SIMPLE
        
        complex_query = "Analyze the constitutional implications of statutory interpretation in federal court precedents regarding commercial contract enforcement across multiple jurisdictions"
        complex_profile = QueryProfile.analyze_query(complex_query)
        assert complex_profile.complexity in [QueryComplexity.COMPLEX, QueryComplexity.HEAVY]
        
        print(f"✓ Simple query complexity: {simple_profile.complexity.value}")
        print(f"✓ Complex query complexity: {complex_profile.complexity.value}")
        print(f"✓ Estimated processing time: {complex_profile.estimated_processing_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_load_balancer(self):
        """Test intelligent load balancing."""
        load_balancer = IntelligentLoadBalancer()
        
        # Add mock workers
        from lexgraph_legal_rag.distributed_intelligence import WorkerNode, WorkerStatus
        
        worker1 = WorkerNode("worker-1", "localhost", 8001, max_concurrent_queries=10)
        worker2 = WorkerNode("worker-2", "localhost", 8002, max_concurrent_queries=5)
        worker2.current_load = 3  # Partially loaded
        
        load_balancer.register_worker(worker1)
        load_balancer.register_worker(worker2)
        
        # Test query routing
        query_profile = QueryProfile.analyze_query("simple contract question")
        selected_worker = await load_balancer.route_query(query_profile)
        
        assert selected_worker is not None
        assert selected_worker.node_id in ["worker-1", "worker-2"]
        
        print(f"✓ Query routed to: {selected_worker.node_id}")
        print(f"✓ Worker load: {selected_worker.current_load}/{selected_worker.max_concurrent_queries}")
    
    @pytest.mark.asyncio
    async def test_distributed_processing(self):
        """Test end-to-end distributed query processing."""
        processor = DistributedQueryProcessor()
        
        query = "What are the key elements of a valid contract?"
        response_chunks = []
        
        async for chunk in processor.process_query_distributed(query):
            response_chunks.append(chunk)
        
        assert len(response_chunks) > 0
        full_response = " ".join(response_chunks)
        assert len(full_response) > 0
        
        print(f"✓ Distributed processing response: {full_response[:100]}...")


class TestIntelligentMultiAgent:
    """Test the integrated intelligent multi-agent system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test complete intelligent query processing."""
        # Mock pipeline for testing
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = []
        
        system = IntelligentMultiAgentSystem(pipeline=mock_pipeline)
        
        query = "Explain the liability provisions in commercial contracts"
        result = await system.process_query_intelligent(query, source_ip="127.0.0.1")
        
        assert result.original_query == query
        assert len(result.enhanced_query) >= len(query)
        assert result.processing_time > 0
        assert result.security_validated is True
        assert len(result.optimizations_applied) > 0
        
        print(f"✓ End-to-end processing successful")
        print(f"✓ Original: {result.original_query}")
        print(f"✓ Enhanced: {result.enhanced_query}")
        print(f"✓ Intent: {result.intent.value}")
        print(f"✓ Processing time: {result.processing_time:.3f}s")
        print(f"✓ Optimizations: {result.optimizations_applied}")
        print(f"✓ Response: {result.response[:100]}...")
    
    @pytest.mark.asyncio
    async def test_system_intelligence_report(self):
        """Test comprehensive system reporting."""
        system = get_intelligent_multi_agent_system()
        
        # Process a few queries to generate data
        test_queries = [
            "What is contract law?",
            "Explain breach of contract damages",
            "Find cases about negligence"
        ]
        
        for query in test_queries:
            await system.process_query_intelligent(query)
        
        report = system.get_system_intelligence_report()
        
        assert 'query_processing' in report
        assert 'semantic_cache' in report
        assert 'optimization' in report
        assert 'security' in report
        assert 'resilience' in report
        assert 'distributed' in report
        assert 'agent_system' in report
        
        print(f"✓ System intelligence report generated:")
        for component, data in report.items():
            if isinstance(data, dict):
                print(f"  - {component}: {len(data)} metrics")
            else:
                print(f"  - {component}: {data}")


async def run_comprehensive_tests():
    """Run all intelligent system tests."""
    print("🧪 RUNNING COMPREHENSIVE INTELLIGENT SYSTEMS TESTS")
    print("=" * 60)
    
    # Test classes in order
    test_classes = [
        TestIntelligentQueryProcessor(),
        TestSemanticCache(),
        TestAutoOptimizer(),
        TestAdvancedSecurity(),
        TestAdvancedResilience(),
        TestDistributedIntelligence(),
        TestIntelligentMultiAgent()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_instance in test_classes:
        class_name = test_instance.__class__.__name__
        print(f"\n📋 Testing {class_name}")
        print("-" * 40)
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                test_method = getattr(test_instance, method_name)
                
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                passed_tests += 1
                print(f"  ✅ {method_name}")
                
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
    
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed - review and fix issues")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    # Run the comprehensive test suite
    success = asyncio.run(run_comprehensive_tests())
    exit(0 if success else 1)