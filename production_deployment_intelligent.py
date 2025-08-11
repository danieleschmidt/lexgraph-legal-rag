#!/usr/bin/env python3
"""
Production deployment configuration for the enhanced intelligent legal RAG system.
Includes all new AI enhancements, optimizations, and production-ready configurations.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from lexgraph_legal_rag.intelligent_multi_agent import (
    get_intelligent_multi_agent_system,
    process_legal_query,
    get_system_report
)
from lexgraph_legal_rag.intelligent_query_processor import get_query_processor
from lexgraph_legal_rag.semantic_cache import get_semantic_cache
from lexgraph_legal_rag.auto_optimizer import get_auto_optimizer, OptimizationStrategy
from lexgraph_legal_rag.advanced_security import get_security_manager
from lexgraph_legal_rag.distributed_intelligence import get_distributed_processor


class ProductionIntelligentRAGSystem:
    """Production-ready intelligent legal RAG system with all enhancements."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.system = None
        self.config = self._load_production_config()
        
        self.logger.info("Initializing Production Intelligent Legal RAG System v4.0")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup production logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('intelligent_rag_system.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_production_config(self) -> Dict[str, Any]:
        """Load production configuration."""
        return {
            # Cache Configuration
            'cache': {
                'max_size': int(os.getenv('CACHE_MAX_SIZE', '10000')),
                'ttl_seconds': int(os.getenv('CACHE_TTL_SECONDS', '7200')),
                'similarity_threshold': float(os.getenv('CACHE_SIMILARITY_THRESHOLD', '0.75'))
            },
            
            # Optimization Configuration
            'optimization': {
                'strategy': os.getenv('OPTIMIZATION_STRATEGY', 'adaptive'),
                'auto_tune': os.getenv('AUTO_TUNE_ENABLED', 'true').lower() == 'true'
            },
            
            # Security Configuration
            'security': {
                'rate_limit': int(os.getenv('RATE_LIMIT_PER_MINUTE', '300')),
                'enable_advanced_validation': os.getenv('ADVANCED_VALIDATION', 'true').lower() == 'true',
                'block_on_critical': os.getenv('BLOCK_CRITICAL_THREATS', 'true').lower() == 'true'
            },
            
            # Distributed Processing
            'distributed': {
                'enable_distributed': os.getenv('ENABLE_DISTRIBUTED', 'true').lower() == 'true',
                'worker_nodes': int(os.getenv('WORKER_NODES', '3')),
                'load_balancing_strategy': os.getenv('LOAD_BALANCING', 'complexity_aware')
            },
            
            # Performance Targets
            'performance': {
                'target_response_time': float(os.getenv('TARGET_RESPONSE_TIME', '2.0')),
                'target_cache_hit_rate': float(os.getenv('TARGET_CACHE_HIT_RATE', '0.85')),
                'max_concurrent_queries': int(os.getenv('MAX_CONCURRENT_QUERIES', '100'))
            }
        }
    
    async def initialize(self) -> None:
        """Initialize all intelligent subsystems."""
        self.logger.info("Initializing intelligent subsystems...")
        
        # Initialize core system
        self.system = get_intelligent_multi_agent_system()
        
        # Configure semantic cache
        cache = get_semantic_cache()
        cache.max_size = self.config['cache']['max_size']
        cache.ttl_seconds = self.config['cache']['ttl_seconds']
        cache.similarity_threshold = self.config['cache']['similarity_threshold']
        
        # Configure auto-optimizer
        optimizer = get_auto_optimizer()
        strategy_name = self.config['optimization']['strategy'].upper()
        if hasattr(OptimizationStrategy, strategy_name):
            optimizer.strategy = getattr(OptimizationStrategy, strategy_name)
            self.logger.info(f"Set optimization strategy to: {strategy_name}")
        
        # Configure security
        security = get_security_manager()
        security.block_on_critical = self.config['security']['block_on_critical']
        
        # Setup distributed processing
        if self.config['distributed']['enable_distributed']:
            distributed = get_distributed_processor()
            
            # Add worker nodes based on configuration
            worker_count = self.config['distributed']['worker_nodes']
            for i in range(worker_count):
                distributed.add_worker(
                    f'worker-{i+1}',
                    'localhost',
                    8001 + i,
                    max_concurrent=20,
                    specializations=['general-law', 'contract-law']
                )
            
            self.logger.info(f"Configured {worker_count} distributed workers")
        
        self.logger.info("✅ All intelligent subsystems initialized successfully")
    
    async def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a legal query with full intelligent enhancement."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Process with intelligent multi-agent system
            result = await process_legal_query(query, **kwargs)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Check performance against targets
            target_time = self.config['performance']['target_response_time']
            if processing_time > target_time:
                self.logger.warning(f"Query exceeded target response time: {processing_time:.2f}s > {target_time}s")
            
            return {
                'success': True,
                'result': result,
                'actual_processing_time': processing_time,
                'performance_metrics': {
                    'meets_target_time': processing_time <= target_time,
                    'optimizations_applied': len(result.optimizations_applied),
                    'cache_hit': result.cache_hit,
                    'security_validated': result.security_validated
                }
            }
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'actual_processing_time': asyncio.get_event_loop().time() - start_time
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        health_status = {
            'overall': 'healthy',
            'components': {},
            'performance': {},
            'timestamp': asyncio.get_event_loop().time()
        }
        
        try:
            # Get comprehensive system report
            report = get_system_report()
            
            # Check cache performance
            cache_stats = report['semantic_cache']
            cache_healthy = (
                cache_stats['hit_rate'] >= self.config['performance']['target_cache_hit_rate'] * 0.8
                and cache_stats['size'] < cache_stats['max_size'] * 0.9
            )
            health_status['components']['cache'] = 'healthy' if cache_healthy else 'degraded'
            
            # Check optimization status
            optimization_stats = report['optimization']
            optimization_healthy = optimization_stats['active_issues'] < 3
            health_status['components']['optimization'] = 'healthy' if optimization_healthy else 'degraded'
            
            # Check security status
            security_stats = report['security']
            security_healthy = security_stats['blocked_requests'] < security_stats['total_events'] * 0.1
            health_status['components']['security'] = 'healthy' if security_healthy else 'warning'
            
            # Check distributed system
            distributed_stats = report['distributed']
            if 'cluster_stats' in distributed_stats and 'cluster_overview' in distributed_stats['cluster_stats']:
                cluster_overview = distributed_stats['cluster_stats']['cluster_overview']
                distributed_healthy = (
                    cluster_overview['healthy_workers'] > 0
                    and cluster_overview['cluster_utilization'] < 0.9
                )
                health_status['components']['distributed'] = 'healthy' if distributed_healthy else 'degraded'
            
            # Overall performance metrics
            health_status['performance'] = {
                'cache_hit_rate': cache_stats['hit_rate'],
                'optimization_active_issues': optimization_stats['active_issues'],
                'security_threat_rate': security_stats.get('blocked_requests', 0) / max(security_stats.get('total_events', 1), 1),
                'total_queries_processed': report['agent_system']['total_queries_processed']
            }
            
            # Determine overall health
            component_statuses = list(health_status['components'].values())
            if 'degraded' in component_statuses:
                health_status['overall'] = 'degraded'
            elif component_statuses.count('warning') > 1:
                health_status['overall'] = 'warning'
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health_status['overall'] = 'error'
            health_status['error'] = str(e)
        
        return health_status
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        try:
            report = get_system_report()
            
            return {
                'query_processing': report['query_processing'],
                'cache_performance': {
                    'hit_rate': report['semantic_cache']['hit_rate'],
                    'semantic_hit_rate': report['semantic_cache']['semantic_hit_rate'],
                    'total_hits': report['semantic_cache']['total_hits'],
                    'total_misses': report['semantic_cache']['total_misses'],
                    'cache_utilization': report['semantic_cache']['size'] / report['semantic_cache']['max_size']
                },
                'optimization_status': {
                    'strategy': report['optimization']['strategy'],
                    'recent_optimizations': report['optimization']['recent_optimizations'],
                    'success_rate': report['optimization']['success_rate'],
                    'active_issues': report['optimization']['active_issues']
                },
                'security_metrics': {
                    'total_events': report['security']['total_events'],
                    'blocked_requests': report['security']['blocked_requests'],
                    'threat_sources': len(report['security']['top_threat_sources'])
                },
                'system_performance': {
                    'average_processing_time': report['agent_system']['average_processing_time'],
                    'total_queries': report['agent_system']['total_queries_processed']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {'error': str(e)}
    
    async def run_benchmark(self, num_queries: int = 100) -> Dict[str, Any]:
        """Run performance benchmark with intelligent queries."""
        self.logger.info(f"Starting benchmark with {num_queries} queries...")
        
        # Diverse test queries representing different complexities
        test_queries = [
            "What is contract law?",
            "Explain breach of contract damages in commercial agreements",
            "Find cases about negligence liability in healthcare",
            "Analyze the constitutional implications of statutory interpretation",
            "Compare common law and statutory approaches to intellectual property",
            "What are the key elements of a valid contract formation?",
            "Summarize recent developments in employment discrimination law",
            "Define force majeure in international commercial contracts",
            "What remedies are available for trademark infringement?",
            "Explain the doctrine of precedent in legal decision making"
        ]
        
        results = {
            'total_queries': num_queries,
            'successful_queries': 0,
            'failed_queries': 0,
            'cache_hits': 0,
            'total_processing_time': 0.0,
            'response_times': [],
            'optimizations_applied': [],
            'error_details': []
        }
        
        start_time = asyncio.get_event_loop().time()
        
        # Process queries
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            if i > 0:  # Add variation to avoid identical queries
                query += f" (query {i})"
            
            try:
                result = await self.process_query(query, source_ip=f"192.168.1.{i % 255}")
                
                if result['success']:
                    results['successful_queries'] += 1
                    processing_time = result['actual_processing_time']
                    results['total_processing_time'] += processing_time
                    results['response_times'].append(processing_time)
                    
                    # Track performance metrics
                    if result['performance_metrics']['cache_hit']:
                        results['cache_hits'] += 1
                    
                    results['optimizations_applied'].append(
                        result['performance_metrics']['optimizations_applied']
                    )
                else:
                    results['failed_queries'] += 1
                    results['error_details'].append(result.get('error', 'Unknown error'))
                
                # Progress logging
                if (i + 1) % 25 == 0:
                    self.logger.info(f"Benchmark progress: {i + 1}/{num_queries} queries processed")
                
            except Exception as e:
                results['failed_queries'] += 1
                results['error_details'].append(str(e))
                self.logger.error(f"Benchmark query {i} failed: {e}")
        
        total_benchmark_time = asyncio.get_event_loop().time() - start_time
        
        # Calculate statistics
        if results['response_times']:
            results['average_response_time'] = sum(results['response_times']) / len(results['response_times'])
            results['min_response_time'] = min(results['response_times'])
            results['max_response_time'] = max(results['response_times'])
            results['median_response_time'] = sorted(results['response_times'])[len(results['response_times']) // 2]
        
        results['cache_hit_rate'] = results['cache_hits'] / max(results['successful_queries'], 1)
        results['success_rate'] = results['successful_queries'] / num_queries
        results['queries_per_second'] = num_queries / total_benchmark_time
        results['total_benchmark_time'] = total_benchmark_time
        results['average_optimizations'] = sum(results['optimizations_applied']) / max(len(results['optimizations_applied']), 1)
        
        # Performance assessment
        target_time = self.config['performance']['target_response_time']
        target_cache_rate = self.config['performance']['target_cache_hit_rate']
        
        results['performance_assessment'] = {
            'meets_response_time_target': results.get('average_response_time', float('inf')) <= target_time,
            'meets_cache_hit_target': results['cache_hit_rate'] >= target_cache_rate,
            'overall_performance': 'excellent' if (
                results.get('average_response_time', 0) <= target_time * 0.8 and
                results['cache_hit_rate'] >= target_cache_rate and
                results['success_rate'] >= 0.95
            ) else 'good' if (
                results.get('average_response_time', 0) <= target_time and
                results['success_rate'] >= 0.9
            ) else 'needs_improvement'
        }
        
        self.logger.info(f"✅ Benchmark completed: {results['success_rate']:.1%} success rate, "
                        f"{results.get('average_response_time', 0):.3f}s avg response time")
        
        return results


async def main():
    """Main production deployment and benchmark."""
    print("🚀 PRODUCTION INTELLIGENT LEGAL RAG SYSTEM v4.0")
    print("=" * 60)
    
    # Initialize system
    system = ProductionIntelligentRAGSystem()
    await system.initialize()
    
    # Run health check
    print("\n🏥 SYSTEM HEALTH CHECK")
    print("-" * 30)
    health = await system.health_check()
    print(f"Overall Status: {health['overall'].upper()}")
    
    for component, status in health['components'].items():
        status_emoji = "✅" if status == "healthy" else "⚠️" if status == "warning" else "❌"
        print(f"{status_emoji} {component}: {status}")
    
    if 'performance' in health:
        perf = health['performance']
        print(f"\n📊 Performance Metrics:")
        print(f"  - Cache Hit Rate: {perf['cache_hit_rate']:.1%}")
        print(f"  - Active Issues: {perf['optimization_active_issues']}")
        print(f"  - Security Threat Rate: {perf['security_threat_rate']:.1%}")
        print(f"  - Total Queries Processed: {perf['total_queries_processed']}")
    
    # Run performance benchmark
    print("\n🏃 PERFORMANCE BENCHMARK")
    print("-" * 30)
    
    benchmark_results = await system.run_benchmark(50)  # 50 queries for demo
    
    print(f"Success Rate: {benchmark_results['success_rate']:.1%}")
    print(f"Cache Hit Rate: {benchmark_results['cache_hit_rate']:.1%}")
    print(f"Average Response Time: {benchmark_results.get('average_response_time', 0):.3f}s")
    print(f"Queries per Second: {benchmark_results['queries_per_second']:.1f}")
    print(f"Performance Assessment: {benchmark_results['performance_assessment']['overall_performance'].upper()}")
    
    if benchmark_results['failed_queries'] > 0:
        print(f"\n⚠️ {benchmark_results['failed_queries']} queries failed")
    
    # Test sample queries
    print("\n🧪 SAMPLE INTELLIGENT QUERY PROCESSING")
    print("-" * 45)
    
    sample_queries = [
        "What are the key elements of contract formation?",
        "Explain breach of warranty in software licensing",
        "Find cases about negligence in autonomous vehicles"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{i}. Query: {query}")
        result = await system.process_query(query, source_ip="127.0.0.1")
        
        if result['success']:
            query_result = result['result']
            print(f"   Intent: {query_result.intent.value}")
            print(f"   Enhanced: {query_result.enhanced_query[:100]}...")
            print(f"   Processing Time: {result['actual_processing_time']:.3f}s")
            print(f"   Cache Hit: {'Yes' if query_result.cache_hit else 'No'}")
            print(f"   Optimizations: {', '.join(query_result.optimizations_applied)}")
            print(f"   Response: {query_result.response[:150]}...")
        else:
            print(f"   ❌ Failed: {result['error']}")
    
    # Final system report
    print("\n📈 COMPREHENSIVE SYSTEM REPORT")
    print("-" * 35)
    
    metrics = await system.get_performance_metrics()
    if 'error' not in metrics:
        print("Query Processing:")
        if 'total_queries' in metrics['query_processing']:
            print(f"  - Total Processed: {metrics['query_processing']['total_queries']}")
        if 'popular_terms' in metrics['query_processing']:
            print(f"  - Popular Terms: {list(metrics['query_processing']['popular_terms'].keys())[:5]}")
        
        print(f"\nCache Performance:")
        cache_perf = metrics['cache_performance']
        print(f"  - Hit Rate: {cache_perf['hit_rate']:.1%}")
        print(f"  - Semantic Hit Rate: {cache_perf['semantic_hit_rate']:.1%}")
        print(f"  - Cache Utilization: {cache_perf['cache_utilization']:.1%}")
        
        print(f"\nOptimization Status:")
        opt_status = metrics['optimization_status']
        print(f"  - Strategy: {opt_status['strategy']}")
        print(f"  - Recent Optimizations: {opt_status['recent_optimizations']}")
        print(f"  - Success Rate: {opt_status['success_rate']:.1%}")
    
    print(f"\n🎉 PRODUCTION DEPLOYMENT ASSESSMENT: {'✅ READY' if health['overall'] == 'healthy' else '⚠️ NEEDS ATTENTION'}")
    
    if health['overall'] == 'healthy' and benchmark_results['performance_assessment']['overall_performance'] in ['excellent', 'good']:
        print("\n✨ System is ready for production deployment!")
        print("   All intelligent enhancements are functioning optimally.")
        print("   Performance meets or exceeds production requirements.")
        print("   Security, caching, and optimization systems operational.")
    else:
        print(f"\n⚠️ System needs optimization before full production deployment.")
        print(f"   Health Status: {health['overall']}")
        print(f"   Performance: {benchmark_results['performance_assessment']['overall_performance']}")


if __name__ == "__main__":
    asyncio.run(main())