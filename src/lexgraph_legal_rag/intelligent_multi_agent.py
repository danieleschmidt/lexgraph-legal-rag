"""Enhanced multi-agent system with AI-powered intelligence and distributed processing.

This module integrates all the intelligent enhancements:
- AI-powered query processing and enhancement
- Semantic caching with similarity matching  
- Autonomous optimization and performance tuning
- Advanced security and input validation
- Distributed processing with intelligent load balancing
- Comprehensive error handling and resilience
"""

from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, AsyncIterator, Any, Tuple
import logging

# Import our intelligent systems
from .intelligent_query_processor import (
    get_query_processor, 
    IntelligentQueryProcessor,
    QueryEnhancement,
    QueryIntent
)
from .semantic_cache import (
    get_semantic_cache,
    SemanticQueryCache,
    cached_query
)
from .auto_optimizer import (
    get_auto_optimizer,
    record_performance,
    get_optimal_parameter
)
from .advanced_security import (
    get_security_manager,
    validate_request
)
from .advanced_resilience import (
    get_resilience_manager,
    resilient,
    setup_default_fallbacks
)
from .distributed_intelligence import (
    get_distributed_processor,
    process_query_at_scale
)

# Import base components
from .multi_agent import RetrieverAgent, SummarizerAgent, ClauseExplainerAgent, CitationAgent

logger = logging.getLogger(__name__)


@dataclass 
class IntelligentQueryResult:
    """Result from intelligent query processing."""
    original_query: str
    enhanced_query: str
    intent: QueryIntent
    response: str
    processing_time: float
    cache_hit: bool = False
    security_validated: bool = True
    optimizations_applied: List[str] = field(default_factory=list)
    worker_node: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentMultiAgentSystem:
    """Advanced multi-agent system with comprehensive AI enhancements."""
    
    def __init__(self, pipeline: Any = None):
        self.pipeline = pipeline
        
        # Initialize intelligent subsystems
        self.query_processor = get_query_processor()
        self.semantic_cache = get_semantic_cache()
        self.auto_optimizer = get_auto_optimizer()
        self.security_manager = get_security_manager()
        self.resilience_manager = get_resilience_manager()
        self.distributed_processor = get_distributed_processor()
        
        # Initialize base agents
        self.retriever = RetrieverAgent(pipeline=pipeline)
        self.summarizer = SummarizerAgent()
        self.clause_explainer = ClauseExplainerAgent()
        self.citation_agent = CitationAgent()
        
        # Setup fallback handlers
        setup_default_fallbacks()
        
        # Performance tracking
        self.query_count = 0
        self.total_processing_time = 0.0
        
        logger.info("Intelligent multi-agent system initialized")
    
    @resilient(operation_name="intelligent_query_processing")
    async def process_query_intelligent(
        self, 
        query: str, 
        source_ip: str = "127.0.0.1",
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        priority: float = 1.0,
        use_distributed: bool = True
    ) -> IntelligentQueryResult:
        """Process query using full intelligent pipeline."""
        
        start_time = time.time()
        optimizations_applied = []
        
        try:
            # 1. Security Validation
            security_valid, security_info = await validate_request(query, source_ip, user_id)
            if not security_valid:
                logger.warning(f"Security validation failed: {security_info.get('blocked_reason')}")
                return IntelligentQueryResult(
                    original_query=query,
                    enhanced_query=query,
                    intent=QueryIntent.SEARCH,
                    response=f"Query blocked: {security_info.get('blocked_reason', 'Security validation failed')}",
                    processing_time=time.time() - start_time,
                    security_validated=False
                )
            
            # 2. Query Enhancement
            query_enhancement = await self.query_processor.process_query(query, conversation_id)
            enhanced_query = query_enhancement.enhanced_query
            optimizations_applied.append("query_enhancement")
            
            # 3. Semantic Cache Check
            cache_result = await self.semantic_cache.get(enhanced_query)
            if cache_result is not None:
                cached_response, cache_metadata = cache_result
                logger.info(f"Cache hit for query: {query[:50]}...")
                
                # Still record performance for optimization
                processing_time = time.time() - start_time
                await record_performance(
                    query, processing_time, True,
                    cache_hit=True,
                    similarity_score=cache_metadata.get('similarity_score', 1.0)
                )
                
                return IntelligentQueryResult(
                    original_query=query,
                    enhanced_query=enhanced_query,
                    intent=query_enhancement.intent,
                    response=cached_response,
                    processing_time=processing_time,
                    cache_hit=True,
                    optimizations_applied=optimizations_applied + ["semantic_cache"],
                    metadata=cache_metadata
                )
            
            # 4. Get Optimal Parameters from Auto-Optimizer
            top_k = get_optimal_parameter('top_k_results', 3)
            timeout = get_optimal_parameter('timeout_seconds', 30)
            optimizations_applied.append("parameter_optimization")
            
            # 5. Process Query (Distributed or Local)
            if use_distributed and query_enhancement.intent not in [QueryIntent.SEARCH]:  # Complex queries go distributed
                response = await self._process_distributed(enhanced_query, priority)
                optimizations_applied.append("distributed_processing")
            else:
                response = await self._process_local(enhanced_query, query_enhancement.intent, top_k)
                optimizations_applied.append("local_processing")
            
            # 6. Cache the Result
            processing_time = time.time() - start_time
            await self.semantic_cache.put(
                enhanced_query, 
                response, 
                processing_time,
                intent=query_enhancement.intent.value,
                confidence=query_enhancement.confidence
            )
            
            # 7. Record Performance for Optimization
            await record_performance(
                query, processing_time, True,
                cache_hit=False,
                query_intent=query_enhancement.intent.value,
                optimizations_applied=len(optimizations_applied)
            )
            
            # 8. Update Statistics
            self.query_count += 1
            self.total_processing_time += processing_time
            
            return IntelligentQueryResult(
                original_query=query,
                enhanced_query=enhanced_query,
                intent=query_enhancement.intent,
                response=response,
                processing_time=processing_time,
                optimizations_applied=optimizations_applied,
                metadata={
                    'security_info': security_info,
                    'legal_terms_found': query_enhancement.legal_terms,
                    'synonyms_added': query_enhancement.synonyms_added
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in intelligent query processing: {e}")
            
            # Record failure for optimization
            await record_performance(query, processing_time, False, error=str(e))
            
            # Return error result
            return IntelligentQueryResult(
                original_query=query,
                enhanced_query=query,
                intent=QueryIntent.SEARCH,
                response=f"Processing error: {str(e)}",
                processing_time=processing_time,
                optimizations_applied=optimizations_applied
            )
    
    async def _process_distributed(self, query: str, priority: float) -> str:
        """Process query using distributed system."""
        response_chunks = []
        async for chunk in process_query_at_scale(query, priority):
            response_chunks.append(chunk)
        
        return " ".join(response_chunks)
    
    async def _process_local(self, query: str, intent: QueryIntent, top_k: int) -> str:
        """Process query using local agents."""
        # Use the intelligent routing based on intent
        if intent == QueryIntent.SEARCH:
            # Direct retrieval for search queries
            retrieved = await self.retriever.run(query)
            return retrieved
        
        elif intent == QueryIntent.SUMMARIZE:
            # Retrieve then summarize
            retrieved = await self.retriever.run(query)
            summary = await self.summarizer.run(retrieved)
            return summary
        
        elif intent in [QueryIntent.EXPLAIN, QueryIntent.ANALYZE, QueryIntent.DEFINITION]:
            # Full pipeline: retrieve, summarize, explain
            retrieved = await self.retriever.run(query)
            summary = await self.summarizer.run(retrieved)
            explanation = await self.clause_explainer.run(summary)
            return explanation
        
        else:
            # Default: full pipeline
            retrieved = await self.retriever.run(query)
            summary = await self.summarizer.run(retrieved)
            explanation = await self.clause_explainer.run(summary)
            return explanation
    
    async def stream_response_with_citations(
        self, 
        query: str, 
        source_ip: str = "127.0.0.1",
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response with citations using intelligent processing."""
        
        # Process query intelligently
        result = await self.process_query_intelligent(query, source_ip, **kwargs)
        
        # Yield main response
        yield result.response
        
        # Add citations if we have pipeline access
        if self.pipeline and result.security_validated:
            try:
                # Get documents for citations
                search_results = self.pipeline.search(result.enhanced_query, top_k=3)
                docs = [doc for doc, _ in search_results]
                
                # Generate citations
                citations = []
                for doc in docs:
                    ref = doc.metadata.get("path", doc.id) if hasattr(doc, 'metadata') else doc.id
                    snippet = doc.text[:100] + "..." if hasattr(doc, 'text') and len(doc.text) > 100 else str(doc)[:100]
                    citations.append(f'{doc.id}: {ref} - "{snippet}"')
                
                if citations:
                    yield "\n\nCitations:\n" + "\n".join(citations)
                    
            except Exception as e:
                logger.error(f"Error generating citations: {e}")
                yield f"\n\nNote: Citations unavailable due to system error."
    
    def get_system_intelligence_report(self) -> Dict[str, Any]:
        """Get comprehensive intelligence system report."""
        return {
            'query_processing': self.query_processor.analyze_query_patterns(),
            'semantic_cache': self.semantic_cache.get_stats(),
            'optimization': self.auto_optimizer.get_optimization_report(),
            'security': self.security_manager.get_security_report(),
            'resilience': self.resilience_manager.get_resilience_report(),
            'distributed': self.distributed_processor.get_system_status(),
            'agent_system': {
                'total_queries_processed': self.query_count,
                'average_processing_time': self.total_processing_time / max(self.query_count, 1),
                'pipeline_connected': self.pipeline is not None
            }
        }
    
    async def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Get intelligent query suggestions."""
        return self.query_processor.get_query_suggestions(partial_query)
    
    async def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity and provide insights."""
        enhancement = await self.query_processor.process_query(query)
        
        return {
            'original_query': query,
            'enhanced_query': enhancement.enhanced_query,
            'intent': enhancement.intent.value,
            'confidence': enhancement.confidence,
            'legal_terms': enhancement.legal_terms,
            'synonyms_added': enhancement.synonyms_added,
            'context_expansions': enhancement.context_expansions,
            'complexity_score': len(enhancement.legal_terms) + len(enhancement.synonyms_added)
        }
    
    def update_optimization_strategy(self, strategy: str) -> bool:
        """Update auto-optimization strategy."""
        try:
            from .auto_optimizer import OptimizationStrategy
            if hasattr(OptimizationStrategy, strategy.upper()):
                new_strategy = getattr(OptimizationStrategy, strategy.upper())
                self.auto_optimizer.strategy = new_strategy
                logger.info(f"Updated optimization strategy to: {strategy}")
                return True
        except Exception as e:
            logger.error(f"Failed to update optimization strategy: {e}")
        
        return False


# Global intelligent system instance
_global_intelligent_system = None

def get_intelligent_multi_agent_system(pipeline: Any = None) -> IntelligentMultiAgentSystem:
    """Get global intelligent multi-agent system."""
    global _global_intelligent_system
    if _global_intelligent_system is None:
        _global_intelligent_system = IntelligentMultiAgentSystem(pipeline)
    elif pipeline and _global_intelligent_system.pipeline is None:
        _global_intelligent_system.pipeline = pipeline
    
    return _global_intelligent_system


async def process_legal_query(
    query: str,
    source_ip: str = "127.0.0.1", 
    **kwargs
) -> IntelligentQueryResult:
    """Convenience function for intelligent legal query processing."""
    system = get_intelligent_multi_agent_system()
    return await system.process_query_intelligent(query, source_ip, **kwargs)


async def stream_legal_query(query: str, **kwargs) -> AsyncIterator[str]:
    """Convenience function for streaming legal query responses."""
    system = get_intelligent_multi_agent_system()
    async for chunk in system.stream_response_with_citations(query, **kwargs):
        yield chunk


def get_system_report() -> Dict[str, Any]:
    """Get comprehensive system intelligence report."""
    system = get_intelligent_multi_agent_system()
    return system.get_system_intelligence_report()