"""Context-aware reasoning across related legal documents with advanced optimization."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import AsyncIterator
from typing import Iterator

from .document_pipeline import LegalDocumentPipeline
from .multi_agent import CitationAgent
from .multi_agent import MultiAgentGraph
from .performance_optimization import AdaptiveCache
from .performance_optimization import performance_monitor
from .resilience import CircuitBreakerConfig
from .resilience import ResilientOperation
from .resilience import RetryConfig
from .resilience import get_circuit_breaker
from .scalable_index import get_index_pool
from .validation import QueryValidator
from .validation import SecurityLevel


if TYPE_CHECKING:
    from .models import LegalDocument


logger = logging.getLogger(__name__)


@dataclass
class ContextAwareReasoner:
    """High-performance multi-hop reasoning over legal documents with advanced optimizations."""

    pipeline: LegalDocumentPipeline = field(default_factory=LegalDocumentPipeline)
    agent_graph: MultiAgentGraph = field(default_factory=MultiAgentGraph)
    use_scalable_index: bool = True
    enable_caching: bool = True
    query_validator: QueryValidator = field(
        default_factory=lambda: QueryValidator(SecurityLevel.STRICT)
    )

    def __post_init__(self):
        """Initialize optimized components."""
        if self.enable_caching:
            self.reasoning_cache = AdaptiveCache(
                initial_size=500,
                max_size=5000,
                ttl_seconds=7200,  # 2 hours
                hit_rate_threshold=0.85,
            )

        # Configure agent graph with pipeline reference
        if self.agent_graph.pipeline is None:
            self.agent_graph.pipeline = self.pipeline

    @performance_monitor("context_reasoning")
    async def reason(self, query: str, hops: int = 3) -> str:
        """Return an explanation that considers related documents with advanced optimization."""
        # Input validation
        validation_result = self.query_validator.validate_query(query)
        if not validation_result.is_valid:
            error_msg = f"Invalid query: {'; '.join(validation_result.errors)}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        # Use sanitized query
        sanitized_query = validation_result.sanitized_input
        logger.info("Reasoning about query: %s", sanitized_query[:100])

        # Check cache first
        if self.enable_caching:
            cache_key = f"reason:{sanitized_query}:{hops}"
            cached_result = self.reasoning_cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit for reasoning query")
                return cached_result

        # Use resilient operation for reasoning
        resilient_reasoning = ResilientOperation(
            circuit_breaker=get_circuit_breaker(
                "context_reasoning",
                CircuitBreakerConfig(
                    failure_threshold=5, recovery_timeout=60.0, name="context_reasoning"
                ),
            ),
            retry_config=RetryConfig(max_attempts=2, base_delay=1.0),
            operation_timeout=30.0,
        )

        try:
            result = await resilient_reasoning.execute(
                self._execute_reasoning, sanitized_query, hops
            )

            # Cache successful results
            if self.enable_caching and result:
                cache_key = f"reason:{sanitized_query}:{hops}"
                self.reasoning_cache.put(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Reasoning failed for query '{sanitized_query[:50]}': {e}")
            # Return graceful fallback
            return "I apologize, but I encountered an error while analyzing your query about legal matters. Please try rephrasing your question or contact support if the issue persists."

    async def _execute_reasoning(self, query: str, hops: int) -> str:
        """Execute the core reasoning logic."""
        # Get search results using optimized index pool or pipeline
        if self.use_scalable_index:
            index_pool = get_index_pool()
            await index_pool.initialize(
                self.pipeline.index._docs
                if hasattr(self.pipeline.index, "_docs")
                else None
            )
            results = await index_pool.search(query, top_k=hops)
        else:
            results = self.pipeline.search(query, top_k=hops)

        if not results:
            logger.debug("No documents found; using agent graph directly")
            return await self.agent_graph.run(query)

        # Intelligently build context from results
        context = self._build_optimized_context(results, query)
        logger.debug("Passing context of %d documents to agent graph", len(results))

        return await self.agent_graph.run(context)

    def _build_optimized_context(
        self, results: list[tuple[LegalDocument, float]], query: str
    ) -> str:
        """Build optimized context from search results."""
        context_parts = []
        total_length = 0
        max_context_length = 4000  # Reasonable limit for context

        # Sort results by relevance score
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        for doc, score in sorted_results:
            # Create informative context entry
            source = doc.metadata.get("path", doc.id) if doc.metadata else doc.id

            # Extract most relevant portion of document relative to query
            relevant_text = self._extract_relevant_text(doc.text, query, max_length=800)

            context_entry = (
                f"[Source: {source}, Relevance: {score:.3f}]\n{relevant_text}"
            )

            # Check if adding this would exceed limit
            if total_length + len(context_entry) > max_context_length:
                break

            context_parts.append(context_entry)
            total_length += len(context_entry)

        return "\n\n".join(context_parts)

    def _extract_relevant_text(
        self, text: str, query: str, max_length: int = 800
    ) -> str:
        """Extract most relevant portion of text based on query."""
        if len(text) <= max_length:
            return text

        query_words = set(query.lower().split())
        sentences = text.split(".")

        # Score sentences based on query word overlap
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            sentence_scores.append((overlap, i, sentence.strip()))

        # Sort by score and select best sentences that fit within limit
        sentence_scores.sort(reverse=True, key=lambda x: x[0])

        selected_text = ""
        for score, _idx, sentence in sentence_scores:
            if score == 0:  # No more relevant sentences
                break

            if len(selected_text) + len(sentence) + 2 <= max_length:  # +2 for '. '
                if selected_text:
                    selected_text += ". "
                selected_text += sentence
            else:
                break

        # If no sentences were selected, take the beginning
        if not selected_text:
            selected_text = text[: max_length - 3] + "..."

        return selected_text

    async def reason_with_citations(
        self, query: str, hops: int = 3
    ) -> AsyncIterator[str]:
        """Yield reasoning results with citations to source documents."""
        results = self.pipeline.search(query, top_k=hops)
        docs = [doc for doc, _ in results]
        answer = await self.reason(query, hops=hops)
        citation_agent = CitationAgent()
        for chunk in citation_agent.stream(answer, docs, query):
            yield chunk

    def reason_with_citations_sync(self, query: str, hops: int = 3) -> Iterator[str]:
        """Synchronous wrapper around :meth:`reason_with_citations`."""

        async def gather() -> list[str]:
            return [
                chunk async for chunk in self.reason_with_citations(query, hops=hops)
            ]

        yield from asyncio.run(gather())
