"""Context-aware reasoning across related legal documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator

import logging

from .document_pipeline import LegalDocumentPipeline
from .multi_agent import CitationAgent, MultiAgentGraph


logger = logging.getLogger(__name__)


@dataclass
class ContextAwareReasoner:
    """Perform multi-hop reasoning over legal documents."""

    pipeline: LegalDocumentPipeline = field(default_factory=LegalDocumentPipeline)
    agent_graph: MultiAgentGraph = field(default_factory=MultiAgentGraph)

    async def reason(self, query: str, hops: int = 3) -> str:
        """Return an explanation that considers related documents."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        logger.info("Reasoning about query: %s", query)
        results = self.pipeline.search(query, top_k=hops)
        if not results:
            logger.debug("No documents found; using agent graph directly")
            return await self.agent_graph.run(query)
        context = " \n".join(doc.text for doc, _ in results)
        logger.debug("Passing context of %d documents to agent graph", len(results))
        return await self.agent_graph.run(context)

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
        import asyncio

        async def gather() -> list[str]:
            return [
                chunk async for chunk in self.reason_with_citations(query, hops=hops)
            ]

        for chunk in asyncio.run(gather()):
            yield chunk
