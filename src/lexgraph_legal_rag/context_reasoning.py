"""Context-aware reasoning across related legal documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .document_pipeline import LegalDocumentPipeline
from .multi_agent import CitationAgent, MultiAgentGraph


@dataclass
class ContextAwareReasoner:
    """Perform multi-hop reasoning over legal documents."""

    pipeline: LegalDocumentPipeline = field(default_factory=LegalDocumentPipeline)
    agent_graph: MultiAgentGraph = field(default_factory=MultiAgentGraph)

    def reason(self, query: str, hops: int = 3) -> str:
        """Return an explanation that considers related documents."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        results = self.pipeline.search(query, top_k=hops)
        if not results:
            return self.agent_graph.run(query)
        context = " \n".join(doc.text for doc, _ in results)
        return self.agent_graph.run(context)

    def reason_with_citations(self, query: str, hops: int = 3) -> Iterator[str]:
        """Yield reasoning results with citations to source documents."""
        results = self.pipeline.search(query, top_k=hops)
        docs = [doc for doc, _ in results]
        answer = self.reason(query, hops=hops)
        citation_agent = CitationAgent()
        yield from citation_agent.stream(answer, docs, query)
