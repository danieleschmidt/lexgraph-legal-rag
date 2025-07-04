"""Basic multi-agent architecture with simple routing logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Iterable, Iterator, Protocol

import logging

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from .document_pipeline import LegalDocumentPipeline
    from .models import LegalDocument


logger = logging.getLogger(__name__)


class Agent(Protocol):
    """Protocol for all agents."""

    async def run(self, text: str) -> str:
        """Process input text and return output text."""


@dataclass
class RetrieverAgent:
    """Stub retriever agent."""

    async def run(self, query: str) -> str:
        """Retrieve relevant text for a query."""
        logger.debug("RetrieverAgent running with query: %s", query)
        return f"retrieved: {query}"


@dataclass
class SummarizerAgent:
    """Stub summarizer agent."""

    async def run(self, text: str) -> str:
        """Return a summary of the given text."""
        logger.debug("SummarizerAgent summarizing text: %s", text)
        return f"summary of {text}"


@dataclass
class ClauseExplainerAgent:
    """Stub clause explainer agent."""

    async def run(self, summary: str) -> str:
        """Return an explanation of the summarized clause."""
        logger.debug("ClauseExplainerAgent explaining summary: %s", summary)
        return f"explanation of {summary}"


@dataclass
class CitationAgent:
    """Return answers annotated with citations."""

    window: int = 40

    def _snippet(self, doc: "LegalDocument", query: str) -> str:
        text_lower = doc.text.lower()
        idx = text_lower.find(query.lower())
        if idx == -1:
            start = 0
        else:
            start = max(idx - self.window, 0)
        end = min(len(doc.text), start + self.window * 2)
        return doc.text[start:end].replace("\n", " ")

    def stream(
        self, answer: str, docs: Iterable["LegalDocument"], query: str
    ) -> Iterator[str]:
        """Yield the answer followed by citation references."""
        yield answer
        citations = []
        for doc in docs:
            ref = doc.metadata.get("path", doc.id)
            snippet = self._snippet(doc, query)
            citations.append(f'{doc.id}: {ref} - "{snippet}..."')
        if citations:
            yield "Citations:\n" + "\n".join(citations)


@dataclass
class RouterAgent:
    """Basic router to determine which agents to invoke."""

    explain_keywords: tuple[str, ...] = ("explain", "meaning", "interpret")

    def decide(self, query: str) -> bool:
        """Return ``True`` if an explanation is requested."""
        lowered = query.lower()
        return any(keyword in lowered for keyword in self.explain_keywords)


@dataclass
class MultiAgentGraph:
    """Simple pipeline connecting individual agents."""

    retriever: Agent = field(default_factory=RetrieverAgent)
    summarizer: Agent = field(default_factory=SummarizerAgent)
    clause_explainer: Agent = field(default_factory=ClauseExplainerAgent)
    router: RouterAgent = field(default_factory=RouterAgent)
    max_depth: int = 1

    async def run(self, query: str) -> str:
        """Run the pipeline for the given query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        logger.info("Running agent graph for query: %s", query)
        return await self._run_recursive(query, depth=0)

    async def _run_recursive(self, query: str, depth: int) -> str:
        """Execute agents recursively up to ``max_depth``."""
        logger.debug("Depth %d: retrieving", depth)
        retrieved = await self.retriever.run(query)
        logger.debug("Depth %d: summarizing", depth)
        summary = await self.summarizer.run(retrieved)
        if self.router.decide(query) and depth < self.max_depth:
            logger.debug("Depth %d: explaining", depth)
            explanation = await self.clause_explainer.run(summary)
            if depth + 1 < self.max_depth and self.router.decide(explanation):
                return await self._run_recursive(explanation, depth + 1)
            return explanation
        return summary

    async def run_with_citations(
        self, query: str, pipeline: "LegalDocumentPipeline", top_k: int = 3
    ) -> AsyncIterator[str]:
        """Run the pipeline and yield an answer with citations."""
        logger.info("Running agent graph with citations for query: %s", query)
        answer = await self.run(query)
        results = pipeline.search(query, top_k=top_k)
        docs = [doc for doc, _ in results]
        citation_agent = CitationAgent()
        for chunk in citation_agent.stream(answer, docs, query):
            yield chunk

    def run_with_citations_sync(
        self, query: str, pipeline: "LegalDocumentPipeline", top_k: int = 3
    ) -> Iterator[str]:
        """Synchronous wrapper around :meth:`run_with_citations`."""
        import asyncio

        async def gather() -> list[str]:
            return [
                chunk async for chunk in self.run_with_citations(query, pipeline, top_k)
            ]

        for chunk in asyncio.run(gather()):
            yield chunk
