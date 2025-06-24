"""Basic multi-agent architecture with simple routing logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Iterator, Protocol

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from .document_pipeline import LegalDocumentPipeline
    from .models import LegalDocument


class Agent(Protocol):
    """Protocol for all agents."""

    def run(self, text: str) -> str:
        """Process input text and return output text."""


@dataclass
class RetrieverAgent:
    """Stub retriever agent."""

    def run(self, query: str) -> str:
        """Retrieve relevant text for a query."""
        return f"retrieved: {query}"


@dataclass
class SummarizerAgent:
    """Stub summarizer agent."""

    def run(self, text: str) -> str:
        """Return a summary of the given text."""
        return f"summary of {text}"


@dataclass
class ClauseExplainerAgent:
    """Stub clause explainer agent."""

    def run(self, summary: str) -> str:
        """Return an explanation of the summarized clause."""
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

    def decide(self, query: str) -> bool:
        """Return ``True`` if an explanation is requested."""
        lowered = query.lower()
        return any(
            keyword in lowered for keyword in ("explain", "meaning", "interpret")
        )


@dataclass
class MultiAgentGraph:
    """Simple pipeline connecting individual agents."""

    retriever: Agent = field(default_factory=RetrieverAgent)
    summarizer: Agent = field(default_factory=SummarizerAgent)
    clause_explainer: Agent = field(default_factory=ClauseExplainerAgent)
    router: RouterAgent = field(default_factory=RouterAgent)
    max_depth: int = 1

    def run(self, query: str) -> str:
        """Run the pipeline for the given query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        return self._run_recursive(query, depth=0)

    def _run_recursive(self, query: str, depth: int) -> str:
        """Execute agents recursively up to ``max_depth``."""
        retrieved = self.retriever.run(query)
        summary = self.summarizer.run(retrieved)
        if self.router.decide(query) and depth < self.max_depth:
            explanation = self.clause_explainer.run(summary)
            if depth + 1 < self.max_depth and self.router.decide(explanation):
                return self._run_recursive(explanation, depth + 1)
            return explanation
        return summary

    def run_with_citations(
        self, query: str, pipeline: "LegalDocumentPipeline", top_k: int = 3
    ) -> Iterator[str]:
        """Run the pipeline and yield an answer with citations."""
        answer = self.run(query)
        results = pipeline.search(query, top_k=top_k)
        docs = [doc for doc, _ in results]
        citation_agent = CitationAgent()
        yield from citation_agent.stream(answer, docs, query)
