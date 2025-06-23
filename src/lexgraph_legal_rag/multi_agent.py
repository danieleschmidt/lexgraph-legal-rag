"""Simple multi-agent architecture with rudimentary routing logic."""

from __future__ import annotations


class RetrieverAgent:
    """Stub retriever agent."""

    def run(self, query: str) -> str:
        """Retrieve relevant text for a query."""
        return f"retrieved: {query}"


class SummarizerAgent:
    """Stub summarizer agent."""

    def run(self, text: str) -> str:
        """Return a summary of the given text."""
        return f"summary of {text}"


class ClauseExplainerAgent:
    """Stub clause explainer agent."""

    def run(self, summary: str) -> str:
        """Return an explanation of the summarized clause."""
        return f"explanation of {summary}"


class RouterAgent:
    """Basic router to determine which agents to invoke."""

    def decide(self, query: str) -> bool:
        """Return ``True`` if an explanation is requested."""
        return "explain" in query.lower()


class MultiAgentGraph:
    """Minimal pipeline connecting individual agents."""

    def __init__(self) -> None:
        self.retriever = RetrieverAgent()
        self.summarizer = SummarizerAgent()
        self.clause_explainer = ClauseExplainerAgent()
        self.router = RouterAgent()

    def run(self, query: str) -> str:
        """Run the pipeline for the given query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        retrieved = self.retriever.run(query)
        summary = self.summarizer.run(retrieved)
        if self.router.decide(query):
            return self.clause_explainer.run(summary)
        return summary
