"""Basic multi-agent architecture with simple routing logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Iterable, Iterator, Protocol, Any

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
    """Agent that retrieves relevant legal documents based on queries."""
    
    pipeline: Any = None  # LegalDocumentPipeline to avoid circular import
    top_k: int = 3
    
    async def run(self, query: str) -> str:
        """Retrieve relevant text for a query."""
        logger.debug("RetrieverAgent running with query: %s", query)
        
        if self.pipeline is None:
            logger.warning("No pipeline configured for RetrieverAgent, returning stub response")
            return f"retrieved: {query}"
        
        # Search for relevant documents
        results = self.pipeline.search(query, top_k=self.top_k)
        
        # Filter out results with very low relevance scores
        relevant_results = [(doc, score) for doc, score in results if score > 0.01]
        
        if not relevant_results:
            logger.info("No relevant documents found for query: %s", query)
            return f"No relevant documents found for: {query}"
        
        results = relevant_results
        
        # Combine retrieved documents into a single text
        retrieved_texts = []
        for doc, score in results:
            # Include document metadata for context
            source = doc.metadata.get("path", doc.id)
            retrieved_texts.append(f"[Source: {source}, Relevance: {score:.3f}]\n{doc.text[:500]}...")
        
        combined_text = "\n\n".join(retrieved_texts)
        logger.debug("Retrieved %d documents with total length %d", len(results), len(combined_text))
        
        return combined_text


@dataclass
class SummarizerAgent:
    """Agent that summarizes legal documents and extracts key information."""
    
    max_length: int = 200
    
    async def run(self, text: str) -> str:
        """Return a summary of the given text."""
        logger.debug("SummarizerAgent summarizing text length: %d", len(text))
        
        if not text or text.startswith("No relevant documents"):
            return text
        
        # Extract key legal concepts and create a summary
        lines = text.split('\n')
        
        # Find important legal terms and concepts
        important_sections = []
        current_section = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for section headers or source markers
            if line.startswith('[Source:') or any(keyword in line.lower() for keyword in 
                ['section', 'clause', 'article', 'provision', 'whereas', 'therefore']):
                if current_section:
                    important_sections.append(' '.join(current_section))
                    current_section = []
                if not line.startswith('[Source:'):
                    current_section.append(line)
            else:
                current_section.append(line)
        
        if current_section:
            important_sections.append(' '.join(current_section))
        
        # Create summary by taking key sections and limiting length
        if important_sections:
            summary = '. '.join(important_sections[:3])  # Take top 3 sections
            if len(summary) > self.max_length:
                summary = summary[:self.max_length] + '...'
        else:
            # Fallback: take first portion of text
            clean_text = text.replace('[Source:', '').replace(']', '')
            summary = clean_text[:self.max_length] + '...' if len(clean_text) > self.max_length else clean_text
        
        logger.debug("Generated summary of length: %d", len(summary))
        return summary


@dataclass
class ClauseExplainerAgent:
    """Agent that provides detailed legal explanations of clauses and concepts."""
    
    async def run(self, summary: str) -> str:
        """Return an explanation of the summarized clause."""
        logger.debug("ClauseExplainerAgent explaining summary: %s", summary)
        
        if not summary or summary.startswith("No relevant documents"):
            return "No explanation available - no relevant documents found."
        
        # Analyze the summary for legal concepts that need explanation
        legal_terms = self._identify_legal_terms(summary)
        
        # Build explanation based on identified terms and context
        explanations = []
        
        # Add context about the document type or legal area
        if any(term in summary.lower() for term in ['contract', 'agreement', 'terms']):
            explanations.append("This appears to be contractual language.")
        elif any(term in summary.lower() for term in ['statute', 'law', 'regulation']):
            explanations.append("This appears to be statutory or regulatory text.")
        elif any(term in summary.lower() for term in ['case', 'court', 'judgment']):
            explanations.append("This appears to be from a legal case or court decision.")
        
        # Explain key legal terms found
        if legal_terms:
            term_explanations = []
            for term in legal_terms[:3]:  # Limit to top 3 terms
                explanation = self._explain_term(term)
                if explanation:
                    term_explanations.append(f"'{term}': {explanation}")
            
            if term_explanations:
                explanations.append("Key legal terms: " + "; ".join(term_explanations))
        
        # Add practical implications
        implications = self._analyze_implications(summary)
        if implications:
            explanations.append(f"Practical implications: {implications}")
        
        # Combine all explanations
        if explanations:
            full_explanation = " ".join(explanations)
        else:
            full_explanation = f"Legal analysis: {summary[:100]}... (This clause requires further legal interpretation based on specific context.)"
        
        logger.debug("Generated explanation of length: %d", len(full_explanation))
        return full_explanation
    
    def _identify_legal_terms(self, text: str) -> list[str]:
        """Identify important legal terms in the text."""
        legal_keywords = [
            'liability', 'breach', 'damages', 'indemnify', 'warranty', 'covenant',
            'consideration', 'force majeure', 'jurisdiction', 'arbitration',
            'intellectual property', 'confidential', 'termination', 'notice',
            'compliance', 'governing law', 'severability', 'amendment'
        ]
        
        found_terms = []
        text_lower = text.lower()
        for term in legal_keywords:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _explain_term(self, term: str) -> str:
        """Provide basic explanation for common legal terms."""
        explanations = {
            'liability': 'legal responsibility for damages or obligations',
            'breach': 'failure to fulfill contractual obligations',
            'damages': 'monetary compensation for losses or harm',
            'indemnify': 'protect against or compensate for losses',
            'warranty': 'guarantee about the quality or condition of something',
            'covenant': 'formal promise or agreement to do or not do something',
            'consideration': 'something of value exchanged in a contract',
            'force majeure': 'unforeseeable circumstances preventing contract fulfillment',
            'jurisdiction': 'authority of a court to hear and decide cases',
            'arbitration': 'dispute resolution outside of court',
            'intellectual property': 'creations of the mind (patents, trademarks, copyrights)',
            'confidential': 'information that must be kept secret',
            'termination': 'ending of a contract or agreement',
            'notice': 'formal communication required by law or contract',
            'compliance': 'conforming to rules, regulations, or standards',
            'governing law': 'jurisdiction whose laws will be applied to interpret the contract',
            'severability': 'if one part is invalid, the rest remains valid',
            'amendment': 'formal change or addition to a legal document'
        }
        
        return explanations.get(term.lower(), '')
    
    def _analyze_implications(self, text: str) -> str:
        """Analyze practical implications of the legal text."""
        text_lower = text.lower()
        
        if 'liability' in text_lower and 'limit' in text_lower:
            return "This may limit legal responsibility for certain damages."
        elif 'terminate' in text_lower or 'termination' in text_lower:
            return "This relates to conditions under which agreements may be ended."
        elif 'payment' in text_lower or 'fee' in text_lower:
            return "This involves financial obligations or compensation."
        elif 'confidential' in text_lower:
            return "This establishes obligations to protect sensitive information."
        elif 'warranty' in text_lower or 'guarantee' in text_lower:
            return "This creates assurances about quality or performance."
        else:
            return "This establishes legal rights, obligations, or procedures."


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
    """Intelligent router to determine which agents to invoke based on query analysis."""

    explain_keywords: tuple[str, ...] = ("explain", "meaning", "interpret", "what does", "clarify", "define")
    summary_keywords: tuple[str, ...] = ("summarize", "summary", "overview", "brief", "key points")
    search_keywords: tuple[str, ...] = ("find", "search", "locate", "show me", "get")
    
    def decide(self, query: str) -> bool:
        """Return ``True`` if an explanation is requested."""
        lowered = query.lower()
        return any(keyword in lowered for keyword in self.explain_keywords)
    
    def needs_summary_only(self, query: str) -> bool:
        """Return ``True`` if only a summary is requested (no explanation)."""
        lowered = query.lower()
        has_summary_keywords = any(keyword in lowered for keyword in self.summary_keywords)
        has_explain_keywords = any(keyword in lowered for keyword in self.explain_keywords)
        
        # If summary keywords present but no explain keywords, just summarize
        return has_summary_keywords and not has_explain_keywords
    
    def is_search_query(self, query: str) -> bool:
        """Return ``True`` if this is primarily a search/retrieval query."""
        lowered = query.lower()
        return any(keyword in lowered for keyword in self.search_keywords)
    
    def analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity and return routing decision."""
        if self.needs_summary_only(query):
            return "summary"
        elif self.decide(query):
            return "explain"
        elif self.is_search_query(query):
            return "search"
        else:
            return "default"


@dataclass
class MultiAgentGraph:
    """Intelligent pipeline connecting specialized legal RAG agents."""

    retriever: Agent = field(default_factory=RetrieverAgent)
    summarizer: Agent = field(default_factory=SummarizerAgent)
    clause_explainer: Agent = field(default_factory=ClauseExplainerAgent)
    router: RouterAgent = field(default_factory=RouterAgent)
    max_depth: int = 1
    pipeline: Any = None  # LegalDocumentPipeline to avoid circular import
    
    def __post_init__(self):
        """Configure agents with pipeline reference after initialization."""
        if self.pipeline and hasattr(self.retriever, 'pipeline'):
            self.retriever.pipeline = self.pipeline

    async def run(self, query: str) -> str:
        """Run the pipeline for the given query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        logger.info("Running agent graph for query: %s", query)
        return await self._run_recursive(query, depth=0)

    async def _run_recursive(self, query: str, depth: int) -> str:
        """Execute agents intelligently based on query analysis up to ``max_depth``."""
        
        # Analyze query to determine optimal routing
        routing_decision = self.router.analyze_query_complexity(query)
        logger.debug("Depth %d: routing decision '%s' for query: %s", depth, routing_decision, query)
        
        # Always start with retrieval
        logger.debug("Depth %d: retrieving", depth)
        retrieved = await self.retriever.run(query)
        
        # Check if we found any relevant documents
        if retrieved.startswith("No relevant documents"):
            logger.info("No relevant documents found, returning retrieval result")
            return retrieved
        
        # Handle different routing decisions
        if routing_decision == "search":
            # For search queries, return retrieved documents with minimal processing
            logger.debug("Depth %d: search query, returning retrieved results", depth)
            return retrieved
        
        # Always summarize unless it's a pure search query
        logger.debug("Depth %d: summarizing", depth)
        summary = await self.summarizer.run(retrieved)
        
        if routing_decision == "summary":
            # For summary-only queries, return just the summary
            logger.debug("Depth %d: summary-only query, returning summary", depth)
            return summary
        
        # For explain queries or complex queries, add explanation
        if routing_decision == "explain" or (routing_decision == "default" and self.router.decide(query)):
            if depth < self.max_depth:
                logger.debug("Depth %d: explaining", depth)
                explanation = await self.clause_explainer.run(summary)
                
                # Check if we should recurse further
                if depth + 1 < self.max_depth and self.router.decide(explanation):
                    logger.debug("Depth %d: recursing for deeper explanation", depth)
                    return await self._run_recursive(explanation, depth + 1)
                
                return explanation
        
        # Default: return summary
        return summary

    async def run_with_citations(
        self, query: str, pipeline: Any, top_k: int = 3  # pipeline: LegalDocumentPipeline
    ) -> AsyncIterator[str]:
        """Run the pipeline and yield an answer with citations."""
        logger.info("Running agent graph with citations for query: %s", query)
        
        # Set pipeline for this run if not already set
        if self.pipeline is None:
            self.pipeline = pipeline
            if hasattr(self.retriever, 'pipeline'):
                self.retriever.pipeline = pipeline
        
        answer = await self.run(query)
        results = pipeline.search(query, top_k=top_k)
        docs = [doc for doc, _ in results]
        citation_agent = CitationAgent()
        for chunk in citation_agent.stream(answer, docs, query):
            yield chunk

    def run_with_citations_sync(
        self, query: str, pipeline: Any, top_k: int = 3  # pipeline: LegalDocumentPipeline
    ) -> Iterator[str]:
        """Synchronous wrapper around :meth:`run_with_citations`."""
        import asyncio

        async def gather() -> list[str]:
            return [
                chunk async for chunk in self.run_with_citations(query, pipeline, top_k)
            ]

        for chunk in asyncio.run(gather()):
            yield chunk
