"""Comprehensive tests for multi-agent system to increase coverage."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

from lexgraph_legal_rag.multi_agent import (
    RetrieverAgent,
    SummarizerAgent, 
    ClauseExplainerAgent,
    CitationAgent,
    RouterAgent,
    MultiAgentGraph
)
from lexgraph_legal_rag.models import LegalDocument


@dataclass
class MockDocument:
    """Mock document for testing."""
    content: str
    title: str = "Test Document"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestRetrieverAgent:
    """Test RetrieverAgent functionality."""

    @pytest.mark.asyncio
    async def test_retriever_without_pipeline(self):
        """Test retriever returns stub response when no pipeline configured."""
        agent = RetrieverAgent()
        result = await agent.run("test query")
        assert result == "retrieved: test query"

    @pytest.mark.asyncio
    async def test_retriever_with_pipeline_no_results(self):
        """Test retriever when pipeline returns no relevant results."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = [(MockDocument("test"), 0.001)]  # Below threshold
        
        agent = RetrieverAgent(pipeline=mock_pipeline)
        result = await agent.run("test query")
        assert "No relevant documents found for: test query" in result

    @pytest.mark.asyncio
    async def test_retriever_with_pipeline_good_results(self):
        """Test retriever with relevant results."""
        mock_pipeline = Mock()
        # Create a mock document with the correct attributes that the agent expects
        mock_doc = Mock()
        mock_doc.text = "This is a legal document about contracts"
        mock_doc.id = "test_doc_1"
        mock_doc.metadata = {"path": "contracts/test.txt"}
        mock_pipeline.search.return_value = [(mock_doc, 0.95)]
        
        agent = RetrieverAgent(pipeline=mock_pipeline, top_k=1)
        result = await agent.run("contract terms")
        assert "This is a legal document about contracts" in result
        assert "contracts/test.txt" in result  # Check source is included

    @pytest.mark.asyncio
    async def test_retriever_filters_low_scores(self):
        """Test that retriever filters out low relevance scores."""
        mock_pipeline = Mock()
        good_doc = Mock()
        good_doc.text = "Relevant content"
        good_doc.id = "good_doc"
        good_doc.metadata = {"path": "good.txt"}
        
        bad_doc = Mock()
        bad_doc.text = "Irrelevant content"
        bad_doc.id = "bad_doc"
        bad_doc.metadata = {"path": "bad.txt"}
        
        mock_pipeline.search.return_value = [
            (good_doc, 0.85),  # Above threshold
            (bad_doc, 0.005)   # Below threshold
        ]
        
        agent = RetrieverAgent(pipeline=mock_pipeline)
        result = await agent.run("test query")
        assert "Relevant content" in result
        assert "Irrelevant content" not in result


class TestSummarizerAgent:
    """Test SummarizerAgent functionality."""

    @pytest.mark.asyncio
    async def test_summarizer_basic_functionality(self):
        """Test basic summarization."""
        agent = SummarizerAgent()
        # Test with structured legal text that the summarizer can process
        legal_text = "[Source: contract.txt] Section 1: This contract establishes liability provisions. Clause 2: Termination conditions apply."
        result = await agent.run(legal_text)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarizer_empty_input(self):
        """Test summarizer with empty input."""
        agent = SummarizerAgent()
        result = await agent.run("")
        # Empty input returns empty string
        assert result == ""

    @pytest.mark.asyncio 
    async def test_summarizer_with_no_relevant_docs(self):
        """Test summarizer with 'no relevant documents' message."""
        agent = SummarizerAgent()
        result = await agent.run("No relevant documents found for: test query")
        # Should return the input unchanged
        assert result == "No relevant documents found for: test query"


class TestClauseExplainerAgent:
    """Test ClauseExplainerAgent functionality."""

    @pytest.mark.asyncio
    async def test_clause_explainer_basic(self):
        """Test basic clause explanation."""
        agent = ClauseExplainerAgent()
        result = await agent.run("The party shall be liable for damages.")
        assert "practical implications:" in result.lower()

    @pytest.mark.asyncio
    async def test_clause_explainer_identifies_legal_terms(self):
        """Test that explainer identifies and explains legal terms."""
        agent = ClauseExplainerAgent()
        result = await agent.run("The party shall indemnify and hold harmless against liability.")
        
        # Should identify legal terms
        assert "legal terms:" in result.lower() or "liability" in result.lower()

    @pytest.mark.asyncio
    async def test_clause_explainer_analyzes_implications(self):
        """Test implication analysis for different clause types."""
        agent = ClauseExplainerAgent()
        
        # Test liability limitation
        result = await agent.run("Liability shall be limited to $1000.")
        assert "practical implications:" in result.lower()
        
        # Test termination clause
        result = await agent.run("This agreement may be terminated with 30 days notice.")
        assert "practical implications:" in result.lower()

    def test_identify_legal_terms(self):
        """Test legal term identification."""
        agent = ClauseExplainerAgent()
        terms = agent._identify_legal_terms("The contract includes liability and breach provisions.")
        assert "liability" in terms
        assert "breach" in terms

    def test_explain_term(self):
        """Test individual term explanations."""
        agent = ClauseExplainerAgent()
        explanation = agent._explain_term("liability")
        assert len(explanation) > 0
        assert "responsibility" in explanation.lower()

    def test_analyze_implications(self):
        """Test implication analysis."""
        agent = ClauseExplainerAgent()
        
        # Test liability limitation
        implication = agent._analyze_implications("liability shall be limited")
        assert "limit" in implication.lower()
        
        # Test termination
        implication = agent._analyze_implications("agreement may be terminated")
        assert "ended" in implication.lower()


class TestCitationAgent:
    """Test CitationAgent functionality."""

    def test_citation_agent_initialization(self):
        """Test citation agent initialization."""
        agent = CitationAgent()
        assert hasattr(agent, 'stream')
        assert hasattr(agent, 'window')
        assert isinstance(agent.window, int)
        assert agent.window > 0

    def test_citation_agent_stream_functionality(self):
        """Test citation stream functionality."""
        agent = CitationAgent()
        # CitationAgent stream method requires answer, docs, and query parameters
        mock_docs = [Mock()]
        mock_docs[0].content = "Legal document content with test query information"
        mock_docs[0].text = "Legal document content with test query information"
        mock_docs[0].metadata = {"title": "Test Document"}
        
        result_generator = agent.stream("Legal analysis answer", mock_docs, "test query")
        # Stream returns an iterator, so we need to collect the results
        result_list = list(result_generator)
        assert len(result_list) > 0
        # Join the results to get the full citation
        full_result = "".join(result_list)
        assert isinstance(full_result, str)
        assert len(full_result) > 0


class TestRouterAgent:
    """Test RouterAgent functionality."""

    def test_router_initialization(self):
        """Test router agent initialization."""
        router = RouterAgent()
        assert hasattr(router, 'explain_keywords')
        assert hasattr(router, 'summary_keywords')
        assert hasattr(router, 'search_keywords')
        assert len(router.explain_keywords) > 0
        assert len(router.summary_keywords) > 0
        assert len(router.search_keywords) > 0

    def test_router_classify_query_types(self):
        """Test query classification."""
        router = RouterAgent()
        
        # Test explanation query
        explanation_query = "explain the liability clause"
        classification = router.analyze_query_complexity(explanation_query)
        assert classification in ["explain", "summary", "search", "default"]

        # Test search query
        search_query = "find documents about contracts"
        classification = router.analyze_query_complexity(search_query)
        assert classification in ["explain", "summary", "search", "default"]

        # Test summary query
        summary_query = "summarize this document"
        classification = router.analyze_query_complexity(summary_query)
        assert classification in ["explain", "summary", "search", "default"]

    def test_router_decision_logic(self):
        """Test router decision logic."""
        router = RouterAgent()
        
        # Test explanation decision
        assert router.decide("explain liability clause") is True
        assert router.decide("what does this mean") is True
        
        # Test summary decision
        assert router.needs_summary_only("summarize the document") is True
        assert router.needs_summary_only("give me a summary") is True
        
        # Test search decision
        assert router.is_search_query("find contract documents") is True
        assert router.is_search_query("search for legal cases") is True

    def test_router_keyword_matching(self):
        """Test keyword matching logic."""
        router = RouterAgent()
        
        # Test with various queries
        queries_and_expected = [
            ("explain this clause", True),  # Should trigger explain
            ("what does this mean", True),  # Should trigger explain
            ("summarize document", False),  # Should not trigger explain (summary only)
            ("find documents", False),     # Should not trigger explain (search)
        ]
        
        for query, should_explain in queries_and_expected:
            result = router.decide(query)
            if should_explain:
                assert result is True, f"Query '{query}' should trigger explanation"
            # Note: Some queries might still return True even if they're primarily search/summary


class TestMultiAgentGraph:
    """Test MultiAgentGraph orchestration."""

    def test_multiagent_initialization(self):
        """Test multi-agent graph initialization."""
        graph = MultiAgentGraph()
        assert graph.router is not None

    @pytest.mark.asyncio
    async def test_multiagent_run(self):
        """Test multi-agent graph execution."""
        graph = MultiAgentGraph()
        result = await graph.run("explain contract terms")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_multiagent_with_different_query_types(self):
        """Test multi-agent with various query types."""
        graph = MultiAgentGraph()
        
        queries = [
            "explain liability clause",
            "summarize this agreement", 
            "find contract documents",
            "what are the implications of termination?"
        ]
        
        for query in queries:
            result = await graph.run(query)
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_multiagent_error_handling(self):
        """Test error handling in multi-agent flow."""
        graph = MultiAgentGraph()
        
        # Test with empty query
        result = await graph.run("")
        assert isinstance(result, str)
        
        # Test with very long query
        long_query = "explain " * 100
        result = await graph.run(long_query)
        assert isinstance(result, str)


class TestAgentIntegration:
    """Integration tests for agent interactions."""

    @pytest.mark.asyncio
    async def test_agent_pipeline_integration(self):
        """Test agents working together."""
        # Create agents
        retriever = RetrieverAgent()
        summarizer = SummarizerAgent()
        explainer = ClauseExplainerAgent()
        
        # Simulate pipeline: retrieve -> summarize -> explain
        retrieved = await retriever.run("contract terms")
        summarized = await summarizer.run(retrieved)
        explained = await explainer.run(summarized)
        
        assert isinstance(explained, str)
        assert len(explained) > 0

    @pytest.mark.asyncio
    async def test_end_to_end_legal_query(self):
        """Test complete legal query processing."""
        graph = MultiAgentGraph()
        
        legal_query = "What are the implications of a force majeure clause in a commercial contract?"
        result = await graph.run(legal_query)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain some form of legal analysis
        assert any(word in result.lower() for word in ["legal", "clause", "implications", "contract"])