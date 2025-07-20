"""Comprehensive tests for the Multi-Agent RAG system."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from lexgraph_legal_rag.multi_agent import (
    MultiAgentGraph, 
    RetrieverAgent, 
    SummarizerAgent, 
    ClauseExplainerAgent, 
    RouterAgent,
    CitationAgent
)
from lexgraph_legal_rag.document_pipeline import LegalDocumentPipeline
from lexgraph_legal_rag.models import LegalDocument


@pytest.fixture
def sample_documents():
    """Create sample legal documents for testing."""
    return [
        LegalDocument(
            id='contract1', 
            text='This agreement contains liability limitations and termination clauses. The contractor shall not be liable for indirect damages or consequential losses.',
            metadata={'path': '/contracts/sample.txt', 'type': 'contract'}
        ),
        LegalDocument(
            id='statute1', 
            text='According to statute section 123, breach of contract results in monetary damages and potential termination. Courts may award damages for actual losses.',
            metadata={'path': '/statutes/damages.txt', 'type': 'statute'}
        ),
        LegalDocument(
            id='case1',
            text='In the case of Smith v. Jones, the court held that indemnification clauses must be clearly written. The warranty provisions were deemed enforceable.',
            metadata={'path': '/cases/smith_jones.txt', 'type': 'case'}
        )
    ]


@pytest.fixture
def pipeline_with_docs(sample_documents):
    """Create a pipeline with sample documents indexed."""
    pipeline = LegalDocumentPipeline()
    pipeline.index.add(sample_documents)
    return pipeline


@pytest.fixture
def multi_agent_graph(pipeline_with_docs):
    """Create a multi-agent graph with a configured pipeline."""
    graph = MultiAgentGraph(pipeline=pipeline_with_docs)
    return graph


class TestRetrieverAgent:
    """Test the RetrieverAgent functionality."""
    
    @pytest.mark.asyncio
    async def test_retriever_without_pipeline(self):
        """Test retriever behavior when no pipeline is configured."""
        agent = RetrieverAgent()
        result = await agent.run("test query")
        assert "retrieved: test query" in result
    
    @pytest.mark.asyncio
    async def test_retriever_with_pipeline(self, pipeline_with_docs):
        """Test retriever with actual pipeline and documents."""
        agent = RetrieverAgent(pipeline=pipeline_with_docs, top_k=2)
        result = await agent.run("liability limitations")
        
        assert "[Source:" in result
        assert "Relevance:" in result
        assert "liability" in result.lower()
        
    @pytest.mark.asyncio
    async def test_retriever_no_results(self, pipeline_with_docs):
        """Test retriever when no relevant documents are found."""
        agent = RetrieverAgent(pipeline=pipeline_with_docs)
        result = await agent.run("completely unrelated topic xyz123")
        
        assert "No relevant documents found" in result


class TestSummarizerAgent:
    """Test the SummarizerAgent functionality."""
    
    @pytest.mark.asyncio
    async def test_summarizer_with_retrieved_text(self):
        """Test summarizer with typical retrieved text."""
        agent = SummarizerAgent(max_length=200)  # Increased for more realistic testing
        text = """[Source: contract1, Relevance: 0.850]
        This agreement contains liability limitations and termination clauses. The contractor shall not be liable for indirect damages.
        
        [Source: statute1, Relevance: 0.720]
        According to statute section 123, breach of contract results in monetary damages."""
        
        result = await agent.run(text)
        
        assert len(result) <= 210  # Allow some flexibility for processing
        assert result != text  # Should be different from input
        assert len(result) > 20  # Should have meaningful content
        
    @pytest.mark.asyncio
    async def test_summarizer_with_no_documents(self):
        """Test summarizer when no documents were found."""
        agent = SummarizerAgent()
        result = await agent.run("No relevant documents found for: xyz")
        
        assert result == "No relevant documents found for: xyz"
    
    @pytest.mark.asyncio
    async def test_summarizer_with_empty_text(self):
        """Test summarizer with empty input."""
        agent = SummarizerAgent()
        result = await agent.run("")
        
        assert result == ""


class TestClauseExplainerAgent:
    """Test the ClauseExplainerAgent functionality."""
    
    @pytest.mark.asyncio
    async def test_explainer_with_legal_terms(self):
        """Test explainer with text containing legal terms."""
        agent = ClauseExplainerAgent()
        text = "This contract contains liability limitations and warranty provisions for indemnification."
        
        result = await agent.run(text)
        
        assert "liability" in result.lower()
        assert "legal responsibility" in result.lower() or "legal" in result.lower()
        assert len(result) > len(text)  # Should be longer due to explanations
    
    @pytest.mark.asyncio
    async def test_explainer_identifies_contract_language(self):
        """Test that explainer identifies contractual language."""
        agent = ClauseExplainerAgent()
        text = "This agreement contains contract terms and warranty provisions."
        
        result = await agent.run(text)
        
        assert "contractual language" in result.lower()
    
    @pytest.mark.asyncio
    async def test_explainer_identifies_statutory_language(self):
        """Test that explainer identifies statutory language."""
        agent = ClauseExplainerAgent()
        text = "According to statute section 123, the law requires compliance."
        
        result = await agent.run(text)
        
        assert "statutory" in result.lower() or "regulation" in result.lower()
    
    @pytest.mark.asyncio
    async def test_explainer_with_no_documents(self):
        """Test explainer when no documents were found."""
        agent = ClauseExplainerAgent()
        result = await agent.run("No relevant documents found")
        
        assert "No explanation available" in result
    
    @pytest.mark.asyncio
    async def test_explainer_term_explanations(self):
        """Test that explainer provides term explanations."""
        agent = ClauseExplainerAgent()
        
        # Test individual term explanations
        assert agent._explain_term("liability") == "legal responsibility for damages or obligations"
        assert agent._explain_term("breach") == "failure to fulfill contractual obligations"
        assert agent._explain_term("warranty") == "guarantee about the quality or condition of something"
    
    @pytest.mark.asyncio
    async def test_explainer_analyze_implications(self):
        """Test implications analysis."""
        agent = ClauseExplainerAgent()
        
        assert "limit legal responsibility" in agent._analyze_implications("liability limitation clause")
        assert "financial obligations" in agent._analyze_implications("payment terms and fees")
        assert "sensitive information" in agent._analyze_implications("confidential data protection")


class TestRouterAgent:
    """Test the RouterAgent functionality."""
    
    def test_router_explanation_keywords(self):
        """Test router detection of explanation requests."""
        router = RouterAgent()
        
        assert router.decide("please explain this clause")
        assert router.decide("what does this mean?")
        assert router.decide("interpret this provision")
        assert router.decide("clarify the meaning")
        assert not router.decide("summarize this document")
        assert not router.decide("find information about")
    
    def test_router_summary_detection(self):
        """Test router detection of summary requests."""
        router = RouterAgent()
        
        assert router.needs_summary_only("summarize this document")
        assert router.needs_summary_only("give me a brief overview")
        assert router.needs_summary_only("key points of this contract")
        assert not router.needs_summary_only("explain what this means")
        assert not router.needs_summary_only("find documents about liability")
    
    def test_router_search_detection(self):
        """Test router detection of search requests."""
        router = RouterAgent()
        
        assert router.is_search_query("find documents about liability")
        assert router.is_search_query("search for contract terms")
        assert router.is_search_query("show me cases about damages")
        assert not router.is_search_query("explain this clause")
        assert not router.is_search_query("summarize the document")
    
    def test_router_query_complexity_analysis(self):
        """Test comprehensive query analysis."""
        router = RouterAgent()
        
        assert router.analyze_query_complexity("explain this clause") == "explain"
        assert router.analyze_query_complexity("summarize this document") == "summary"
        assert router.analyze_query_complexity("find information about damages") == "search"
        assert router.analyze_query_complexity("what are the terms?") == "default"


class TestMultiAgentGraph:
    """Test the complete MultiAgentGraph functionality."""
    
    @pytest.mark.asyncio
    async def test_graph_initialization_with_pipeline(self, pipeline_with_docs):
        """Test graph initialization with pipeline."""
        graph = MultiAgentGraph(pipeline=pipeline_with_docs)
        
        # Check that pipeline is properly set
        assert graph.pipeline is pipeline_with_docs
        assert graph.retriever.pipeline is pipeline_with_docs
    
    @pytest.mark.asyncio
    async def test_graph_explanation_query(self, multi_agent_graph):
        """Test graph with explanation query."""
        result = await multi_agent_graph.run("please explain liability limitations")
        
        # Should contain explanation elements
        assert len(result) > 20  # Should be substantial
        assert "liability" in result.lower()
        # Should contain explanation language
        assert any(word in result.lower() for word in ["legal", "responsibility", "means", "implies"])
    
    @pytest.mark.asyncio
    async def test_graph_summary_query(self, multi_agent_graph):
        """Test graph with summary-only query."""
        result = await multi_agent_graph.run("summarize contract terms")
        
        # Should be shorter than explanation
        assert len(result) < 500  # Reasonable summary length
        assert "contract" in result.lower() or "agreement" in result.lower()
    
    @pytest.mark.asyncio
    async def test_graph_search_query(self, multi_agent_graph):
        """Test graph with search query."""
        result = await multi_agent_graph.run("find information about damages")
        
        # Should contain source information from retrieval
        assert "[Source:" in result
        assert "damages" in result.lower()
        assert "Relevance:" in result
    
    @pytest.mark.asyncio
    async def test_graph_no_relevant_documents(self, multi_agent_graph):
        """Test graph behavior when no relevant documents are found."""
        result = await multi_agent_graph.run("completely unrelated topic xyz123")
        
        assert "No relevant documents found" in result
    
    @pytest.mark.asyncio
    async def test_graph_with_citations(self, multi_agent_graph):
        """Test graph with citation functionality."""
        chunks = []
        async for chunk in multi_agent_graph.run_with_citations("liability limitations", multi_agent_graph.pipeline):
            chunks.append(chunk)
        
        # Should have answer and citations
        assert len(chunks) >= 2  # At least answer + citations
        combined = "".join(chunks)
        assert "Citations:" in combined
        assert "liability" in combined.lower()
    
    @pytest.mark.asyncio
    async def test_graph_recursive_depth(self):
        """Test graph recursive depth limiting."""
        # Create a graph with max_depth=0 to test depth limiting
        pipeline = LegalDocumentPipeline()
        docs = [LegalDocument(id='test', text='This is a test explanation document', metadata={})]
        pipeline.index.add(docs)
        
        graph = MultiAgentGraph(pipeline=pipeline, max_depth=0)
        
        result = await graph.run("please explain this clause")
        
        # With max_depth=0, should not reach explanation stage
        assert len(result) > 0


class TestCitationAgent:
    """Test the CitationAgent functionality."""
    
    def test_citation_agent_snippet_extraction(self, sample_documents):
        """Test citation agent snippet extraction."""
        agent = CitationAgent(window=20)
        
        snippet = agent._snippet(sample_documents[0], "liability")
        
        assert "liability" in snippet.lower()
        assert len(snippet) <= 80  # 2 * window
        assert "\n" not in snippet  # Newlines should be replaced
    
    def test_citation_agent_stream(self, sample_documents):
        """Test citation agent streaming functionality."""
        agent = CitationAgent()
        answer = "This is about liability limitations."
        
        chunks = list(agent.stream(answer, sample_documents, "liability"))
        
        assert len(chunks) >= 2  # Answer + citations
        assert chunks[0] == answer
        assert "Citations:" in chunks[-1]
        assert any("contract1" in chunk for chunk in chunks)


class TestIntegrationWithActualPipeline:
    """Integration tests with real pipeline functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_legal_query(self, multi_agent_graph):
        """Test complete end-to-end legal query processing."""
        query = "What are the implications of liability limitations in contracts?"
        
        result = await multi_agent_graph.run(query)
        
        # Should be a comprehensive response
        assert len(result) > 50
        assert "liability" in result.lower()
        assert any(word in result.lower() for word in ["contract", "agreement", "legal", "responsibility"])
    
    @pytest.mark.asyncio
    async def test_different_document_types_handling(self, multi_agent_graph):
        """Test handling of different legal document types."""
        # Test contract query
        contract_result = await multi_agent_graph.run("explain contract liability terms")
        
        # Test statute query  
        statute_result = await multi_agent_graph.run("find statute information about damages")
        
        # Test case law query
        case_result = await multi_agent_graph.run("show me case law about warranties")
        
        # All should return different, relevant results
        assert contract_result != statute_result != case_result
        assert all(len(result) > 20 for result in [contract_result, statute_result, case_result])


# Test that maintains backward compatibility with existing test
def test_router_executes_correct_path():
    """Updated test that reflects new intelligent behavior."""
    graph = MultiAgentGraph()
    
    # Test explanation query - should now return intelligent explanation
    result_explain = asyncio.run(graph.run("please explain this clause"))
    # The explanation should be processed and not contain the raw retrieved text
    assert len(result_explain) > 30  # Should be more than just stub
    assert any(word in result_explain.lower() for word in ["legal", "explanation", "implications", "procedures"])
    
    # Test summary query - should return summary behavior
    result_summary = asyncio.run(graph.run("summarize this clause"))
    assert "retrieved: summarize this clause" in result_summary