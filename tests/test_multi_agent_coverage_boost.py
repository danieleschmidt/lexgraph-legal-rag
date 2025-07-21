"""Additional tests to boost multi-agent module coverage to 80%+.

This test suite targets the specific uncovered lines in multi_agent.py to reach the coverage goal.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass

from lexgraph_legal_rag.multi_agent import (
    RetrieverAgent,
    SummarizerAgent,
    ClauseExplainerAgent,
    CitationAgent,
    MultiAgentGraph,
    Agent,
)


class TestRetrieverAgentEdgeCases:
    """Test uncovered edge cases in RetrieverAgent."""
    
    def test_retriever_with_single_result_filtering(self):
        """Test retriever agent with single result that gets filtered out."""
        # Mock pipeline with low-score results
        mock_pipeline = Mock()
        mock_doc = Mock()
        mock_doc.text = "some legal text"
        mock_doc.source = "doc.pdf"
        
        # Return result with very low score (should be filtered)
        mock_pipeline.search.return_value = [(mock_doc, 0.001)]  # Below 0.01 threshold
        
        agent = RetrieverAgent(pipeline=mock_pipeline, top_k=1)
        
        import asyncio
        result = asyncio.run(agent.run("test query"))
        assert "No relevant documents found" in result

    def test_retriever_mixed_score_filtering(self):
        """Test retriever agent filters mixed high/low scores correctly."""
        mock_pipeline = Mock()
        
        # Create documents with mixed scores
        high_doc = Mock()
        high_doc.text = "relevant legal text"
        high_doc.source = "relevant.pdf"
        
        low_doc = Mock()
        low_doc.text = "irrelevant text"
        low_doc.source = "irrelevant.pdf"
        
        mock_pipeline.search.return_value = [
            (high_doc, 0.8),   # High score - should keep
            (low_doc, 0.005),  # Low score - should filter out
        ]
        
        agent = RetrieverAgent(pipeline=mock_pipeline)
        
        import asyncio
        result = asyncio.run(agent.run("test query"))
        
        # Should contain the high-score document
        assert "relevant legal text" in result
        assert "irrelevant text" not in result


class TestSummarizerAgentEdgeCases:
    """Test uncovered edge cases in SummarizerAgent."""
    
    def test_summarizer_with_no_documents(self):
        """Test summarizer agent with no document text."""
        agent = SummarizerAgent()
        
        import asyncio
        result = asyncio.run(agent.run(""))
        assert "No documents to summarize" in result

    def test_summarizer_with_whitespace_only(self):
        """Test summarizer agent with whitespace-only input."""
        agent = SummarizerAgent()
        
        import asyncio
        result = asyncio.run(agent.run("   \n\t  "))
        assert "No documents to summarize" in result


class TestClauseExplainerEdgeCases:
    """Test uncovered edge cases in ClauseExplainerAgent."""
    
    def test_clause_explainer_empty_input(self):
        """Test clause explainer with empty input."""
        agent = ClauseExplainerAgent()
        
        import asyncio
        result = asyncio.run(agent.run(""))
        assert "No text provided" in result

    def test_clause_explainer_whitespace_input(self):
        """Test clause explainer with whitespace-only input."""
        agent = ClauseExplainerAgent()
        
        import asyncio
        result = asyncio.run(agent.run("   \n  \t  "))
        assert "No text provided" in result


class TestCitationAgentEdgeCases:
    """Test uncovered edge cases in CitationAgent including the failing test."""
    
    def test_citation_agent_snippet_with_string_index(self):
        """Test citation agent _snippet method with proper string content."""
        agent = CitationAgent(window=50)
        
        # Create proper document mock with string content
        mock_doc = Mock()
        mock_doc.text = "This is a sample legal document with important clauses and provisions for testing the snippet extraction functionality."
        mock_doc.source = "test_document.pdf"
        
        # Test snippet extraction
        query = "important clauses"
        snippet = agent._snippet(mock_doc, query)
        
        # Should find the query text and create a snippet
        assert "important clauses" in snippet
        assert len(snippet) <= 150  # Should respect window constraints

    def test_citation_agent_snippet_query_not_found(self):
        """Test citation agent _snippet when query is not found in document."""
        agent = CitationAgent(window=50)
        
        mock_doc = Mock()
        mock_doc.text = "This document does not contain the searched term."
        mock_doc.source = "test.pdf"
        
        snippet = agent._snippet(mock_doc, "nonexistent query")
        
        # Should return beginning of document when query not found
        assert "This document does not" in snippet
        assert len(snippet) <= 150

    def test_citation_agent_stream_with_proper_documents(self):
        """Test citation agent stream functionality with properly mocked documents."""
        agent = CitationAgent(window=50)
        
        # Create proper document mocks
        doc1 = Mock()
        doc1.text = "First legal document containing the search term for testing purposes."
        doc1.source = "doc1.pdf"
        
        doc2 = Mock()
        doc2.text = "Second document with relevant legal content for comprehensive testing."
        doc2.source = "doc2.pdf"
        
        documents = [(doc1, 0.9), (doc2, 0.8)]
        query = "search term"
        
        # Test the stream method
        citations = list(agent.stream(documents, query))
        
        assert len(citations) == 2
        for citation in citations:
            assert "source" in citation
            assert "snippet" in citation
            assert "score" in citation

    def test_citation_agent_snippet_edge_boundaries(self):
        """Test citation agent snippet at document boundaries."""
        agent = CitationAgent(window=10)
        
        # Short document
        short_doc = Mock()
        short_doc.text = "Short doc"
        short_doc.source = "short.pdf"
        
        snippet = agent._snippet(short_doc, "doc")
        assert snippet == "Short doc"  # Should return full text when shorter than window

    def test_citation_agent_snippet_query_at_start(self):
        """Test citation agent snippet when query is at document start."""
        agent = CitationAgent(window=20)
        
        doc = Mock()
        doc.text = "Query term is at the very beginning of this document text."
        doc.source = "start.pdf"
        
        snippet = agent._snippet(doc, "Query")
        assert snippet.startswith("Query term")
        assert len(snippet) <= 60  # window * 3

    def test_citation_agent_snippet_query_at_end(self):
        """Test citation agent snippet when query is at document end."""
        agent = CitationAgent(window=20)
        
        doc = Mock()
        doc.text = "This document has the search term at the very end Query"
        doc.source = "end.pdf"
        
        snippet = agent._snippet(doc, "Query")
        assert snippet.endswith("Query")
        assert "search term" in snippet


class TestMultiAgentGraphEdgeCases:
    """Test uncovered edge cases in MultiAgentGraph."""
    
    def test_multiagent_graph_unknown_query_type(self):
        """Test multiagent graph with unknown query type."""
        # Create mock agents
        mock_retriever = AsyncMock()
        mock_retriever.run.return_value = "retrieved: unknown query"
        
        mock_summarizer = AsyncMock() 
        mock_summarizer.run.return_value = "summary: unknown type"
        
        mock_explainer = AsyncMock()
        mock_explainer.run.return_value = "explanation: unknown"
        
        graph = MultiAgentGraph(
            retriever=mock_retriever,
            summarizer=mock_summarizer, 
            explainer=mock_explainer
        )
        
        import asyncio
        result = asyncio.run(graph.run("some unknown query type"))
        
        # Should use retriever as default
        assert "retrieved" in result

    def test_multiagent_graph_route_decision_logic(self):
        """Test the specific routing logic branches."""
        mock_retriever = AsyncMock()
        mock_retriever.run.return_value = "retrieved content"
        
        mock_summarizer = AsyncMock()
        mock_summarizer.run.return_value = "summary content"
        
        mock_explainer = AsyncMock() 
        mock_explainer.run.return_value = "explained content"
        
        graph = MultiAgentGraph(
            retriever=mock_retriever,
            summarizer=mock_summarizer,
            explainer=mock_explainer
        )
        
        import asyncio
        
        # Test summary routing
        summary_queries = ["summarize this document", "give me a summary"]
        for query in summary_queries:
            result = asyncio.run(graph.run(query))
            mock_summarizer.run.assert_called()
        
        # Test explain routing
        explain_queries = ["explain this clause", "what does this mean"]
        for query in explain_queries:
            result = asyncio.run(graph.run(query))
            mock_explainer.run.assert_called()

    def test_multiagent_graph_case_insensitive_routing(self):
        """Test that routing is case insensitive."""
        mock_retriever = AsyncMock()
        mock_summarizer = AsyncMock()
        mock_summarizer.run.return_value = "summary result"
        mock_explainer = AsyncMock()
        
        graph = MultiAgentGraph(
            retriever=mock_retriever,
            summarizer=mock_summarizer,
            explainer=mock_explainer
        )
        
        import asyncio
        
        # Test uppercase
        result = asyncio.run(graph.run("SUMMARIZE this document"))
        mock_summarizer.run.assert_called()
        
        # Test mixed case
        result = asyncio.run(graph.run("Explain This Clause"))
        mock_explainer.run.assert_called()


class TestAgentProtocol:
    """Test agent protocol compliance."""
    
    def test_all_agents_implement_protocol(self):
        """Test that all agent classes implement the Agent protocol correctly."""
        from lexgraph_legal_rag.multi_agent import Agent
        
        # Test that our agents can be used as Agent protocol
        agents = [
            RetrieverAgent(),
            SummarizerAgent(),
            ClauseExplainerAgent(),
            CitationAgent(),
        ]
        
        for agent in agents:
            # Check that agent has run method
            assert hasattr(agent, 'run')
            assert callable(getattr(agent, 'run'))


class TestDataclassDefaults:
    """Test dataclass default values and field configurations."""
    
    def test_retriever_agent_defaults(self):
        """Test RetrieverAgent default field values."""
        agent = RetrieverAgent()
        assert agent.pipeline is None
        assert agent.top_k == 3

    def test_citation_agent_defaults(self):
        """Test CitationAgent default field values."""
        agent = CitationAgent()
        assert agent.window == 50

    def test_summarizer_agent_defaults(self):
        """Test SummarizerAgent default values."""
        agent = SummarizerAgent()
        # SummarizerAgent has no explicit defaults, but should initialize

    def test_clause_explainer_defaults(self):
        """Test ClauseExplainerAgent default values."""
        agent = ClauseExplainerAgent()
        # ClauseExplainerAgent has no explicit defaults, but should initialize


class TestLoggingBehavior:
    """Test logging behavior in multi-agent components."""
    
    @patch('lexgraph_legal_rag.multi_agent.logger')
    def test_retriever_logging_debug(self, mock_logger):
        """Test RetrieverAgent debug logging."""
        agent = RetrieverAgent()
        
        import asyncio
        result = asyncio.run(agent.run("test query"))
        
        # Should log debug message
        mock_logger.debug.assert_called_once()

    @patch('lexgraph_legal_rag.multi_agent.logger')
    def test_retriever_logging_warning(self, mock_logger):
        """Test RetrieverAgent warning logging when no pipeline."""
        agent = RetrieverAgent(pipeline=None)
        
        import asyncio
        result = asyncio.run(agent.run("test query"))
        
        # Should log warning about no pipeline
        mock_logger.warning.assert_called_once()

    @patch('lexgraph_legal_rag.multi_agent.logger')
    def test_retriever_logging_info_no_results(self, mock_logger):
        """Test RetrieverAgent info logging when no relevant results."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = []  # No results
        
        agent = RetrieverAgent(pipeline=mock_pipeline)
        
        import asyncio
        result = asyncio.run(agent.run("test query"))
        
        # Should log info about no relevant documents
        mock_logger.info.assert_called_once()


class TestTypeHinting:
    """Test type hinting and TYPE_CHECKING blocks."""
    
    def test_type_checking_imports(self):
        """Test that TYPE_CHECKING imports don't cause runtime errors."""
        # This test ensures that the TYPE_CHECKING block doesn't break
        from lexgraph_legal_rag import multi_agent
        
        # Should be able to import without issues
        assert hasattr(multi_agent, 'RetrieverAgent')
        assert hasattr(multi_agent, 'MultiAgentGraph')