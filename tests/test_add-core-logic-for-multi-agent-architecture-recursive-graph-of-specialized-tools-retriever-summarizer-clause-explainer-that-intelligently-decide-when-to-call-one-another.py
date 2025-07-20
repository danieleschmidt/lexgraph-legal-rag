from lexgraph_legal_rag.multi_agent import MultiAgentGraph
import asyncio


def test_router_executes_correct_path():
    """Test that router executes the correct path for different query types."""
    graph = MultiAgentGraph()
    
    # Test explanation query - should trigger explanation logic and return comprehensive response
    result_explain = asyncio.run(graph.run("please explain this clause"))
    assert len(result_explain) > 50  # Should be comprehensive
    assert any(word in result_explain.lower() for word in ["legal", "explanation", "implications", "procedures"])
    
    # Test summary query - should trigger summary-only logic and be shorter
    result_summary = asyncio.run(graph.run("summarize this clause"))
    # The summary query with router logic should return the stub text since it's summary-only
    assert "retrieved: summarize this clause" in result_summary
    
    # Test that different query types produce different results
    assert result_explain != result_summary
    
    # Test search query - should return retrieved content
    result_search = asyncio.run(graph.run("find information about clauses"))
    assert "retrieved: find information about clauses" in result_search
