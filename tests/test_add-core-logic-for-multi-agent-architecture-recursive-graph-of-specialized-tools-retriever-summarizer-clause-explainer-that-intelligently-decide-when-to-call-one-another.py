from lexgraph_legal_rag.multi_agent import MultiAgentGraph
import asyncio


def test_router_executes_correct_path():
    graph = MultiAgentGraph()
    result_explain = asyncio.run(graph.run("please explain this clause"))
    assert (
        result_explain
        == "explanation of summary of retrieved: please explain this clause"
    )
    result_summary = asyncio.run(graph.run("summarize this clause"))
    assert result_summary == "summary of retrieved: summarize this clause"
