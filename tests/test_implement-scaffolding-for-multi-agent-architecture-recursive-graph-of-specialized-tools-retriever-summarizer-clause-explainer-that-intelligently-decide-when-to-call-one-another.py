from lexgraph_legal_rag.multi_agent import MultiAgentGraph


import asyncio


def test_pipeline_runs_successfully():
    graph = MultiAgentGraph()
    result = asyncio.run(graph.run("explain test query"))
    # The multi-agent system now returns intelligent legal analysis instead of stub responses
    assert "Practical implications:" in result
    assert "This establishes legal rights, obligations, or procedures." in result
