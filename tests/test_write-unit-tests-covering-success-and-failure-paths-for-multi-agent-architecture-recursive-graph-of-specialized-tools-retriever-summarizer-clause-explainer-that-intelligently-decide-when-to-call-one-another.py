import pytest

# allow imports from src
from lexgraph_legal_rag.multi_agent import MultiAgentGraph
import asyncio


def test_success_and_failure_paths():
    graph = MultiAgentGraph()
    # success path: explanation requested
    result_explain = asyncio.run(graph.run("please explain arbitration clause"))
    assert (
        result_explain
        == "explanation of summary of retrieved: please explain arbitration clause"
    )
    # failure path: only summarization
    result_summary = asyncio.run(graph.run("summarize arbitration clause"))
    assert result_summary == "summary of retrieved: summarize arbitration clause"


def test_invalid_input_type():
    graph = MultiAgentGraph()
    with pytest.raises(TypeError):
        asyncio.run(graph.run(None))
