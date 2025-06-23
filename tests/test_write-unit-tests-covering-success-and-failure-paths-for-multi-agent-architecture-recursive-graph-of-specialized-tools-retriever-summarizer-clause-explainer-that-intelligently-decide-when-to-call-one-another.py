import pathlib
import sys
import pytest

# allow imports from src
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lexgraph_legal_rag.multi_agent import MultiAgentGraph


def test_success_and_failure_paths():
    graph = MultiAgentGraph()
    # success path: explanation requested
    result_explain = graph.run("please explain arbitration clause")
    assert result_explain == "explanation of summary of retrieved: please explain arbitration clause"
    # failure path: only summarization
    result_summary = graph.run("summarize arbitration clause")
    assert result_summary == "summary of retrieved: summarize arbitration clause"


def test_invalid_input_type():
    graph = MultiAgentGraph()
    with pytest.raises(TypeError):
        graph.run(None)
