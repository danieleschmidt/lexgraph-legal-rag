import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lexgraph_legal_rag.multi_agent import MultiAgentGraph


def test_pipeline_runs_successfully():
    graph = MultiAgentGraph()
    result = graph.run("explain test query")
    assert result == "explanation of summary of retrieved: explain test query"
