import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lexgraph_legal_rag.multi_agent import MultiAgentGraph


import asyncio


def test_pipeline_runs_successfully():
    graph = MultiAgentGraph()
    result = asyncio.run(graph.run("explain test query"))
    assert result == "explanation of summary of retrieved: explain test query"
