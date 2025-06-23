import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import lexgraph_legal_rag.sample as sample


def test_add():
    assert sample.add(1, 2) == 3
