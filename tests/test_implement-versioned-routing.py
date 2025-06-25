import pathlib
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lexgraph_legal_rag.api import create_api


def test_add_route_versioned_prefix():
    app = create_api("v1")
    client = TestClient(app)
    res = client.get("/v1/add", params={"a": 2, "b": 3})
    assert res.status_code == 200
    assert res.json() == {"result": 5}
    res_root = client.get("/add", params={"a": 2, "b": 3})
    assert res_root.status_code == 200
    assert res_root.json() == {"result": 5}
