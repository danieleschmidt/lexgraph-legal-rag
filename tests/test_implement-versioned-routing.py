from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api


def test_add_route_versioned_prefix():
    app = create_api("v1", api_key="k")
    client = TestClient(app)
    headers = {"X-API-Key": "k"}
    res = client.get("/v1/add", params={"a": 2, "b": 3}, headers=headers)
    assert res.status_code == 200
    assert res.json() == {"result": 5}
    res_root = client.get("/add", params={"a": 2, "b": 3}, headers=headers)
    assert res_root.status_code == 200
    assert res_root.json() == {"result": 5}
