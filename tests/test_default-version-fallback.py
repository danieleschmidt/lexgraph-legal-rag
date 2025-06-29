from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api


def test_default_version_fallback():
    app = create_api(api_key="k")
    client = TestClient(app)
    res = client.get("/ping", headers={"X-API-Key": "k"})
    assert res.status_code == 200
    assert res.json() == {"version": "v1", "ping": "pong"}
