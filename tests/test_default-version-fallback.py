from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api


def test_default_version_fallback():
    app = create_api()
    client = TestClient(app)
    res = client.get("/ping")
    assert res.status_code == 200
    assert res.json() == {"version": "v1", "ping": "pong"}
