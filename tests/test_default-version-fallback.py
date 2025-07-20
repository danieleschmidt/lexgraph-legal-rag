from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api


def test_default_version_fallback():
    app = create_api(api_key="k", test_mode=True)
    client = TestClient(app)
    res = client.get("/ping", headers={"X-API-Key": "k"})
    assert res.status_code == 200
    data = res.json()
    assert data["version"] == "v1"
    assert data["ping"] == "pong"
    assert "timestamp" in data
