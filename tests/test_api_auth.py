from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api


def test_api_key_required():
    app = create_api(api_key="secret")
    client = TestClient(app)
    resp = client.get("/v1/ping")
    assert resp.status_code == 401


def test_api_key_success():
    app = create_api(api_key="secret")
    client = TestClient(app)
    resp = client.get("/v1/ping", headers={"X-API-Key": "secret"})
    assert resp.status_code == 200
