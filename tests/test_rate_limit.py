from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api


def test_rate_limit_exceeded():
    app = create_api(api_key="k", rate_limit=2)
    client = TestClient(app)
    headers = {"X-API-Key": "k"}
    assert client.get("/v1/ping", headers=headers).status_code == 200
    assert client.get("/v1/ping", headers=headers).status_code == 200
    resp = client.get("/v1/ping", headers=headers)
    assert resp.status_code == 429
