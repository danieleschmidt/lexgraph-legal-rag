from fastapi.testclient import TestClient
import logging
from lexgraph_legal_rag.api import create_api


def test_invalid_key_logs(caplog):
    app = create_api(api_key="secret")
    client = TestClient(app)
    with caplog.at_level(logging.WARNING):
        resp = client.get("/v1/ping", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401
    assert any("invalid API key attempt" in rec.message for rec in caplog.records)


def test_rate_limit_logs(caplog):
    app = create_api(api_key="k", rate_limit=1)
    client = TestClient(app)
    headers = {"X-API-Key": "k"}
    assert client.get("/v1/ping", headers=headers).status_code == 200
    with caplog.at_level(logging.WARNING):
        resp = client.get("/v1/ping", headers=headers)
    assert resp.status_code == 429
    assert any("rate limit exceeded" in rec.message for rec in caplog.records)
