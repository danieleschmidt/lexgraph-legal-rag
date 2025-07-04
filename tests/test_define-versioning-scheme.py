from fastapi.testclient import TestClient

# allow imports from src
from lexgraph_legal_rag.api import create_api


def test_ping_endpoint_returns_version():
    app = create_api("v1", api_key="k")
    client = TestClient(app)
    response = client.get("/v1/ping", headers={"X-API-Key": "k"})
    assert response.status_code == 200
    assert response.json() == {"version": "v1", "ping": "pong"}
