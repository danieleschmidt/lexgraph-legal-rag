import pathlib
import sys
from fastapi.testclient import TestClient

# allow imports from src
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lexgraph_legal_rag.api import create_api


def test_ping_endpoint_returns_version():
    app = create_api("v1")
    client = TestClient(app)
    response = client.get("/v1/ping")
    assert response.status_code == 200
    assert response.json() == {"version": "v1", "ping": "pong"}
