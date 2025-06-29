from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api


def test_openapi_includes_models():
    app = create_api(api_key="k")
    client = TestClient(app)
    spec = client.get("/openapi.json", headers={"X-API-Key": "k"}).json()
    schemas = spec.get("components", {}).get("schemas", {})
    assert "PingResponse" in schemas
    assert "AddResponse" in schemas
