"""Tests for health check and readiness endpoints."""

from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api


def test_health_endpoint_no_auth():
    """Test that health endpoint works without authentication."""
    app = create_api(api_key="secret", test_mode=True)
    client = TestClient(app)
    
    resp = client.get("/health")
    assert resp.status_code == 200
    
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "checks" in data
    assert data["checks"]["api_key_configured"] is True


def test_readiness_endpoint_no_auth():
    """Test that readiness endpoint works without authentication."""
    app = create_api(api_key="secret", test_mode=True)
    client = TestClient(app)
    
    resp = client.get("/ready")
    assert resp.status_code == 200
    
    data = resp.json()
    assert "ready" in data
    assert "checks" in data
    assert data["checks"]["api_key"]["status"] == "pass"
    assert "memory" in data["checks"]
    assert "external_services" in data["checks"]


def test_readiness_fails_without_api_key():
    """Test that readiness endpoint fails when API key is not configured."""
    app = create_api(api_key="", test_mode=True)
    client = TestClient(app)
    
    resp = client.get("/ready")
    assert resp.status_code == 200
    
    data = resp.json()
    assert data["ready"] is False
    assert data["checks"]["api_key"]["status"] == "fail"
    assert "API key not configured" in data["checks"]["api_key"]["message"]