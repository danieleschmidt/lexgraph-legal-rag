"""Tests for API key management functionality."""

import os
from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api
from lexgraph_legal_rag.auth import get_key_manager, APIKeyManager


def test_api_key_manager_basic_functionality():
    """Test basic API key manager operations."""
    manager = APIKeyManager()
    
    # Add a key (using test mode for unit testing)
    manager.add_key("test-key-1", test_mode=True)
    assert manager.is_valid_key("test-key-1")
    assert manager.get_active_key_count() == 1
    
    # Add another key
    manager.add_key("test-key-2", test_mode=True)
    assert manager.get_active_key_count() == 2
    
    # Revoke a key
    manager.revoke_key("test-key-1")
    assert not manager.is_valid_key("test-key-1")
    assert manager.is_valid_key("test-key-2")
    
    # Get rotation info
    info = manager.get_rotation_info()
    assert "active_keys" in info
    assert "revoked_keys" in info
    assert info["revoked_keys"] == 1


def test_key_rotation_endpoint():
    """Test the key rotation admin endpoint."""
    app = create_api(api_key="original-key", test_mode=True)
    client = TestClient(app)
    headers = {"X-API-Key": "original-key"}
    
    # Test key rotation
    rotation_data = {"new_primary_key": "new-key-123"}
    resp = client.post("/v1/admin/rotate-keys", json=rotation_data, headers=headers)
    
    assert resp.status_code == 200
    data = resp.json()
    assert "Key rotation completed successfully" in data["message"]
    assert "rotation_info" in data


def test_key_revocation_endpoint():
    """Test the key revocation admin endpoint."""
    app = create_api(api_key="admin-key", test_mode=True)
    client = TestClient(app)
    headers = {"X-API-Key": "admin-key"}
    
    # First add a key to revoke
    rotation_data = {"new_primary_key": "key-to-revoke"}
    client.post("/v1/admin/rotate-keys", json=rotation_data, headers=headers)
    
    # Now revoke it
    revocation_data = {"api_key": "key-to-revoke"}
    resp = client.post("/v1/admin/revoke-key", json=revocation_data, headers=headers)
    
    assert resp.status_code == 200
    data = resp.json()
    assert "Key revocation completed successfully" in data["message"]


def test_key_status_endpoint():
    """Test the key status admin endpoint."""
    app = create_api(api_key="status-key", test_mode=True)
    client = TestClient(app)
    headers = {"X-API-Key": "status-key"}
    
    resp = client.get("/v1/admin/key-status", headers=headers)
    
    assert resp.status_code == 200
    data = resp.json()
    assert "Key status retrieved successfully" in data["message"]
    assert "rotation_info" in data
    assert "active_keys" in data["rotation_info"]


def test_admin_endpoints_require_auth():
    """Test that admin endpoints require authentication."""
    app = create_api(api_key="secret", test_mode=True)
    client = TestClient(app)
    
    # Try without auth header
    resp = client.get("/v1/admin/key-status")
    assert resp.status_code == 401
    
    # Try with wrong key
    resp = client.get("/v1/admin/key-status", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401
    
    # Try rotation without auth
    resp = client.post("/v1/admin/rotate-keys", json={"new_primary_key": "test"})
    assert resp.status_code == 401


def test_multiple_keys_from_environment():
    """Test loading multiple keys from environment variables."""
    # This test would need to set environment variables
    # For now, we'll test the manager directly
    
    # Create manager with no env keys
    manager = APIKeyManager()
    
    # Manually add multiple keys to simulate env loading
    manager.add_key("key1", test_mode=True)
    manager.add_key("key2", test_mode=True)
    manager.add_key("key3", test_mode=True)
    
    assert manager.get_active_key_count() == 3
    assert manager.is_valid_key("key1")
    assert manager.is_valid_key("key2")
    assert manager.is_valid_key("key3")
    
    # Revoke one
    manager.revoke_key("key2")
    assert not manager.is_valid_key("key2")
    assert manager.get_active_key_count() == 2  # Still 3 active, but 1 revoked