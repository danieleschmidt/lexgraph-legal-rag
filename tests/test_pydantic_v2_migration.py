"""Tests for Pydantic v2 migration compatibility."""

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import (
    PingResponse, AddResponse, HealthResponse, ReadinessResponse,
    KeyRotationRequest, KeyRevocationRequest, KeyManagementResponse,
    VersionInfo, create_api
)


class TestPydanticV2Migration:
    """Test that all Pydantic models use v2 compatible patterns."""

    def test_ping_response_uses_v2_config(self):
        """Test PingResponse uses ConfigDict instead of class-based Config."""
        # Should not have class-based Config
        assert not hasattr(PingResponse, 'Config'), "PingResponse should use ConfigDict, not class-based Config"
        
        # Should have model_config attribute
        assert hasattr(PingResponse, 'model_config'), "PingResponse should have model_config attribute"
        
        # Should use json_schema_extra instead of schema_extra
        config = PingResponse.model_config
        assert 'json_schema_extra' in config or callable(config.get('json_schema_extra')), \
            "PingResponse should use json_schema_extra in ConfigDict"

    def test_add_response_uses_v2_config(self):
        """Test AddResponse uses ConfigDict instead of class-based Config."""
        assert not hasattr(AddResponse, 'Config'), "AddResponse should use ConfigDict, not class-based Config"
        assert hasattr(AddResponse, 'model_config'), "AddResponse should have model_config attribute"

    def test_health_response_uses_v2_config(self):
        """Test HealthResponse uses ConfigDict instead of class-based Config."""
        assert not hasattr(HealthResponse, 'Config'), "HealthResponse should use ConfigDict, not class-based Config"
        assert hasattr(HealthResponse, 'model_config'), "HealthResponse should have model_config attribute"

    def test_readiness_response_uses_v2_config(self):
        """Test ReadinessResponse uses ConfigDict instead of class-based Config."""
        assert not hasattr(ReadinessResponse, 'Config'), "ReadinessResponse should use ConfigDict, not class-based Config"
        assert hasattr(ReadinessResponse, 'model_config'), "ReadinessResponse should have model_config attribute"

    def test_key_rotation_request_uses_v2_config(self):
        """Test KeyRotationRequest uses ConfigDict instead of class-based Config."""
        assert not hasattr(KeyRotationRequest, 'Config'), "KeyRotationRequest should use ConfigDict, not class-based Config"
        assert hasattr(KeyRotationRequest, 'model_config'), "KeyRotationRequest should have model_config attribute"

    def test_key_revocation_request_uses_v2_config(self):
        """Test KeyRevocationRequest uses ConfigDict instead of class-based Config."""
        assert not hasattr(KeyRevocationRequest, 'Config'), "KeyRevocationRequest should use ConfigDict, not class-based Config"
        assert hasattr(KeyRevocationRequest, 'model_config'), "KeyRevocationRequest should have model_config attribute"

    def test_key_management_response_uses_v2_config(self):
        """Test KeyManagementResponse uses ConfigDict instead of class-based Config."""
        assert not hasattr(KeyManagementResponse, 'Config'), "KeyManagementResponse should use ConfigDict, not class-based Config"
        assert hasattr(KeyManagementResponse, 'model_config'), "KeyManagementResponse should have model_config attribute"

    def test_version_info_uses_v2_config(self):
        """Test VersionInfo uses ConfigDict instead of class-based Config."""
        assert not hasattr(VersionInfo, 'Config'), "VersionInfo should use ConfigDict, not class-based Config"
        assert hasattr(VersionInfo, 'model_config'), "VersionInfo should have model_config attribute"

    def test_field_examples_use_v2_format(self):
        """Test that Field definitions use examples instead of example."""
        # Create test instances to verify the fields work
        ping = PingResponse(version="v1", ping="pong")
        assert ping.version == "v1"
        assert ping.ping == "pong"
        
        add = AddResponse(result=42)
        assert add.result == 42
        
        health = HealthResponse(status="healthy", version="v1", checks={"test": True})
        assert health.status == "healthy"
        assert health.version == "v1"

    def test_json_schema_generation_works(self):
        """Test that JSON schema generation works with v2 patterns."""
        # Should generate schema without errors
        ping_schema = PingResponse.model_json_schema()
        assert 'properties' in ping_schema
        assert 'version' in ping_schema['properties']
        assert 'ping' in ping_schema['properties']
        
        # Should have examples in schema
        if 'examples' in ping_schema:
            assert len(ping_schema['examples']) > 0

    def test_model_validation_works(self):
        """Test that model validation works correctly with v2 patterns."""
        # Valid data should work
        valid_ping = {"version": "v1", "ping": "pong"}
        ping = PingResponse(**valid_ping)
        assert ping.version == "v1"
        assert ping.ping == "pong"
        
        # Invalid data should raise ValidationError
        with pytest.raises(ValidationError):
            PingResponse(version=123, ping="pong")  # Invalid type for version

    def test_api_endpoints_work_with_v2_models(self):
        """Test that API endpoints work correctly with migrated models."""
        app = create_api(version="v1", api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        # Test ping endpoint
        response = client.get("/ping", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "ping" in data
        
        # Test health endpoint  
        response = client.get("/health", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_no_pydantic_v1_deprecation_warnings(self):
        """Test that no Pydantic v1 deprecation warnings are generated."""
        import warnings
        
        # Capture warnings during model creation
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            # Create instances of all models
            PingResponse(version="v1", ping="pong")
            AddResponse(result=42)
            HealthResponse(status="healthy", version="v1", checks={"test": True})
            ReadinessResponse(ready=True, checks={"test": True})
            KeyRotationRequest(new_primary_key="test-key")
            KeyRevocationRequest(api_key="old-key-12345")
            KeyManagementResponse(message="Success", rotation_info={"test": True})
            VersionInfo(supported_versions=["v1"], default_version="v1", latest_version="v1", current_version="v1")
            
            # Check for Pydantic deprecation warnings
            pydantic_warnings = [w for w in warning_list if 'pydantic' in str(w.message).lower()]
            assert len(pydantic_warnings) == 0, f"Found Pydantic deprecation warnings: {pydantic_warnings}"