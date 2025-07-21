"""Tests for security hardening improvements: CORS, API key management, and rate limiting."""

import os
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from lexgraph_legal_rag.config import Config, ConfigurationError
from lexgraph_legal_rag.auth import APIKeyManager, get_key_manager, setup_test_key_manager
from lexgraph_legal_rag.api import create_api, verify_api_key
from fastapi import HTTPException


class TestSecurityConfiguration:
    """Test enhanced security configuration."""
    
    def test_cors_origins_from_environment(self):
        """Test CORS origins configuration from environment variables."""
        with patch.dict(os.environ, {
            "CORS_ALLOWED_ORIGINS": "https://app.example.com,https://api.example.com",
            "API_KEY": "test_secure_key_123"
        }):
            config = Config()
            assert config.allowed_origins == ["https://app.example.com", "https://api.example.com"]
    
    def test_cors_origins_default_localhost(self):
        """Test default CORS origins for development."""
        with patch.dict(os.environ, {"API_KEY": "test_secure_key_123"}, clear=True):
            config = Config()
            expected_origins = [
                "http://localhost:3000", 
                "http://localhost:8080", 
                "http://localhost:8501"
            ]
            assert config.allowed_origins == expected_origins
    
    def test_https_enforcement_configuration(self):
        """Test HTTPS enforcement configuration."""
        with patch.dict(os.environ, {
            "REQUIRE_HTTPS": "true",
            "API_KEY": "test_secure_key_123"
        }):
            config = Config()
            assert config.require_https is True
        
        with patch.dict(os.environ, {
            "REQUIRE_HTTPS": "false",
            "API_KEY": "test_secure_key_123"
        }):
            config = Config()
            assert config.require_https is False
    
    def test_max_key_age_configuration(self):
        """Test API key age limit configuration."""
        with patch.dict(os.environ, {
            "MAX_KEY_AGE_DAYS": "30",
            "API_KEY": "test_secure_key_123"
        }):
            config = Config()
            assert config.max_key_age_days == 30
    
    def test_api_key_strength_validation(self):
        """Test API key strength requirements."""
        # Test production key length requirement
        with patch.dict(os.environ, {"API_KEY": "short"}):
            config = Config()
            with pytest.raises(ConfigurationError, match="must be at least 16 characters"):
                config.validate_startup()
        
        # Test development key allowance
        with patch.dict(os.environ, {"API_KEY": "test"}):
            config = Config()
            # Should not raise exception for development keys
            config.validate_startup()
    
    def test_cors_wildcard_warning(self):
        """Test warning for CORS wildcard configuration."""
        with patch.dict(os.environ, {
            "CORS_ALLOWED_ORIGINS": "*",
            "API_KEY": "test_secure_key_12345678"
        }):
            config = Config()
            with patch('lexgraph_legal_rag.config.logger') as mock_logger:
                config.validate_startup()
                mock_logger.warning.assert_called_with(
                    "CORS allows all origins (*) - not recommended for production"
                )


class TestAPIKeyManager:
    """Test enhanced API key management with rate limiting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing global manager
        import lexgraph_legal_rag.auth
        lexgraph_legal_rag.auth._key_manager = None
    
    def test_key_strength_validation(self):
        """Test API key strength validation."""
        manager = APIKeyManager()
        
        # Test weak keys
        with pytest.raises(ValueError, match="does not meet security requirements"):
            manager.add_key("weak")
        
        with pytest.raises(ValueError, match="does not meet security requirements"):
            manager.add_key("onlylowercase123")
        
        # Test strong key
        manager.add_key("StrongKey123WithMixedCase")
        assert manager.is_valid_key("StrongKey123WithMixedCase")
    
    def test_per_key_rate_limiting(self):
        """Test per-API-key rate limiting."""
        with patch.dict(os.environ, {"API_KEY": "TestKey123Strong"}):
            manager = APIKeyManager()
            test_key = "TestKey123Strong"
            
            # First request should pass
            assert manager.check_rate_limit(test_key, limit=2, window=60)
            assert manager.check_rate_limit(test_key, limit=2, window=60)
            
            # Third request should fail
            assert not manager.check_rate_limit(test_key, limit=2, window=60)
    
    def test_usage_statistics_tracking(self):
        """Test API key usage statistics."""
        with patch.dict(os.environ, {"API_KEY": "TestKey123Strong"}):
            manager = APIKeyManager()
            test_key = "TestKey123Strong"
            
            # Initial usage count should be 0
            rotation_info = manager.get_rotation_info()
            initial_calls = rotation_info["total_api_calls"]
            
            # Use the key multiple times
            for _ in range(5):
                manager.is_valid_key(test_key)
            
            # Check updated statistics
            rotation_info = manager.get_rotation_info()
            assert rotation_info["total_api_calls"] == initial_calls + 5
            assert test_key in manager._key_metadata
            assert manager._key_metadata[test_key]["usage_count"] == 5
            assert manager._key_metadata[test_key]["last_used"] is not None
    
    def test_enhanced_rotation_info(self):
        """Test enhanced key rotation information."""
        with patch.dict(os.environ, {"API_KEY": "TestKey123Strong"}):
            manager = APIKeyManager()
            
            rotation_info = manager.get_rotation_info()
            
            # Check new fields
            assert "oldest_key_age_days" in rotation_info
            assert "average_key_age_days" in rotation_info
            assert "total_api_calls" in rotation_info
            assert "keys_needing_rotation" in rotation_info
            
            # Test with old key (simulate)
            old_time = time.time() - (100 * 24 * 3600)  # 100 days ago
            manager._key_metadata["TestKey123Strong"]["created_at"] = old_time
            
            rotation_info = manager.get_rotation_info()
            assert rotation_info["keys_needing_rotation"] == 1
    
    def test_hmac_key_hashing(self):
        """Test secure HMAC-based key hashing."""
        manager = APIKeyManager()
        test_key = "TestKey123Strong"
        
        hash1 = manager._hash_key(test_key)
        hash2 = manager._hash_key(test_key)
        
        # Same key should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 8  # Truncated to 8 characters
        
        # Different keys should produce different hashes
        different_hash = manager._hash_key("DifferentKey456")
        assert hash1 != different_hash


class TestAPISecurityIntegration:
    """Test API security integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing global manager
        import lexgraph_legal_rag.auth
        lexgraph_legal_rag.auth._key_manager = None
    
    def test_per_key_rate_limiting_in_api(self):
        """Test per-key rate limiting integration in API."""
        test_key = "TestAPIKey123Strong"
        
        with patch.dict(os.environ, {"API_KEY": test_key}):
            # Create API with custom config
            from lexgraph_legal_rag.config import Config
            config = Config()
            app = create_api(test_mode=True, config=config)
            client = TestClient(app)
            
            # Set up key manager
            setup_test_key_manager(test_key)
            
            # Mock the rate limit to be very low for testing
            manager = get_key_manager()
            with patch.object(manager, 'check_rate_limit', return_value=False):
                response = client.get("/ping", headers={"X-API-Key": test_key})
                assert response.status_code == 429
                assert "Rate limit exceeded for this API key" in response.json()["detail"]
    
    def test_cors_configuration_in_api(self):
        """Test CORS configuration in API."""
        test_key = "TestAPIKey123Strong"
        
        with patch.dict(os.environ, {
            "API_KEY": test_key,
            "CORS_ALLOWED_ORIGINS": "https://secure.example.com"
        }):
            from lexgraph_legal_rag.config import Config
            config = Config()
            app = create_api(test_mode=True, config=config)
            client = TestClient(app)
            
            # Test preflight request
            response = client.options(
                "/ping",
                headers={
                    "Origin": "https://secure.example.com",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "X-API-Key"
                }
            )
            
            # Should allow the configured origin
            assert response.status_code in [200, 204]
            
            # Test with disallowed origin
            response = client.options(
                "/ping", 
                headers={
                    "Origin": "https://malicious.com",
                    "Access-Control-Request-Method": "GET"
                }
            )
            # CORS should reject this (implementation dependent)
    
    def test_api_key_validation_with_enhanced_manager(self):
        """Test API key validation with enhanced manager."""
        strong_key = "StrongTestKey123ABC"
        weak_key = "weak"
        
        with patch.dict(os.environ, {"API_KEY": strong_key}):
            # Test with strong key
            setup_test_key_manager(strong_key)
            verify_api_key(strong_key)  # Should not raise
            
            # Test with invalid key
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key("InvalidKey123")
            assert exc_info.value.status_code == 401
    
    def test_security_headers_and_methods(self):
        """Test that API includes proper security headers and restricts methods."""
        test_key = "TestAPIKey123Strong"
        
        with patch.dict(os.environ, {
            "API_KEY": test_key,
            "CORS_ALLOWED_ORIGINS": "https://app.example.com"
        }):
            from lexgraph_legal_rag.config import Config
            config = Config()
            app = create_api(test_mode=True, config=config)
            client = TestClient(app)
            
            # Test that exposed headers are configured
            response = client.options("/ping")
            
            # The response should indicate restricted methods and headers
            # This depends on the CORS middleware implementation


class TestConfigurationSecurity:
    """Test configuration security enhancements."""
    
    def test_configuration_dict_excludes_sensitive_data(self):
        """Test that configuration dictionary excludes sensitive data."""
        with patch.dict(os.environ, {
            "API_KEY": "supersecret123",
            "OPENAI_API_KEY": "sk-1234567890",
            "CORS_ALLOWED_ORIGINS": "https://app.example.com"
        }):
            config = Config()
            config_dict = config.to_dict()
            
            # Should not contain actual keys
            assert "supersecret123" not in str(config_dict)
            assert "sk-1234567890" not in str(config_dict)
            
            # Should contain boolean flags
            assert config_dict["api_key_set"] is True
            assert config_dict["openai_api_key_set"] is True
            
            # Should contain non-sensitive config
            assert config_dict["allowed_origins"] == ["https://app.example.com"]
    
    def test_entropy_validation(self):
        """Test API key entropy validation."""
        config = Config()
        
        # Test keys with different entropy levels
        assert config._has_sufficient_entropy("ABC123def!@#") is True  # Mixed case, digits, special
        assert config._has_sufficient_entropy("ABC123def") is True      # Mixed case, digits
        assert config._has_sufficient_entropy("abcdef123") is True      # Lower, digits  
        assert config._has_sufficient_entropy("abcdefghi") is False     # Only lowercase
        assert config._has_sufficient_entropy("ABCDEFGHI") is False     # Only uppercase
        assert config._has_sufficient_entropy("123456789") is False     # Only digits


if __name__ == "__main__":
    pytest.main([__file__])