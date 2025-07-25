"""Comprehensive test coverage for the auth module."""

import os
import time
import pytest
import secrets
from unittest.mock import patch, MagicMock
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from lexgraph_legal_rag.auth import (
    APIKeyManager,
    get_key_manager,
    setup_test_key_manager,
    validate_api_key,
    rotate_api_keys,
    revoke_api_key,
    _key_manager,
    _manager_lock
)


class TestAPIKeyManager:
    """Test cases for APIKeyManager class."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_initialization_without_env_keys(self):
        """Test manager initialization without environment keys."""
        with patch.dict(os.environ, {}, clear=True):
            manager = APIKeyManager()
            assert manager.get_active_key_count() == 0
            assert len(manager._active_keys) == 0
            assert len(manager._revoked_keys) == 0
    
    def test_initialization_with_primary_key(self):
        """Test manager initialization with primary API key."""
        test_key = "test_primary_key_123456"
        with patch.dict(os.environ, {"API_KEY": test_key}):
            manager = APIKeyManager()
            assert manager.get_active_key_count() == 1
            assert test_key in manager._active_keys
            assert manager._key_metadata[test_key]["is_primary"] is True
    
    def test_initialization_with_multiple_keys(self):
        """Test initialization with multiple API keys."""
        env_vars = {
            "API_KEY": "primary_key_123456",
            "API_KEY_1": "secondary_key_123456",
            "API_KEY_2": "tertiary_key_123456"
        }
        with patch.dict(os.environ, env_vars):
            manager = APIKeyManager()
            assert manager.get_active_key_count() == 3
            assert manager._key_metadata["primary_key_123456"]["is_primary"] is True
            assert manager._key_metadata["secondary_key_123456"]["is_primary"] is False
    
    def test_secret_key_from_environment(self):
        """Test secret key loaded from environment."""
        secret = "my_secret_key"
        with patch.dict(os.environ, {"API_KEY_SECRET": secret}):
            manager = APIKeyManager()
            assert manager._secret_key == secret.encode()
    
    def test_secret_key_generated_randomly(self):
        """Test random secret key generation when not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            manager = APIKeyManager()
            assert len(manager._secret_key) == 32
            assert isinstance(manager._secret_key, bytes)


class TestAPIKeyValidation:
    """Test cases for API key validation logic."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_valid_key_returns_true(self):
        """Test that valid keys return True."""
        manager = APIKeyManager()
        test_key = "valid_test_key_123456"
        manager.add_key(test_key, test_mode=True)
        
        assert manager.is_valid_key(test_key) is True
    
    def test_invalid_key_returns_false(self):
        """Test that invalid keys return False."""
        manager = APIKeyManager()
        assert manager.is_valid_key("nonexistent_key") is False
    
    def test_revoked_key_returns_false(self):
        """Test that revoked keys return False."""
        manager = APIKeyManager()
        test_key = "test_key_to_revoke_123456"
        manager.add_key(test_key, test_mode=True)
        manager.revoke_key(test_key)
        
        assert manager.is_valid_key(test_key) is False
    
    def test_usage_statistics_updated(self):
        """Test that usage statistics are updated on validation."""
        manager = APIKeyManager()
        test_key = "usage_test_key_123456"
        manager.add_key(test_key, test_mode=True)
        
        initial_count = manager._key_metadata[test_key]["usage_count"]
        manager.is_valid_key(test_key)
        
        assert manager._key_metadata[test_key]["usage_count"] == initial_count + 1
        assert manager._key_metadata[test_key]["last_used"] is not None


class TestKeyStrengthValidation:
    """Test cases for API key strength validation."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_test_mode_accepts_any_nonempty_key(self):
        """Test that test mode accepts any non-empty key."""
        manager = APIKeyManager()
        weak_keys = ["a", "123", "test"]
        
        for key in weak_keys:
            assert manager._validate_key_strength(key, test_mode=True) is True
    
    def test_test_mode_rejects_empty_key(self):
        """Test that test mode rejects empty keys."""
        manager = APIKeyManager()
        assert manager._validate_key_strength("", test_mode=True) is False
    
    def test_production_mode_minimum_length(self):
        """Test minimum length requirement in production mode."""
        manager = APIKeyManager()
        short_key = "short123"  # 8 chars
        long_key = "this_is_a_long_enough_key_ABC123"  # 32 chars
        
        assert manager._validate_key_strength(short_key, test_mode=False) is False
        assert manager._validate_key_strength(long_key, test_mode=False) is True
    
    def test_production_mode_entropy_requirements(self):
        """Test entropy requirements in production mode."""
        manager = APIKeyManager()
        
        # All lowercase - insufficient entropy
        weak_key = "alllowercaseletters"
        assert manager._validate_key_strength(weak_key, test_mode=False) is False
        
        # Uppercase + lowercase - sufficient entropy
        medium_key = "MixedCaseLetters"
        assert manager._validate_key_strength(medium_key, test_mode=False) is True
        
        # Uppercase + lowercase + digits - excellent entropy
        strong_key = "MixedCaseWith123Numbers"
        assert manager._validate_key_strength(strong_key, test_mode=False) is True
    
    def test_add_key_validates_strength(self):
        """Test that add_key validates key strength."""
        manager = APIKeyManager()
        
        with pytest.raises(ValueError, match="security requirements"):
            manager.add_key("weak", test_mode=False)


class TestRateLimiting:
    """Test cases for rate limiting functionality."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_rate_limit_allows_requests_within_limit(self):
        """Test that requests within rate limit are allowed."""
        manager = APIKeyManager()
        test_key = "rate_limit_test_key_123456"
        manager.add_key(test_key, test_mode=True)
        
        # Make requests within limit
        for _ in range(5):
            assert manager.check_rate_limit(test_key, limit=10, window=60) is True
    
    def test_rate_limit_blocks_requests_over_limit(self):
        """Test that requests over rate limit are blocked."""
        manager = APIKeyManager()
        test_key = "rate_limit_test_key_123456"
        manager.add_key(test_key, test_mode=True)
        
        # Fill up the rate limit
        for _ in range(5):
            manager.check_rate_limit(test_key, limit=5, window=60)
        
        # Next request should be blocked
        assert manager.check_rate_limit(test_key, limit=5, window=60) is False
    
    def test_rate_limit_window_sliding(self):
        """Test that rate limit window slides properly."""
        manager = APIKeyManager()
        test_key = "rate_limit_test_key_123456"
        manager.add_key(test_key, test_mode=True)
        
        # Mock time to control window sliding
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000
            
            # Fill up rate limit
            for _ in range(3):
                manager.check_rate_limit(test_key, limit=3, window=10)
            
            # Move time forward beyond window
            mock_time.return_value = 1020
            
            # Should allow requests again
            assert manager.check_rate_limit(test_key, limit=3, window=10) is True
    
    def test_rate_limit_per_key_isolation(self):
        """Test that rate limits are isolated per key."""
        manager = APIKeyManager()
        key1 = "rate_limit_key1_123456"
        key2 = "rate_limit_key2_123456"
        
        manager.add_key(key1, test_mode=True)
        manager.add_key(key2, test_mode=True)
        
        # Exhaust rate limit for key1
        for _ in range(3):
            manager.check_rate_limit(key1, limit=3, window=60)
        
        # key2 should still be allowed
        assert manager.check_rate_limit(key2, limit=3, window=60) is True
        assert manager.check_rate_limit(key1, limit=3, window=60) is False


class TestKeyRotation:
    """Test cases for key rotation functionality."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_rotate_keys_adds_new_key(self):
        """Test that key rotation adds new keys."""
        manager = APIKeyManager()
        original_count = manager.get_active_key_count()
        
        new_key = "new_rotated_key_123456"
        manager.rotate_keys(new_key)
        
        assert manager.get_active_key_count() == original_count + 1
        assert new_key in manager._active_keys
    
    def test_rotation_info_tracking(self):
        """Test rotation info tracking."""
        manager = APIKeyManager()
        
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000
            
            key1 = "rotation_test_key1_123456"
            manager.add_key(key1, test_mode=True)
            
            # Move time forward and add another key
            mock_time.return_value = 1100
            key2 = "rotation_test_key2_123456"
            manager.rotate_keys(key2)
            
            info = manager.get_rotation_info()
            assert info["active_keys"] == 2
            assert info["revoked_keys"] == 0
            assert info["oldest_key_age_days"] > 0


class TestKeyRevocation:
    """Test cases for key revocation functionality."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_revoke_existing_key(self):
        """Test revoking an existing key."""
        manager = APIKeyManager()
        test_key = "key_to_revoke_123456"
        manager.add_key(test_key, test_mode=True)
        
        assert manager.is_valid_key(test_key) is True
        manager.revoke_key(test_key)
        assert manager.is_valid_key(test_key) is False
        assert test_key in manager._revoked_keys
    
    def test_revoke_nonexistent_key(self):
        """Test attempting to revoke a nonexistent key."""
        manager = APIKeyManager()
        # Should not raise an exception
        manager.revoke_key("nonexistent_key_123456")
    
    def test_revoked_key_count_tracking(self):
        """Test that revoked key count is tracked properly."""
        manager = APIKeyManager()
        test_key = "revocation_count_test_123456"
        manager.add_key(test_key, test_mode=True)
        
        initial_active = manager.get_active_key_count()
        manager.revoke_key(test_key)
        
        assert manager.get_active_key_count() == initial_active - 1
        
        info = manager.get_rotation_info()
        assert info["revoked_keys"] == 1
    
    def test_re_add_revoked_key(self):
        """Test re-adding a previously revoked key."""
        manager = APIKeyManager()
        test_key = "re_add_test_key_123456"
        manager.add_key(test_key, test_mode=True)
        manager.revoke_key(test_key)
        
        # Re-add the key
        manager.add_key(test_key, test_mode=True)
        
        assert manager.is_valid_key(test_key) is True
        assert test_key not in manager._revoked_keys


class TestThreadSafety:
    """Test cases for thread safety."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_concurrent_key_validation(self):
        """Test concurrent key validation operations."""
        manager = APIKeyManager()
        test_key = "concurrent_test_key_123456"
        manager.add_key(test_key, test_mode=True)
        
        results = []
        
        def validate_key():
            results.append(manager.is_valid_key(test_key))
        
        # Run multiple validations concurrently
        threads = [Thread(target=validate_key) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All validations should succeed
        assert all(results)
        assert len(results) == 10
    
    def test_concurrent_rate_limiting(self):
        """Test concurrent rate limiting operations."""
        manager = APIKeyManager()
        test_key = "concurrent_rate_limit_key_123456"
        manager.add_key(test_key, test_mode=True)
        
        results = []
        
        def check_rate_limit():
            results.append(manager.check_rate_limit(test_key, limit=5, window=60))
        
        # Run rate limit checks concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(check_rate_limit) for _ in range(10)]
            for future in futures:
                future.result()
        
        # Exactly 5 should succeed due to rate limit
        assert sum(results) == 5
    
    def test_concurrent_key_operations(self):
        """Test concurrent key add/revoke operations."""
        manager = APIKeyManager()
        
        def add_keys():
            for i in range(5):
                manager.add_key(f"concurrent_key_{i}_123456", test_mode=True)
        
        def revoke_keys():
            for i in range(5):
                manager.revoke_key(f"concurrent_key_{i}_123456")
        
        # Run operations concurrently
        threads = [Thread(target=add_keys), Thread(target=revoke_keys)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should not crash and maintain data consistency
        assert isinstance(manager.get_active_key_count(), int)


class TestHMACKeyHashing:
    """Test cases for HMAC key hashing."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_hash_key_consistency(self):
        """Test that key hashing is consistent."""
        manager = APIKeyManager()
        test_key = "hash_test_key_123456"
        
        hash1 = manager._hash_key(test_key)
        hash2 = manager._hash_key(test_key)
        
        assert hash1 == hash2
        assert len(hash1) == 8  # Truncated to 8 characters
    
    def test_hash_key_different_inputs(self):
        """Test that different keys produce different hashes."""
        manager = APIKeyManager()
        
        hash1 = manager._hash_key("key1_123456")
        hash2 = manager._hash_key("key2_123456")
        
        assert hash1 != hash2
    
    def test_hash_key_safe_for_logging(self):
        """Test that hashed keys are safe for logging."""
        manager = APIKeyManager()
        test_key = "sensitive_api_key_123456789"
        
        hashed = manager._hash_key(test_key)
        
        # Hash should not contain the original key
        assert test_key not in hashed
        assert len(hashed) == 8


class TestGlobalManagerFunctions:
    """Test cases for global manager functions."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_get_key_manager_singleton(self):
        """Test that get_key_manager returns singleton instance."""
        manager1 = get_key_manager()
        manager2 = get_key_manager()
        
        assert manager1 is manager2
    
    def test_setup_test_key_manager(self):
        """Test setup_test_key_manager function."""
        test_key = "test_setup_key_123456"
        setup_test_key_manager(test_key)
        
        manager = get_key_manager()
        assert manager.is_valid_key(test_key) is True
    
    def test_validate_api_key_function(self):
        """Test global validate_api_key function."""
        test_key = "global_validate_test_123456"
        setup_test_key_manager(test_key)
        
        assert validate_api_key(test_key) is True
        assert validate_api_key("invalid_key") is False
    
    def test_rotate_api_keys_function(self):
        """Test global rotate_api_keys function."""
        old_key = "old_key_123456"
        new_key = "new_key_123456"
        
        setup_test_key_manager(old_key)
        rotate_api_keys(new_key)
        
        manager = get_key_manager()
        assert manager.is_valid_key(old_key) is True  # Old key still valid
        assert manager.is_valid_key(new_key) is True  # New key added
    
    def test_revoke_api_key_function(self):
        """Test global revoke_api_key function."""
        test_key = "global_revoke_test_123456"
        setup_test_key_manager(test_key)
        
        assert validate_api_key(test_key) is True
        revoke_api_key(test_key)
        assert validate_api_key(test_key) is False


class TestErrorHandling:
    """Test cases for error handling and edge cases."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_empty_string_key_validation(self):
        """Test validation of empty string keys."""
        manager = APIKeyManager()
        assert manager.is_valid_key("") is False
    
    def test_none_key_validation(self):
        """Test validation of None keys."""
        manager = APIKeyManager()
        with pytest.raises(TypeError):
            manager.is_valid_key(None)
    
    def test_invalid_rate_limit_parameters(self):
        """Test rate limiting with invalid parameters."""
        manager = APIKeyManager()
        test_key = "rate_limit_edge_test_123456"
        manager.add_key(test_key, test_mode=True)
        
        # Zero limit should block all requests
        assert manager.check_rate_limit(test_key, limit=0, window=60) is False
        
        # Negative window should work (treated as immediate expiry)
        assert manager.check_rate_limit(test_key, limit=10, window=-1) is True
    
    def test_rotation_info_with_no_keys(self):
        """Test rotation info when no keys exist."""
        manager = APIKeyManager()
        info = manager.get_rotation_info()
        
        assert info["active_keys"] == 0
        assert info["revoked_keys"] == 0
        assert info["oldest_key_age_days"] == 0
        assert info["average_key_age_days"] == 0
        assert info["total_api_calls"] == 0
    
    def test_large_number_of_keys(self):
        """Test handling large numbers of keys."""
        manager = APIKeyManager()
        
        # Add many keys
        for i in range(100):
            manager.add_key(f"bulk_key_{i}_123456", test_mode=True)
        
        assert manager.get_active_key_count() == 100
        
        # Validate performance doesn't degrade significantly
        start_time = time.time()
        for i in range(100):
            manager.is_valid_key(f"bulk_key_{i}_123456")
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second)
        assert end_time - start_time < 1.0


class TestSecurityAspects:
    """Test cases focusing on security aspects."""
    
    def setup_method(self):
        """Reset state before each test."""
        global _key_manager
        with _manager_lock:
            _key_manager = None
    
    def test_secret_key_not_logged(self):
        """Test that secret keys are not logged."""
        with patch('lexgraph_legal_rag.auth.logger') as mock_logger:
            manager = APIKeyManager()
            test_key = "secret_test_key_123456"
            manager.add_key(test_key, test_mode=True)
            
            # Check that logged messages don't contain the actual key
            for call in mock_logger.info.call_args_list:
                message = str(call)
                assert test_key not in message
    
    def test_key_metadata_doesnt_store_plaintext(self):
        """Test that key metadata doesn't store plaintext keys."""
        manager = APIKeyManager()
        test_key = "metadata_test_key_123456"
        manager.add_key(test_key, test_mode=True)
        
        # Check metadata structure
        metadata = manager._key_metadata[test_key]
        
        # Metadata should contain expected fields but not expose the key
        assert "created_at" in metadata
        assert "is_primary" in metadata
        assert "usage_count" in metadata
        assert "last_used" in metadata
        
        # The key itself is the dictionary key, which is expected,
        # but metadata values shouldn't contain the plaintext key
        for value in metadata.values():
            if isinstance(value, str):
                assert test_key not in value
    
    def test_rate_limit_logging_uses_hashed_key(self):
        """Test that rate limit violations log hashed keys only."""
        with patch('lexgraph_legal_rag.auth.logger') as mock_logger:
            manager = APIKeyManager()
            test_key = "rate_limit_security_test_123456"
            manager.add_key(test_key, test_mode=True)
            
            # Trigger rate limit violation
            for _ in range(6):
                manager.check_rate_limit(test_key, limit=5, window=60)
            
            # Check that warning was logged with hashed key
            mock_logger.warning.assert_called()
            warning_message = str(mock_logger.warning.call_args)
            assert test_key not in warning_message
            assert "Rate limit exceeded" in warning_message