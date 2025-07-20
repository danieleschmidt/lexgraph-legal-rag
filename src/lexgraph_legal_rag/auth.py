"""API key management and rotation."""

from __future__ import annotations

import os
import time
import hashlib
import logging
from typing import Set, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages API key rotation and validation."""
    
    def __init__(self) -> None:
        self._active_keys: Set[str] = set()
        self._revoked_keys: Set[str] = set()
        self._lock = Lock()
        self._last_rotation = time.time()
        
        # Load initial key from environment
        primary_key = os.environ.get("API_KEY")
        if primary_key:
            self._active_keys.add(primary_key)
        
        # Load additional keys for rotation if available
        for i in range(1, 10):  # Support up to 9 additional keys
            additional_key = os.environ.get(f"API_KEY_{i}")
            if additional_key:
                self._active_keys.add(additional_key)
    
    def is_valid_key(self, api_key: str) -> bool:
        """Check if the API key is valid and not revoked."""
        with self._lock:
            return api_key in self._active_keys and api_key not in self._revoked_keys
    
    def add_key(self, api_key: str) -> None:
        """Add a new API key to the active set."""
        with self._lock:
            self._active_keys.add(api_key)
            # Remove from revoked if it was previously revoked
            self._revoked_keys.discard(api_key)
            logger.info(f"Added new API key (hash: {self._hash_key(api_key)})")
    
    def revoke_key(self, api_key: str) -> None:
        """Revoke an API key."""
        with self._lock:
            if api_key in self._active_keys:
                self._revoked_keys.add(api_key)
                logger.warning(f"Revoked API key (hash: {self._hash_key(api_key)})")
            else:
                logger.warning(f"Attempted to revoke unknown API key (hash: {self._hash_key(api_key)})")
    
    def rotate_keys(self, new_primary_key: str) -> None:
        """Rotate API keys - add new primary and optionally revoke old ones."""
        with self._lock:
            old_keys = self._active_keys.copy()
            self.add_key(new_primary_key)
            self._last_rotation = time.time()
            
            logger.info(f"Key rotation completed. Total active keys: {len(self._active_keys)}")
            
            # Optional: Auto-revoke old keys after grace period
            # This would be implemented based on business requirements
    
    def get_active_key_count(self) -> int:
        """Get the number of active (non-revoked) keys."""
        with self._lock:
            return len(self._active_keys - self._revoked_keys)
    
    def get_rotation_info(self) -> dict:
        """Get information about key rotation status."""
        with self._lock:
            return {
                "active_keys": len(self._active_keys),
                "revoked_keys": len(self._revoked_keys),
                "last_rotation": self._last_rotation,
                "days_since_rotation": (time.time() - self._last_rotation) / 86400
            }
    
    def _hash_key(self, api_key: str) -> str:
        """Create a safe hash of the API key for logging."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:8]


# Global instance
_key_manager: Optional[APIKeyManager] = None
_manager_lock = Lock()


def get_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _key_manager
    with _manager_lock:
        if _key_manager is None:
            _key_manager = APIKeyManager()
        return _key_manager


def setup_test_key_manager(test_api_key: str) -> None:
    """Setup key manager for testing with a specific API key."""
    global _key_manager
    with _manager_lock:
        _key_manager = APIKeyManager()
        _key_manager.add_key(test_api_key)


def validate_api_key(api_key: str) -> bool:
    """Validate an API key using the global key manager."""
    return get_key_manager().is_valid_key(api_key)


def rotate_api_keys(new_primary_key: str) -> None:
    """Rotate API keys using the global key manager."""
    get_key_manager().rotate_keys(new_primary_key)


def revoke_api_key(api_key: str) -> None:
    """Revoke an API key using the global key manager."""
    get_key_manager().revoke_key(api_key)