"""API key management and rotation."""

from __future__ import annotations

import os
import time
import hashlib
import logging
import hmac
import secrets
from collections import defaultdict, deque
from typing import Set, Optional, Dict
from threading import Lock

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages API key rotation, validation, and rate limiting."""
    
    def __init__(self) -> None:
        self._active_keys: Set[str] = set()
        self._revoked_keys: Set[str] = set()
        self._key_metadata: Dict[str, dict] = {}  # Store creation time, usage stats
        self._key_rate_limits: Dict[str, deque] = defaultdict(lambda: deque())
        self._lock = Lock()
        self._last_rotation = time.time()
        self._secret_key = self._get_or_create_secret_key()
        
        # Load initial key from environment
        primary_key = os.environ.get("API_KEY")
        if primary_key:
            self._add_key_with_metadata(primary_key, is_primary=True)
        
        # Load additional keys for rotation if available
        for i in range(1, 10):  # Support up to 9 additional keys
            additional_key = os.environ.get(f"API_KEY_{i}")
            if additional_key:
                self._add_key_with_metadata(additional_key, is_primary=False)
    
    def _get_or_create_secret_key(self) -> bytes:
        """Get or create HMAC secret key for API key hashing."""
        secret_env = os.environ.get("API_KEY_SECRET")
        if secret_env:
            return secret_env.encode()
        
        # Generate random secret for this session
        # In production, this should be persisted securely
        return secrets.token_bytes(32)
    
    def _add_key_with_metadata(self, api_key: str, is_primary: bool = False) -> None:
        """Add key with metadata tracking."""
        self._active_keys.add(api_key)
        self._key_metadata[api_key] = {
            "created_at": time.time(),
            "is_primary": is_primary,
            "usage_count": 0,
            "last_used": None
        }
    
    def is_valid_key(self, api_key: str) -> bool:
        """Check if the API key is valid and not revoked."""
        with self._lock:
            if api_key not in self._active_keys or api_key in self._revoked_keys:
                return False
            
            # Update usage statistics
            if api_key in self._key_metadata:
                self._key_metadata[api_key]["usage_count"] += 1
                self._key_metadata[api_key]["last_used"] = time.time()
            
            return True
    
    def check_rate_limit(self, api_key: str, limit: int = 60, window: int = 60) -> bool:
        """Check per-key rate limit. Returns True if within limit."""
        with self._lock:
            now = time.time()
            key_requests = self._key_rate_limits[api_key]
            
            # Remove old requests outside the window
            while key_requests and now - key_requests[0] > window:
                key_requests.popleft()
            
            # Check if within limit
            if len(key_requests) >= limit:
                logger.warning(f"Rate limit exceeded for API key {self._hash_key(api_key)}")
                return False
            
            # Add current request
            key_requests.append(now)
            return True
    
    def add_key(self, api_key: str) -> None:
        """Add a new API key to the active set."""
        with self._lock:
            # Validate key strength
            if not self._validate_key_strength(api_key):
                raise ValueError("API key does not meet security requirements")
            
            self._add_key_with_metadata(api_key, is_primary=False)
            # Remove from revoked if it was previously revoked
            self._revoked_keys.discard(api_key)
            logger.info(f"Added new API key (hash: {self._hash_key(api_key)})")
    
    def _validate_key_strength(self, api_key: str) -> bool:
        """Validate API key meets security requirements."""
        if len(api_key) < 16:
            return False
        
        # Check for sufficient entropy
        import string
        has_upper = any(c in string.ascii_uppercase for c in api_key)
        has_lower = any(c in string.ascii_lowercase for c in api_key)
        has_digit = any(c in string.digits for c in api_key)
        
        return sum([has_upper, has_lower, has_digit]) >= 2
    
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
            now = time.time()
            key_ages = []
            usage_stats = []
            
            for key, metadata in self._key_metadata.items():
                if key in self._active_keys and key not in self._revoked_keys:
                    key_age_days = (now - metadata["created_at"]) / 86400
                    key_ages.append(key_age_days)
                    usage_stats.append(metadata["usage_count"])
            
            return {
                "active_keys": len(self._active_keys - self._revoked_keys),
                "revoked_keys": len(self._revoked_keys),
                "last_rotation": self._last_rotation,
                "days_since_rotation": (time.time() - self._last_rotation) / 86400,
                "oldest_key_age_days": max(key_ages) if key_ages else 0,
                "average_key_age_days": sum(key_ages) / len(key_ages) if key_ages else 0,
                "total_api_calls": sum(usage_stats),
                "keys_needing_rotation": sum(1 for age in key_ages if age > 90)
            }
    
    def _hash_key(self, api_key: str) -> str:
        """Create a safe HMAC hash of the API key for logging."""
        return hmac.new(
            self._secret_key,
            api_key.encode(),
            hashlib.sha256
        ).hexdigest()[:8]


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