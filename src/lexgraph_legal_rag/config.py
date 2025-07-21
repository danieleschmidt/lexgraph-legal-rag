"""Configuration validation and management."""

from __future__ import annotations

import os
import sys
from typing import Any
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class Config:
    """Application configuration with validation."""

    def __init__(self) -> None:
        self.api_key = self._get_required_env("API_KEY")
        self.openai_api_key = self._get_optional_env("OPENAI_API_KEY")
        self.pinecone_api_key = self._get_optional_env("PINECONE_API_KEY")
        self.legal_corpus_path = self._get_optional_env("LEGAL_CORPUS_PATH", "./data/legal_docs")
        
        # Security configuration
        self.allowed_origins = self._get_cors_origins()
        self.require_https = self._get_optional_env("REQUIRE_HTTPS", "false").lower() == "true"
        self.max_key_age_days = int(self._get_optional_env("MAX_KEY_AGE_DAYS", "90"))
        
    def _get_cors_origins(self) -> list[str]:
        """Get CORS allowed origins from environment."""
        origins_env = self._get_optional_env("CORS_ALLOWED_ORIGINS", "")
        if not origins_env:
            # Default to localhost for development
            return ["http://localhost:3000", "http://localhost:8080", "http://localhost:8501"]
        
        # Parse comma-separated origins
        origins = [origin.strip() for origin in origins_env.split(",")]
        return [origin for origin in origins if origin]  # Filter empty strings
        
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise ConfigurationError."""
        value = os.environ.get(key)
        if not value:
            raise ConfigurationError(f"Required environment variable {key} is not set")
        return value
    
    def _get_optional_env(self, key: str, default: str = "") -> str:
        """Get optional environment variable with default."""
        return os.environ.get(key, default)
    
    def validate_startup(self) -> None:
        """Validate all configuration at startup."""
        logger.info("Validating configuration...")
        
        # Check for development/test keys first
        development_keys = ["test", "development", "dev", "mysecret"]
        is_development_key = self.api_key.lower() in development_keys
        
        if is_development_key:
            logger.warning("Using development API key - not suitable for production")
        else:
            # Validate API key format and length for non-development keys
            if len(self.api_key) < 16:
                raise ConfigurationError("API_KEY must be at least 16 characters long for production")
            
            # Validate key complexity
            if not self._has_sufficient_entropy(self.api_key):
                logger.warning("API_KEY has low entropy - consider using a more complex key")
        
        # Validate CORS origins
        if "*" in self.allowed_origins:
            logger.warning("CORS allows all origins (*) - not recommended for production")
        
        # Validate optional configurations if present
        if self.openai_api_key and not self.openai_api_key.startswith("sk-"):
            logger.warning("OPENAI_API_KEY does not appear to be valid format")
        
        # Validate security settings
        if self.require_https:
            logger.info("HTTPS enforcement enabled")
        
        logger.info("Configuration validation completed")
    
    def _has_sufficient_entropy(self, key: str) -> bool:
        """Check if API key has sufficient entropy."""
        import string
        has_upper = any(c in string.ascii_uppercase for c in key)
        has_lower = any(c in string.ascii_lowercase for c in key)
        has_digit = any(c in string.digits for c in key)
        has_special = any(c in string.punctuation for c in key)
        
        complexity_score = sum([has_upper, has_lower, has_digit, has_special])
        return complexity_score >= 3
    
    def to_dict(self) -> dict[str, Any]:
        """Return configuration as dictionary (excluding sensitive values)."""
        return {
            "api_key_set": bool(self.api_key),
            "openai_api_key_set": bool(self.openai_api_key),
            "pinecone_api_key_set": bool(self.pinecone_api_key),
            "legal_corpus_path": self.legal_corpus_path,
            "allowed_origins": self.allowed_origins,
            "require_https": self.require_https,
            "max_key_age_days": self.max_key_age_days,
        }


def validate_environment(allow_test_mode: bool = False) -> Config:
    """Validate environment configuration at startup."""
    try:
        config = Config()
        config.validate_startup()
        return config
    except ConfigurationError as e:
        if allow_test_mode:
            logger.warning(f"Configuration validation failed in test mode: {e}")
            # Create a test config with dummy values
            os.environ.setdefault("API_KEY", "test_api_key_12345")
            config = Config()
            config.validate_startup()
            return config
        else:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)