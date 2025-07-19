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
        
        # Validate API key format and length
        if len(self.api_key) < 8:
            raise ConfigurationError("API_KEY must be at least 8 characters long")
        
        # Check for development/test keys in production
        if self.api_key.lower() in ["test", "development", "dev", "mysecret"]:
            logger.warning("Using development API key - not suitable for production")
        
        # Validate optional configurations if present
        if self.openai_api_key and not self.openai_api_key.startswith("sk-"):
            logger.warning("OPENAI_API_KEY does not appear to be valid format")
        
        logger.info("Configuration validation completed")
    
    def to_dict(self) -> dict[str, Any]:
        """Return configuration as dictionary (excluding sensitive values)."""
        return {
            "api_key_set": bool(self.api_key),
            "openai_api_key_set": bool(self.openai_api_key),
            "pinecone_api_key_set": bool(self.pinecone_api_key),
            "legal_corpus_path": self.legal_corpus_path,
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