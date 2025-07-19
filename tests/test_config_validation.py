"""Tests for configuration validation."""

import os
import pytest
from unittest.mock import patch

from lexgraph_legal_rag.config import Config, ConfigurationError, validate_environment


def test_config_with_valid_api_key():
    """Test configuration with valid API key."""
    with patch.dict(os.environ, {"API_KEY": "valid-api-key-123"}):
        config = Config()
        assert config.api_key == "valid-api-key-123"
        
        # Should not raise
        config.validate_startup()


def test_config_missing_required_api_key():
    """Test configuration fails without required API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ConfigurationError, match="Required environment variable API_KEY is not set"):
            Config()


def test_config_with_short_api_key():
    """Test configuration fails with short API key."""
    with patch.dict(os.environ, {"API_KEY": "short"}):
        config = Config()
        with pytest.raises(ConfigurationError, match="API_KEY must be at least 8 characters long"):
            config.validate_startup()


def test_config_warns_about_development_keys():
    """Test configuration warns about development API keys."""
    development_keys = ["test", "development", "dev", "mysecret"]
    
    for dev_key in development_keys:
        with patch.dict(os.environ, {"API_KEY": dev_key}):
            config = Config()
            
            # Should warn but not fail
            with patch('lexgraph_legal_rag.config.logger') as mock_logger:
                config.validate_startup()
                mock_logger.warning.assert_called_with(
                    "Using development API key - not suitable for production"
                )


def test_config_with_optional_keys():
    """Test configuration with optional environment variables."""
    env_vars = {
        "API_KEY": "valid-key-123",
        "OPENAI_API_KEY": "sk-test123",
        "PINECONE_API_KEY": "pinecone-123",
        "LEGAL_CORPUS_PATH": "/custom/path"
    }
    
    with patch.dict(os.environ, env_vars):
        config = Config()
        
        assert config.openai_api_key == "sk-test123"
        assert config.pinecone_api_key == "pinecone-123"
        assert config.legal_corpus_path == "/custom/path"


def test_config_invalid_openai_key_format():
    """Test warning for invalid OpenAI API key format."""
    with patch.dict(os.environ, {
        "API_KEY": "valid-key-123",
        "OPENAI_API_KEY": "invalid-format"
    }):
        config = Config()
        
        with patch('lexgraph_legal_rag.config.logger') as mock_logger:
            config.validate_startup()
            mock_logger.warning.assert_called_with(
                "OPENAI_API_KEY does not appear to be valid format"
            )


def test_config_to_dict():
    """Test configuration dictionary representation."""
    with patch.dict(os.environ, {
        "API_KEY": "secret-key",
        "OPENAI_API_KEY": "sk-test",
        "LEGAL_CORPUS_PATH": "/test/path"
    }):
        config = Config()
        config_dict = config.to_dict()
        
        # Should not expose actual keys
        assert config_dict["api_key_set"] is True
        assert config_dict["openai_api_key_set"] is True
        assert config_dict["pinecone_api_key_set"] is False
        assert config_dict["legal_corpus_path"] == "/test/path"
        
        # Should not contain actual secret values
        assert "secret-key" not in str(config_dict)
        assert "sk-test" not in str(config_dict)


def test_validate_environment_success():
    """Test successful environment validation."""
    with patch.dict(os.environ, {"API_KEY": "valid-key-123"}):
        config = validate_environment()
        assert isinstance(config, Config)
        assert config.api_key == "valid-key-123"


def test_validate_environment_failure_exits():
    """Test that validation failure exits the program."""
    with patch.dict(os.environ, {}, clear=True):
        with patch('sys.exit') as mock_exit:
            with patch('lexgraph_legal_rag.config.logger') as mock_logger:
                validate_environment()
                
                mock_logger.error.assert_called()
                mock_exit.assert_called_with(1)


def test_config_defaults():
    """Test configuration defaults."""
    with patch.dict(os.environ, {"API_KEY": "valid-key-123"}):
        config = Config()
        
        # Test defaults
        assert config.openai_api_key == ""
        assert config.pinecone_api_key == ""
        assert config.legal_corpus_path == "./data/legal_docs"