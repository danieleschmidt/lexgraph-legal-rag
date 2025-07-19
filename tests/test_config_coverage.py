"""Test coverage for config module."""

import os
import pytest
from unittest.mock import patch
from lexgraph_legal_rag.config import Config, ConfigurationError, validate_environment


def test_config_with_all_env_vars():
    """Test config creation with all environment variables set."""
    with patch.dict(os.environ, {
        'API_KEY': 'test_api_key_12345',
        'OPENAI_API_KEY': 'sk-test123',
        'PINECONE_API_KEY': 'pinecone-test',
        'LEGAL_CORPUS_PATH': '/test/path'
    }):
        config = Config()
        assert config.api_key == 'test_api_key_12345'
        assert config.openai_api_key == 'sk-test123'
        assert config.pinecone_api_key == 'pinecone-test'
        assert config.legal_corpus_path == '/test/path'


def test_config_missing_required_api_key():
    """Test config raises error when API_KEY is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ConfigurationError, match="Required environment variable API_KEY is not set"):
            Config()


def test_config_with_defaults():
    """Test config with optional variables using defaults."""
    with patch.dict(os.environ, {'API_KEY': 'test_key_12345'}, clear=True):
        config = Config()
        assert config.api_key == 'test_key_12345'
        assert config.openai_api_key == ''
        assert config.pinecone_api_key == ''
        assert config.legal_corpus_path == './data/legal_docs'


def test_config_validate_startup_short_api_key():
    """Test validation fails with short API key."""
    with patch.dict(os.environ, {'API_KEY': 'short'}, clear=True):
        config = Config()
        with pytest.raises(ConfigurationError, match="API_KEY must be at least 8 characters long"):
            config.validate_startup()


def test_config_validate_startup_development_key():
    """Test validation warns about development keys."""
    with patch.dict(os.environ, {'API_KEY': 'mysecret'}, clear=True):
        config = Config()
        # This should complete without raising an exception
        config.validate_startup()  # Should complete validation despite warning


def test_config_validate_startup_invalid_openai_key():
    """Test validation warns about invalid OpenAI key format."""
    with patch.dict(os.environ, {
        'API_KEY': 'test_key_12345',
        'OPENAI_API_KEY': 'invalid-format'
    }, clear=True):
        config = Config()
        config.validate_startup()  # Should complete without error


def test_config_to_dict():
    """Test config to_dict method."""
    with patch.dict(os.environ, {
        'API_KEY': 'test_key_12345',
        'OPENAI_API_KEY': 'sk-test123',
        'LEGAL_CORPUS_PATH': '/custom/path'
    }, clear=True):
        config = Config()
        result = config.to_dict()
        expected = {
            'api_key_set': True,
            'openai_api_key_set': True,
            'pinecone_api_key_set': False,
            'legal_corpus_path': '/custom/path'
        }
        assert result == expected


def test_validate_environment_success():
    """Test validate_environment with valid config."""
    with patch.dict(os.environ, {'API_KEY': 'test_key_12345'}, clear=True):
        config = validate_environment(allow_test_mode=True)
        assert isinstance(config, Config)
        assert config.api_key == 'test_key_12345'


def test_validate_environment_test_mode():
    """Test validate_environment in test mode."""
    with patch.dict(os.environ, {}, clear=True):
        config = validate_environment(allow_test_mode=True)
        assert isinstance(config, Config)
        assert config.api_key == 'test_api_key_12345'