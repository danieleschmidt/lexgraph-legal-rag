"""Tests for logging configuration."""

import logging
from unittest.mock import patch, Mock

import structlog

from lexgraph_legal_rag.logging_config import configure_logging


def test_configure_logging_default_level():
    """Test logging configuration with default level."""
    with patch('lexgraph_legal_rag.logging_config.logging.basicConfig') as mock_basic_config, \
         patch('lexgraph_legal_rag.logging_config.structlog.configure') as mock_structlog_config:
        
        configure_logging()
        
        # Verify basicConfig called with default INFO level
        mock_basic_config.assert_called_once_with(level=logging.INFO, format="%(message)s")
        
        # Verify structlog configure called
        mock_structlog_config.assert_called_once()
        
        # Check the structlog configuration arguments
        config_args = mock_structlog_config.call_args[1]
        assert 'processors' in config_args
        assert 'wrapper_class' in config_args
        assert 'logger_factory' in config_args
        assert 'context_class' in config_args
        assert config_args['cache_logger_on_first_use'] is True


def test_configure_logging_custom_level():
    """Test logging configuration with custom level."""
    with patch('lexgraph_legal_rag.logging_config.logging.basicConfig') as mock_basic_config, \
         patch('lexgraph_legal_rag.logging_config.structlog.configure') as mock_structlog_config:
        
        configure_logging(level=logging.DEBUG)
        
        # Verify basicConfig called with DEBUG level
        mock_basic_config.assert_called_once_with(level=logging.DEBUG, format="%(message)s")
        
        # Verify structlog configure called
        mock_structlog_config.assert_called_once()


def test_configure_logging_processors():
    """Test that required processors are configured."""
    with patch('lexgraph_legal_rag.logging_config.logging.basicConfig'), \
         patch('lexgraph_legal_rag.logging_config.structlog.configure') as mock_structlog_config:
        
        configure_logging()
        
        # Get the processors argument
        config_args = mock_structlog_config.call_args[1]
        processors = config_args['processors']
        
        # Should have 4 processors
        assert len(processors) == 4
        
        # Verify processor types (checking class names since they're instances)
        processor_types = [type(p).__name__ for p in processors]
        assert 'CorrelationIdProcessor' in processor_types
        assert 'TimeStamper' in processor_types
        assert 'JSONRenderer' in processor_types


def test_configure_logging_logger_factory():
    """Test that PrintLoggerFactory is configured."""
    with patch('lexgraph_legal_rag.logging_config.logging.basicConfig'), \
         patch('lexgraph_legal_rag.logging_config.structlog.configure') as mock_structlog_config:
        
        configure_logging()
        
        config_args = mock_structlog_config.call_args[1]
        logger_factory = config_args['logger_factory']
        
        # Should be PrintLoggerFactory instance
        assert type(logger_factory).__name__ == 'PrintLoggerFactory'


def test_configure_logging_integration():
    """Test actual logging configuration (integration test)."""
    # This test actually configures logging to verify it works
    configure_logging(level=logging.WARNING)
    
    # Get a structured logger
    logger = structlog.get_logger("test")
    
    # This should not raise an exception
    assert logger is not None
    
    # Test that we can log without errors
    try:
        logger.info("Test message", test_key="test_value")
        # If we get here, logging works
        assert True
    except Exception:
        assert False, "Logging configuration failed"